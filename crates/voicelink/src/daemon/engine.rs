//! The engine: wires audio events, STT, the concierge, the speaker and the
//! controller link together. Everything it needs is injected, so tests drive
//! it with fakes and synthetic audio events.

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use voicelink_backends::{AudioClip, ChatModel, SpeechToText, TextToSpeech};

use super::speaker::{self, SpeakerHandle};
use super::status_file::StatusHandle;
use crate::audio::{AudioEvent, AudioSink};
use crate::concierge::{Concierge, Input};
use crate::config;
use crate::link::{LinkHandle, LinkNotice};
use crate::status::LinkState;

/// Everything the engine runs on.
pub(crate) struct Parts {
    pub stt: Arc<dyn SpeechToText>,
    pub llm: Arc<dyn ChatModel>,
    pub tts: Arc<dyn TextToSpeech>,
    pub audio_events: mpsc::UnboundedReceiver<AudioEvent>,
    pub sink: Arc<dyn AudioSink>,
    pub link: LinkHandle,
    pub notices: mpsc::UnboundedReceiver<LinkNotice>,
    pub audio: config::Audio,
    pub concierge: config::Concierge,
    pub status: StatusHandle,
}

/// Drops transcripts that carry no words (Whisper's `[BLANK_AUDIO]`, `(silence)`, lone punctuation).
pub(crate) fn clean_transcript(raw: &str) -> Option<String> {
    let mut out = String::new();
    let mut depth = 0usize;
    for c in raw.chars() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    let out = out.trim().to_string();
    out.chars().any(char::is_alphanumeric).then_some(out)
}

async fn stt_worker(
    stt: Arc<dyn SpeechToText>,
    mut clips: mpsc::UnboundedReceiver<(AudioClip, Instant)>,
    inbox: mpsc::UnboundedSender<Input>,
    status: StatusHandle,
) {
    while let Some((clip, ended)) = clips.recv().await {
        match stt.transcribe(&clip).await {
            Ok(t) => {
                let ms = ended.elapsed().as_millis().min(u128::from(u32::MAX)) as u32;
                status.update(|s| s.last_stt_ms = Some(ms));
                match clean_transcript(&t.text) {
                    Some(text) => {
                        tracing::info!("heard: {text}");
                        status.mark_transcript();
                        if inbox.send(Input::Transcript(text)).is_err() {
                            return;
                        }
                    }
                    None => tracing::debug!("ignored empty transcript {:?}", t.text),
                }
            }
            Err(e) => {
                tracing::warn!("speech-to-text failed: {e}");
                status.update(|s| s.last_error = Some(format!("speech-to-text: {e}")));
            }
        }
    }
}

fn apply_notice(status: &StatusHandle, n: &LinkNotice, ever_connected: &mut bool) {
    match n {
        LinkNotice::Connected(f) => {
            *ever_connected = true;
            let label = format!("{}:{} {}", f.window, f.window_name, f.command);
            status.update(|s| {
                s.state = LinkState::Listening;
                s.focus = Some(label);
                s.last_error = None;
            });
        }
        LinkNotice::Focus(f) => {
            let label = format!("{}:{} {}", f.window, f.window_name, f.command);
            status.update(|s| s.focus = Some(label));
        }
        LinkNotice::Disconnected(why) => {
            let state = if *ever_connected {
                LinkState::Reconnecting
            } else {
                LinkState::Connecting
            };
            let why = why.clone();
            status.update(|s| {
                s.state = state;
                s.last_error = Some(why);
            });
        }
        LinkNotice::Fatal(why) => {
            let why = why.clone();
            status.update(|s| {
                s.state = LinkState::Reconnecting;
                s.last_error = Some(why);
            });
        }
        LinkNotice::Event(_) => {}
    }
}

/// Runs until `shutdown` resolves (or the audio stream ends).
pub(crate) async fn run(parts: Parts, shutdown: impl Future<Output = ()>) {
    let Parts {
        stt,
        llm,
        tts,
        mut audio_events,
        sink,
        link,
        mut notices,
        audio,
        concierge,
        status,
    } = parts;

    let (say_tx, say_rx) = mpsc::unbounded_channel();
    let (speaker, speaker_task) = speaker::spawn(tts, Arc::clone(&sink), say_rx, status.clone());
    let conc = Concierge::new(llm, link, say_tx, concierge);
    let (inbox_tx, inbox_rx) = mpsc::unbounded_channel();
    let conc_task = tokio::spawn(Arc::clone(&conc).run(inbox_rx));
    let (clip_tx, clip_rx) = mpsc::unbounded_channel();
    let stt_task = tokio::spawn(stt_worker(stt, clip_rx, inbox_tx.clone(), status.clone()));

    status.update(|s| s.state = LinkState::Connecting);
    let mut ever_connected = false;
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            ev = audio_events.recv() => match ev {
                Some(AudioEvent::SpeechStarted) => barge_in(&audio, &speaker),
                Some(AudioEvent::Utterance(clip)) => {
                    let _ = clip_tx.send((clip, Instant::now()));
                }
                None => {
                    tracing::error!("audio stream ended");
                    status.update(|s| s.last_error = Some("audio stream ended".into()));
                    break;
                }
            },
            n = notices.recv() => match n {
                Some(n) => {
                    apply_notice(&status, &n, &mut ever_connected);
                    if inbox_tx.send(Input::Link(n)).is_err() { break; }
                }
                None => break,
            },
        }
    }
    drop((clip_tx, inbox_tx));
    stt_task.abort();
    conc_task.abort();
    speaker_task.abort();
    sink.stop();
}

/// SpeechStarted while we are talking: stop talking.
fn barge_in(audio: &config::Audio, speaker: &SpeakerHandle) {
    if audio.barge_in && speaker.is_active() {
        tracing::info!("barge-in: stopping speech");
        speaker.cancel();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use async_trait::async_trait;
    use serde_json::json;
    use voicelink_backends::{BackendError, Result, Transcript};
    use voicelink_proto::{AgentEvent, AgentKind, ClientMessage, EventKind, ServerMessage};

    use super::super::speaker::fake::{FakeSink, FakeTts};
    use super::*;
    use crate::concierge::fake::{ScriptedLlm, call, calls, say};
    use crate::link::fake::{FakeController, fast_config, focus};
    use crate::link::spawn as spawn_link;
    use crate::status::BackendNames;

    /// Returns the next scripted transcript for each clip, whatever it contains.
    struct FakeStt(Mutex<Vec<String>>);
    #[async_trait]
    impl SpeechToText for FakeStt {
        async fn transcribe(&self, _clip: &AudioClip) -> Result<Transcript> {
            let mut q = self.0.lock().unwrap();
            if q.is_empty() {
                return Err(BackendError::Backend("no more transcripts".into()));
            }
            Ok(Transcript { text: q.remove(0) })
        }
        fn name(&self) -> String {
            "fake-stt".into()
        }
    }

    struct Rig {
        fc: FakeController,
        llm: Arc<ScriptedLlm>,
        tts: Arc<FakeTts>,
        sink: Arc<FakeSink>,
        status: StatusHandle,
        audio_tx: mpsc::UnboundedSender<AudioEvent>,
        stop: Option<tokio::sync::oneshot::Sender<()>>,
        task: tokio::task::JoinHandle<()>,
    }

    fn start(transcripts: &[&str], audio: config::Audio) -> Rig {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        let (link, notices, _t) = spawn_link(fc.clone(), fast_config());
        let llm = Arc::new(ScriptedLlm::default());
        let tts = Arc::new(FakeTts::default());
        let sink = Arc::new(FakeSink::default());
        let status = StatusHandle::new(
            None,
            BackendNames {
                stt: "s".into(),
                llm: "l".into(),
                tts: "t".into(),
            },
        );
        let (audio_tx, audio_events) = mpsc::unbounded_channel();
        let (stop, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let parts = Parts {
            stt: Arc::new(FakeStt(Mutex::new(
                transcripts.iter().map(|s| s.to_string()).collect(),
            ))),
            llm: llm.clone(),
            tts: tts.clone(),
            audio_events,
            sink: sink.clone(),
            link,
            notices,
            audio,
            concierge: config::Concierge::default(),
            status: status.clone(),
        };
        let task = tokio::spawn(run(parts, async move {
            let _ = stop_rx.await;
        }));
        Rig {
            fc,
            llm,
            tts,
            sink,
            status,
            audio_tx,
            stop: Some(stop),
            task,
        }
    }

    fn utterance() -> AudioEvent {
        AudioEvent::Utterance(AudioClip {
            sample_rate: 16_000,
            samples: vec![0.1; 1600],
        })
    }

    async fn until(what: &str, mut f: impl FnMut() -> bool) {
        for _ in 0..300 {
            if f() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for {what}");
    }

    fn spoken(r: &Rig) -> Vec<String> {
        r.tts.spoken.lock().unwrap().clone()
    }

    impl Rig {
        async fn shutdown(mut self) {
            let _ = self.stop.take().unwrap().send(());
            tokio::time::timeout(Duration::from_secs(2), &mut self.task)
                .await
                .unwrap()
                .unwrap();
        }
    }

    #[tokio::test]
    async fn utterance_to_transcript_to_tool_call_to_send_to_ack_to_spoken_confirmation() {
        let r = start(&["fix the build please"], config::Audio::default());
        until("connected", || {
            r.status.snapshot().state == LinkState::Listening
        })
        .await;
        assert_eq!(r.status.snapshot().focus.as_deref(), Some("1:main claude"));
        r.llm.push(calls(vec![call(
            "c1",
            "send_to_focused",
            json!({"text": "Fix the build."}),
        )]));
        r.llm.push(say("Asked Claude to fix the build."));
        r.audio_tx.send(utterance()).unwrap();
        until("spoken confirmation", || {
            spoken(&r).contains(&"Asked Claude to fix the build.".to_string())
        })
        .await;
        assert!(
            r.fc.log()
                .iter()
                .any(|m| matches!(m, ClientMessage::Send { text, .. } if text == "Fix the build."))
        );
        assert_eq!(spoken(&r)[0], "Sending to Claude: Fix the build.");
        let s = r.status.snapshot();
        assert!(
            s.last_stt_ms.is_some() && s.last_reply_ms.is_some(),
            "{s:?}"
        );
        r.shutdown().await;
    }

    #[tokio::test]
    async fn blank_transcripts_are_ignored() {
        let r = start(
            &["[BLANK_AUDIO]", "(silence)", " . "],
            config::Audio::default(),
        );
        for _ in 0..3 {
            r.audio_tx.send(utterance()).unwrap();
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(r.llm.request_count(), 0);
        r.shutdown().await;
    }

    #[tokio::test]
    async fn stop_utterance_interrupts_without_consulting_the_llm() {
        let r = start(&["Stop."], config::Audio::default());
        until("connected", || r.fc.connects.load(Ordering::Relaxed) == 1).await;
        r.audio_tx.send(utterance()).unwrap();
        until("interrupt", || {
            r.fc.log()
                .iter()
                .any(|m| matches!(m, ClientMessage::Interrupt { .. }))
        })
        .await;
        assert_eq!(r.llm.request_count(), 0);
        until("spoken", || spoken(&r) == vec!["Interrupted.".to_string()]).await;
        r.shutdown().await;
    }

    #[tokio::test]
    async fn background_prompt_then_approve_selects_before_keys() {
        let r = start(&["approve"], config::Audio::default());
        until("connected", || {
            r.status.snapshot().state == LinkState::Listening
        })
        .await;
        r.fc.script
            .lock()
            .unwrap()
            .pane_text
            .insert("%5".into(), "Run tests?\n1. Yes\n2. No\n".into());
        r.fc.push(ServerMessage::Event(AgentEvent {
            ts_ms: 1,
            pane: "%5".into(),
            window: Some(2),
            agent: AgentKind::Codex,
            kind: EventKind::Prompt,
            text: Some("Run tests?".into()),
        }));
        until("prompt narrated", || {
            spoken(&r)
                .iter()
                .any(|s| s.contains("Run tests?") && s.contains("option 2: No"))
        })
        .await;
        r.llm.push(calls(vec![
            call("a", "focus_pane", json!({"pane": "%5"})),
            call("b", "press_keys", json!({"keys": ["1"]})),
        ]));
        r.llm.push(say("Approved."));
        r.audio_tx.send(utterance()).unwrap();
        until("keys", || {
            r.fc.log()
                .iter()
                .any(|m| matches!(m, ClientMessage::Keys { .. }))
        })
        .await;
        let log = r.fc.log();
        let sel = log
            .iter()
            .position(|m| matches!(m, ClientMessage::SelectPane { .. }))
            .unwrap();
        let keys = log
            .iter()
            .position(|m| matches!(m, ClientMessage::Keys { .. }))
            .unwrap();
        assert!(sel < keys);
        r.shutdown().await;
    }

    #[tokio::test]
    async fn focused_turn_end_is_summarised_aloud() {
        let r = start(&[], config::Audio::default());
        until("connected", || {
            r.status.snapshot().state == LinkState::Listening
        })
        .await;
        r.llm.push(say("All green."));
        r.fc.push(ServerMessage::Event(AgentEvent {
            ts_ms: 1,
            pane: "%1".into(),
            window: Some(1),
            agent: AgentKind::Claude,
            kind: EventKind::TurnEnd,
            text: Some("I ran the whole suite and everything passed.".into()),
        }));
        until("summary", || {
            spoken(&r) == vec!["Claude: All green.".to_string()]
        })
        .await;
        r.shutdown().await;
    }

    #[tokio::test]
    async fn barge_in_stops_playback_and_cancels_tts_but_only_when_enabled() {
        for (barge_in, expect_cut) in [(true, true), (false, false)] {
            let audio = config::Audio {
                barge_in,
                ..config::Audio::default()
            };
            let r = start(&[], audio);
            until("connected", || {
                r.status.snapshot().state == LinkState::Listening
            })
            .await;
            // An agent finishes in the background: the announcement is not interesting here;
            // use a long summary instead by speaking through a focused turn end.
            r.llm.push(say("a slow reply that goes on"));
            r.fc.push(ServerMessage::Event(AgentEvent {
                ts_ms: 1,
                pane: "%1".into(),
                window: Some(1),
                agent: AgentKind::Claude,
                kind: EventKind::TurnEnd,
                text: Some("done".into()),
            }));
            until("playback", || !r.sink.clips.lock().unwrap().is_empty()).await;
            let stops_before = r.sink.stops.load(Ordering::SeqCst);

            r.audio_tx.send(AudioEvent::SpeechStarted).unwrap();
            if expect_cut {
                until("tts cancelled", || r.tts.cancelled.load(Ordering::SeqCst)).await;
                assert!(!r.sink.is_playing());
                assert!(r.sink.stops.load(Ordering::SeqCst) > stops_before);
            } else {
                tokio::time::sleep(Duration::from_millis(150)).await;
                assert!(
                    !r.tts.cancelled.load(Ordering::SeqCst),
                    "barge_in = false must not cut speech"
                );
                assert!(r.sink.is_playing());
            }
            r.shutdown().await;
        }
    }

    #[tokio::test]
    async fn speech_start_while_silent_is_harmless() {
        let r = start(&[], config::Audio::default());
        r.audio_tx.send(AudioEvent::SpeechStarted).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(r.sink.stops.load(Ordering::SeqCst), 0);
        r.shutdown().await;
    }

    #[tokio::test]
    async fn controller_loss_is_announced_and_the_link_comes_back() {
        let r = start(&["Stop."], config::Audio::default());
        until("connected", || {
            r.status.snapshot().state == LinkState::Listening
        })
        .await;
        // The fake drops the session as it reads the Interrupt.
        r.fc.script.lock().unwrap().drop_next = true;
        r.audio_tx.send(utterance()).unwrap();
        until("second session", || {
            r.fc.connects.load(Ordering::Relaxed) == 2
        })
        .await;
        until("listening again", || {
            r.status.snapshot().state == LinkState::Listening
        })
        .await;
        until("announcements", || {
            let s = spoken(&r);
            s.iter().any(|t| t.starts_with("Lost the connection"))
                && s.iter().any(|t| t == "Reconnected to agentbox.")
        })
        .await;
        r.shutdown().await;
    }

    #[test]
    fn transcript_cleaning() {
        assert_eq!(
            clean_transcript(" Hello there. ").as_deref(),
            Some("Hello there.")
        );
        assert_eq!(clean_transcript("[BLANK_AUDIO]"), None);
        assert_eq!(
            clean_transcript("(music) fix it [noise]").as_deref(),
            Some("fix it")
        );
        assert_eq!(clean_transcript("..."), None);
    }
}
