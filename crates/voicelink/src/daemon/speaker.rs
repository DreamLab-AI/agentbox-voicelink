//! Speaks text: runs TTS and feeds the clips to the playback sink.
//!
//! Each reply gets its own fresh channel, so chunks from a cancelled synthesis
//! cannot leak into the next reply. Cancelling (barge-in) drops the receiver
//! and the in-flight synthesis future, clears the sink and discards queued text.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;
use voicelink_backends::{AudioClip, TextToSpeech};

use super::status_file::StatusHandle;
use crate::audio::AudioSink;

/// Handle for barge-in and activity checks.
#[derive(Clone)]
pub(crate) struct SpeakerHandle {
    cancel: mpsc::UnboundedSender<()>,
    busy: Arc<AtomicBool>,
    sink: Arc<dyn AudioSink>,
}

impl SpeakerHandle {
    /// Whether text is queued or synthesising, or audio is still playing.
    pub(crate) fn is_active(&self) -> bool {
        self.busy.load(Ordering::Acquire) || self.sink.is_playing()
    }

    /// Stops speaking now: playback is cleared on the spot, synthesis is
    /// cancelled and any queued text is dropped.
    pub(crate) fn cancel(&self) {
        self.sink.stop();
        let _ = self.cancel.send(());
    }
}

/// Starts the speaker task. Text arrives on `texts`.
pub(crate) fn spawn(
    tts: Arc<dyn TextToSpeech>,
    sink: Arc<dyn AudioSink>,
    texts: mpsc::UnboundedReceiver<String>,
    status: StatusHandle,
) -> (SpeakerHandle, tokio::task::JoinHandle<()>) {
    let (cancel, cancel_rx) = mpsc::unbounded_channel();
    let busy = Arc::new(AtomicBool::new(false));
    let handle = SpeakerHandle {
        cancel,
        busy: Arc::clone(&busy),
        sink: Arc::clone(&sink),
    };
    let task = tokio::spawn(run(tts, sink, texts, cancel_rx, busy, status));
    (handle, task)
}

async fn run(
    tts: Arc<dyn TextToSpeech>,
    sink: Arc<dyn AudioSink>,
    mut texts: mpsc::UnboundedReceiver<String>,
    mut cancel: mpsc::UnboundedReceiver<()>,
    busy: Arc<AtomicBool>,
    status: StatusHandle,
) {
    loop {
        let text = tokio::select! {
            t = texts.recv() => match t { Some(t) => t, None => return },
            c = cancel.recv() => {
                if c.is_none() { return }
                discard(&mut texts, &sink);
                continue;
            }
        };
        busy.store(true, Ordering::Release);
        let cancelled = speak_one(&*tts, &*sink, &text, &mut cancel, &status).await;
        if cancelled {
            discard(&mut texts, &sink);
        }
        if texts.is_empty() {
            busy.store(false, Ordering::Release);
        }
    }
}

/// Drops every queued text and any audio already handed to the sink.
fn discard(texts: &mut mpsc::UnboundedReceiver<String>, sink: &Arc<dyn AudioSink>) {
    while texts.try_recv().is_ok() {}
    sink.stop();
}

/// Synthesises `text` into the sink. Returns `true` when cancelled midway.
async fn speak_one(
    tts: &dyn TextToSpeech,
    sink: &dyn AudioSink,
    text: &str,
    cancel: &mut mpsc::UnboundedReceiver<()>,
    status: &StatusHandle,
) -> bool {
    let (tx, mut rx) = mpsc::channel::<AudioClip>(8);
    let synth = tts.synthesize(text, tx);
    tokio::pin!(synth);
    let mut synth_done = false;
    loop {
        tokio::select! {
            biased;
            c = cancel.recv() => {
                // Dropping `rx` and `synth` cancels the backend.
                return c.is_some();
            }
            res = &mut synth, if !synth_done => {
                synth_done = true;
                if let Err(e) = res {
                    tracing::warn!("tts failed: {e}");
                }
            }
            clip = rx.recv() => match clip {
                Some(clip) => {
                    status.mark_first_audio();
                    sink.push(clip);
                }
                None => return false,
            },
        }
    }
}

#[cfg(test)]
pub(crate) mod fake {
    //! Fakes for speaker and engine tests.

    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use async_trait::async_trait;
    use voicelink_backends::{BackendError, Result};

    use super::*;

    /// Records pushes and stops without any device.
    #[derive(Default)]
    pub(crate) struct FakeSink {
        pub clips: Mutex<Vec<AudioClip>>,
        pub stops: AtomicUsize,
        pub playing: AtomicBool,
    }

    impl AudioSink for FakeSink {
        fn push(&self, clip: AudioClip) {
            self.clips.lock().unwrap().push(clip);
            self.playing.store(true, Ordering::SeqCst);
        }
        fn stop(&self) {
            self.clips.lock().unwrap().clear();
            self.stops.fetch_add(1, Ordering::SeqCst);
            self.playing.store(false, Ordering::SeqCst);
        }
        fn is_playing(&self) -> bool {
            self.playing.load(Ordering::SeqCst)
        }
    }

    /// Sets the flag if dropped before `done` is set, i.e. cancelled mid-synthesis.
    struct CancelGuard {
        flag: Arc<AtomicBool>,
        done: bool,
    }
    impl Drop for CancelGuard {
        fn drop(&mut self) {
            if !self.done {
                self.flag.store(true, Ordering::SeqCst);
            }
        }
    }

    /// One short clip per text; texts containing "slow" then block forever after the first chunk.
    #[derive(Default)]
    pub(crate) struct FakeTts {
        pub spoken: Mutex<Vec<String>>,
        pub cancelled: Arc<AtomicBool>,
    }

    #[async_trait]
    impl TextToSpeech for FakeTts {
        async fn synthesize(&self, text: &str, sink: mpsc::Sender<AudioClip>) -> Result<()> {
            self.spoken.lock().unwrap().push(text.to_string());
            let mut guard = CancelGuard {
                flag: Arc::clone(&self.cancelled),
                done: false,
            };
            let clip = AudioClip {
                sample_rate: 24_000,
                samples: vec![0.1; 240],
            };
            sink.send(clip.clone())
                .await
                .map_err(|_| BackendError::Cancelled)?;
            if text.contains("slow") {
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    sink.send(clip.clone())
                        .await
                        .map_err(|_| BackendError::Cancelled)?;
                }
            }
            guard.done = true;
            Ok(())
        }
        fn name(&self) -> String {
            "fake-tts".into()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::fake::*;
    use super::*;
    use crate::status::BackendNames;

    fn status() -> StatusHandle {
        StatusHandle::new(
            None,
            BackendNames {
                stt: "s".into(),
                llm: "l".into(),
                tts: "t".into(),
            },
        )
    }

    async fn until(mut f: impl FnMut() -> bool) {
        for _ in 0..200 {
            if f() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("condition not reached");
    }

    #[tokio::test]
    async fn speaks_queued_texts_in_order_into_the_sink() {
        let tts = Arc::new(FakeTts::default());
        let sink = Arc::new(FakeSink::default());
        let (tx, rx) = mpsc::unbounded_channel();
        let (handle, _task) = spawn(tts.clone(), sink.clone(), rx, status());
        tx.send("first".into()).unwrap();
        tx.send("second".into()).unwrap();
        until(|| sink.clips.lock().unwrap().len() == 2).await;
        assert_eq!(
            *tts.spoken.lock().unwrap(),
            vec!["first".to_string(), "second".to_string()]
        );
        until(|| !handle.busy.load(Ordering::SeqCst)).await;
    }

    #[tokio::test]
    async fn cancel_stops_playback_cancels_synthesis_and_drops_queued_text() {
        let tts = Arc::new(FakeTts::default());
        let sink = Arc::new(FakeSink::default());
        let (tx, rx) = mpsc::unbounded_channel();
        let (handle, _task) = spawn(tts.clone(), sink.clone(), rx, status());
        tx.send("a slow reply".into()).unwrap();
        tx.send("never spoken".into()).unwrap();
        until(|| !sink.clips.lock().unwrap().is_empty()).await;
        assert!(handle.is_active());

        handle.cancel();
        assert!(!sink.is_playing(), "playback is cleared synchronously");
        until(|| tts.cancelled.load(Ordering::SeqCst)).await;
        until(|| !handle.busy.load(Ordering::SeqCst)).await;
        assert_eq!(
            *tts.spoken.lock().unwrap(),
            vec!["a slow reply".to_string()]
        );
        // No stragglers from the cancelled synthesis end up in the sink.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(sink.clips.lock().unwrap().is_empty());

        // The speaker is still usable afterwards, on a fresh channel.
        tx.send("hello again".into()).unwrap();
        until(|| !sink.clips.lock().unwrap().is_empty()).await;
        assert_eq!(tts.spoken.lock().unwrap().last().unwrap(), "hello again");
    }
}
