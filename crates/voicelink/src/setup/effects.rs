//! Runs wizard [`Effect`]s on worker threads and reports [`Outcome`]s.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use voicelink_backends::{AudioClip, build_stt, build_tts};

use super::wizard::{Effect, Outcome};
use super::{net, probe, remote, store};
use crate::config::Config;

/// Cancellation flags shared with running effects.
#[derive(Debug, Default, Clone)]
pub struct Controls {
    /// Set to stop the level meter.
    pub meter_stop: Arc<AtomicBool>,
    /// Set to abort downloads (partial files are kept).
    pub cancel: Arc<AtomicBool>,
}

fn lines(what: &'static str, ok: bool, lines: Vec<String>) -> Outcome {
    Outcome::Lines { what, ok, lines }
}

/// Starts `effect` on a new thread.
pub fn spawn(effect: Effect, tx: Sender<Outcome>, ctl: &Controls) {
    let ctl = ctl.clone();
    if matches!(effect, Effect::StopMeter) {
        ctl.meter_stop.store(true, Ordering::Relaxed);
        return;
    }
    std::thread::spawn(move || {
        for o in run(effect, &tx, &ctl) {
            let _ = tx.send(o);
        }
    });
}

fn run(effect: Effect, tx: &Sender<Outcome>, ctl: &Controls) -> Vec<Outcome> {
    match effect {
        Effect::Probe => vec![Outcome::Hardware(probe::probe())],
        Effect::Detect => vec![Outcome::Detected(net::detect_servers())],
        Effect::ToolProbe {
            base_url,
            model,
            extra,
        } => vec![Outcome::Tool(net::tool_probe(
            &base_url, &model, &extra, None,
        ))],
        Effect::ListDevices => vec![Outcome::Devices(
            crate::audio::list_devices().map_err(|e| format!("{e:#}")),
        )],
        Effect::StartMeter(audio) => {
            ctl.meter_stop.store(false, Ordering::Relaxed);
            let stop = ctl.meter_stop.clone();
            let t = tx.clone();
            let r = crate::audio::level_meter(&audio, &mut |db| {
                let _ = t.send(Outcome::Level(db));
                !stop.load(Ordering::Relaxed)
            });
            vec![Outcome::MeterStopped(r.err().map(|e| format!("{e:#}")))]
        }
        Effect::StopMeter => Vec::new(),
        Effect::RecordPlayback(audio) => vec![record_playback(&audio)],
        Effect::Download(entries, dir) => {
            ctl.cancel.store(false, Ordering::Relaxed);
            let mut report = Vec::new();
            let mut ok = true;
            for e in entries {
                let t = tx.clone();
                let name = e.file.to_string();
                match net::download(e, &dir, &ctl.cancel, &mut |d, total| {
                    let _ = t.send(Outcome::Progress(name.clone(), d, total));
                }) {
                    Ok(p) => report.push(format!(
                        "ok   {} (sha256 verified) -> {}",
                        e.file,
                        p.display()
                    )),
                    Err(err) => {
                        ok = false;
                        report.push(format!("FAIL {}: {err:#}", e.file));
                    }
                }
            }
            vec![lines("download", ok, report)]
        }
        Effect::StartStack(cmd) => {
            // The wizard only emits this after an explicit second Enter.
            let (ok, tail) = cmd.run_captured();
            let mut report = vec![format!("ran: {}", cmd.display())];
            report.extend(tail);
            if ok {
                report.push(
                    "Started. The first image pull can take a while: check progress with `docker compose ps`."
                        .into(),
                );
            }
            vec![lines("stack", ok, report)]
        }
        Effect::CheckSsh(conn) => match remote::check_version(&conn) {
            remote::Verdict::Ready(v) => vec![lines(
                "ssh",
                true,
                vec![format!("voice-paned answered: {v}")],
            )],
            remote::Verdict::Problem(p) => vec![lines("ssh", false, vec![p])],
        },
        Effect::Handshake(conn) => match remote::handshake(&conn) {
            Ok(w) => vec![lines(
                "handshake",
                true,
                vec![format!("Welcome from {} (focus {})", w.server, w.focus)],
            )],
            Err(e) => vec![lines("handshake", false, vec![e])],
        },
        Effect::Smoke(cfg) => vec![smoke(&cfg)],
        Effect::Save(cfg) => {
            let r = crate::paths::config_file()
                .and_then(|p| store::save(&p, &cfg).map(|()| p))
                .map_err(|e| format!("{e:#}"));
            vec![Outcome::Saved(r)]
        }
    }
}

fn record_playback(audio: &crate::config::Audio) -> Outcome {
    let clip = match crate::audio::record_clip(audio, 3.0) {
        Ok(c) => c,
        Err(e) => return lines("record", false, vec![format!("recording failed: {e:#}")]),
    };
    let peak = clip.samples.iter().fold(0f32, |m, s| m.max(s.abs()));
    let mut out = vec![format!(
        "recorded {:.1} s, peak {:.0} dBFS",
        clip.samples.len() as f32 / clip.sample_rate.max(1) as f32,
        20.0 * peak.max(1e-6).log10()
    )];
    match crate::audio::play_clip(audio, &clip) {
        Ok(()) => lines("record", true, out),
        Err(e) => {
            out.push(format!("playback failed: {e:#}"));
            lines("record", false, out)
        }
    }
}

fn smoke(cfg: &Config) -> Outcome {
    let mut out = Vec::new();
    let mut ok = true;
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => return lines("smoke", false, vec![format!("cannot start a runtime: {e}")]),
    };
    let remote_hint = |what: &str| {
        format!(
            "  ({what} goes through an ssh forward that only exists during `voicelink connect`; this check will pass there)"
        )
    };

    match build_stt(&cfg.stt) {
        Err(e) => {
            ok = false;
            out.push(format!("STT unavailable: {e}"));
            if matches!(cfg.stt, voicelink_backends::SttConfig::Nemotron { .. }) {
                out.push(remote_hint("Nemotron"));
            }
        }
        Ok(stt) => match crate::audio::record_clip(&cfg.audio, 3.0) {
            Err(e) => {
                ok = false;
                out.push(format!("recording failed: {e:#}"));
            }
            Ok(clip) => {
                let t0 = Instant::now();
                match rt.block_on(stt.transcribe(&clip)) {
                    Ok(t) => out.push(format!(
                        "heard {:?} via {} in {} ms",
                        t.text,
                        stt.name(),
                        t0.elapsed().as_millis()
                    )),
                    Err(e) => {
                        ok = false;
                        out.push(format!("transcription failed: {e}"));
                    }
                }
            }
        },
    }

    match build_tts(&cfg.tts) {
        Err(e) => {
            ok = false;
            out.push(format!("TTS unavailable: {e}"));
        }
        Ok(tts) => {
            let (ctx, mut crx) = tokio::sync::mpsc::channel::<AudioClip>(16);
            let t0 = Instant::now();
            let (res, (chunks, first)) = rt.block_on(async {
                tokio::join!(tts.synthesize("Voicelink is ready.", ctx), async {
                    let mut chunks: Vec<AudioClip> = Vec::new();
                    let mut first = None;
                    while let Some(c) = crx.recv().await {
                        first.get_or_insert_with(|| t0.elapsed());
                        chunks.push(c);
                    }
                    (chunks, first)
                })
            });
            match (res, chunks.first()) {
                (Ok(()), Some(head)) => {
                    let clip = AudioClip {
                        sample_rate: head.sample_rate,
                        samples: chunks
                            .iter()
                            .flat_map(|c| c.samples.iter().copied())
                            .collect(),
                    };
                    out.push(format!(
                        "{} spoke {:.1} s of audio; first audio after {} ms",
                        tts.name(),
                        clip.samples.len() as f32 / clip.sample_rate.max(1) as f32,
                        first.unwrap_or(Duration::ZERO).as_millis()
                    ));
                    if let Err(e) = crate::audio::play_clip(&cfg.audio, &clip) {
                        ok = false;
                        out.push(format!("playback failed: {e:#}"));
                    }
                }
                (Err(e), _) => {
                    ok = false;
                    out.push(format!("synthesis failed: {e}"));
                    if matches!(&cfg.tts, voicelink_backends::TtsConfig::OpenAi { base_url, .. } if *base_url == super::choices::pocket_url())
                    {
                        out.push(remote_hint("Pocket TTS"));
                    }
                }
                (Ok(()), None) => {
                    ok = false;
                    out.push("synthesis produced no audio".into());
                }
            }
        }
    }
    lines("smoke", ok, out)
}
