//! Audio devices and the capture/playback pipeline.
//!
//! The four functions below are the interface the setup wizard calls
//! (`list_devices`, `record_clip`, `play_clip`, `level_meter`). Everything else
//! is the live pipeline used by the daemon:
//!
//! ```text
//! mic -> mono -> 16 kHz -> echo cancel (far-end = what we played) -> VAD -> segmenter -> AudioEvent
//! TTS clips -> PlaybackQueue (resampled to device rate) -> speaker
//! ```
//!
//! Device I/O needs the `audio-io` feature (cpal). Without it the pipeline
//! still compiles and is exercised on synthetic buffers; the device entry
//! points return an error saying so.
// Most of this module is only reachable through the cpal backend; without the
// feature it is exercised by the unit tests alone.
#![cfg_attr(not(feature = "audio-io"), allow(dead_code))]

mod aec;
#[cfg(feature = "audio-io")]
mod device;
mod pipeline;
mod playback;
mod resample;
mod vad;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;

use tokio::sync::mpsc::UnboundedReceiver;
use voicelink_backends::AudioClip;

use crate::config;

pub(crate) use playback::{AudioSink, PlaybackQueue};

#[allow(dead_code)] // called by the setup wizard
/// Audio devices visible to the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceList {
    /// Input device names.
    pub inputs: Vec<String>,
    /// Output device names.
    pub outputs: Vec<String>,
    /// Default input, if any.
    pub default_input: Option<String>,
    /// Default output, if any.
    pub default_output: Option<String>,
}

/// What the microphone side reports to the engine.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioEvent {
    /// Speech began (fires before the utterance is complete; drives barge-in).
    SpeechStarted,
    /// One complete utterance, mono at 16 kHz, with a short pre-roll.
    Utterance(AudioClip),
}

/// An open audio engine: events in, a playback sink out.
pub(crate) struct AudioIo {
    /// Speech events from the microphone.
    pub events: UnboundedReceiver<AudioEvent>,
    /// Where synthesised speech is queued.
    pub sink: Arc<PlaybackQueue>,
    /// Set to mute the microphone path.
    pub muted: Arc<AtomicBool>,
    /// Keeps the device streams alive.
    pub _guard: Box<dyn std::any::Any + Send>,
}

#[cfg(not(feature = "audio-io"))]
const NO_AUDIO_IO: &str =
    "this build has no audio device support (compiled without the `audio-io` feature)";

#[allow(dead_code)] // called by the setup wizard
/// Lists audio devices.
pub fn list_devices() -> anyhow::Result<DeviceList> {
    #[cfg(feature = "audio-io")]
    {
        device::list_devices()
    }
    #[cfg(not(feature = "audio-io"))]
    {
        anyhow::bail!(NO_AUDIO_IO)
    }
}

#[allow(dead_code)] // called by the setup wizard
/// Records `seconds` of mono audio from the configured input (blocking).
///
/// The clip is returned at 16 kHz, ready for speech-to-text.
pub fn record_clip(audio: &config::Audio, seconds: f32) -> anyhow::Result<AudioClip> {
    #[cfg(feature = "audio-io")]
    {
        device::record_clip(audio, seconds)
    }
    #[cfg(not(feature = "audio-io"))]
    {
        let _ = (audio, seconds);
        anyhow::bail!(NO_AUDIO_IO)
    }
}

#[allow(dead_code)] // called by the setup wizard
/// Plays a clip on the configured output and waits for it to finish (blocking).
pub fn play_clip(audio: &config::Audio, clip: &AudioClip) -> anyhow::Result<()> {
    #[cfg(feature = "audio-io")]
    {
        device::play_clip(audio, clip)
    }
    #[cfg(not(feature = "audio-io"))]
    {
        let _ = (audio, clip);
        anyhow::bail!(NO_AUDIO_IO)
    }
}

#[allow(dead_code)] // called by the setup wizard
/// Streams input level in dBFS to `on_level` roughly every 50 ms until it
/// returns `false` (blocking).
pub fn level_meter(
    audio: &config::Audio,
    on_level: &mut dyn FnMut(f32) -> bool,
) -> anyhow::Result<()> {
    #[cfg(feature = "audio-io")]
    {
        device::level_meter(audio, on_level)
    }
    #[cfg(not(feature = "audio-io"))]
    {
        let _ = (audio, on_level);
        anyhow::bail!(NO_AUDIO_IO)
    }
}

/// Opens the configured devices and starts the capture pipeline.
pub(crate) fn open(audio: &config::Audio) -> anyhow::Result<AudioIo> {
    #[cfg(feature = "audio-io")]
    {
        device::open(audio)
    }
    #[cfg(not(feature = "audio-io"))]
    {
        let _ = audio;
        anyhow::bail!(NO_AUDIO_IO)
    }
}

/// Runs the capture pipeline on its own thread, fed raw mono input chunks.
///
/// The DSP stays off the device callback: the callback only forwards samples.
pub(crate) fn spawn_capture_worker(
    input_rate: u32,
    playback: &PlaybackQueue,
    audio: &config::Audio,
    muted: Arc<AtomicBool>,
    chunks: Receiver<Vec<f32>>,
) -> anyhow::Result<UnboundedReceiver<AudioEvent>> {
    let tap = playback.tap();
    let echo_cancel = audio.echo_cancel;
    let seg_cfg = vad::SegmenterConfig::from_audio(audio);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    // The AEC graph is not `Send`, so the pipeline is built on the thread that runs it.
    std::thread::Builder::new()
        .name("voicelink-capture-dsp".into())
        .spawn(move || {
            let built = pipeline::CapturePipeline::new(
                input_rate,
                tap,
                echo_cancel,
                Box::new(vad::EarshotScorer::new()),
                seg_cfg,
                muted,
            );
            let mut pipe = match built {
                Ok(p) => {
                    let _ = ready_tx.send(Ok(()));
                    p
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            let mut events = Vec::new();
            while let Ok(chunk) = chunks.recv() {
                pipe.push_input(&chunk, &mut events);
                for e in events.drain(..) {
                    if tx.send(e).is_err() {
                        return;
                    }
                }
            }
        })?;
    ready_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("capture thread exited during start-up"))?
        .map_err(|e| anyhow::anyhow!(e))?;
    Ok(rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_thread_turns_chunks_into_events_and_stops_with_its_input() {
        let audio = config::Audio {
            echo_cancel: false,
            ..config::Audio::default()
        };
        let q = PlaybackQueue::new(16_000);
        let (tx, rx) = std::sync::mpsc::channel();
        let mut events =
            spawn_capture_worker(16_000, &q, &audio, Arc::new(AtomicBool::new(false)), rx).unwrap();
        // Silence only: no events, and the channel closes when the sender drops.
        for _ in 0..50 {
            tx.send(vec![0.0; 160]).unwrap();
        }
        drop(tx);
        let end = futures_lite_block(async { events.recv().await });
        assert!(end.is_none());
    }

    fn futures_lite_block<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(f)
    }

    #[cfg(not(feature = "audio-io"))]
    #[test]
    fn device_entry_points_explain_the_missing_feature() {
        let a = config::Audio::default();
        let err = record_clip(&a, 1.0).unwrap_err().to_string();
        assert!(err.contains("audio-io"), "{err}");
        assert!(list_devices().is_err());
    }
}
