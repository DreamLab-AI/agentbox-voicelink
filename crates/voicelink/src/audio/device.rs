//! cpal-backed device I/O (feature `audio-io`).
//!
//! `cpal::Stream` is `!Send` on some platforms, so live streams are created and
//! owned by one dedicated thread that parks until the [`Guard`] drops. Callbacks
//! only move samples to and from shared buffers.

use std::any::Any;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig, SupportedStreamConfig};
use voicelink_backends::AudioClip;

use super::resample::resample_clip;
use super::vad::SPEECH_RATE;
use super::{AudioIo, AudioSink, DeviceList, PlaybackQueue, spawn_capture_worker};
use crate::config;

/// Routes libasound's stderr chatter away from the terminal (I5). The handler
/// is thread-local, so every thread that touches ALSA installs its own.
fn silence_alsa() -> Box<dyn Any> {
    #[cfg(target_os = "linux")]
    {
        match alsa::Output::local_error_handler() {
            Ok(h) => Box::new(h),
            Err(_) => Box::new(()),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        Box::new(())
    }
}

fn device_name(d: &Device) -> String {
    d.description()
        .map(|desc| desc.name().to_string())
        .or_else(|_| d.id().map(|id| id.to_string()))
        .unwrap_or_else(|_| "unknown".into())
}

/// A configured device matches by its display name, its full cpal id
/// (`pulseaudio:<node>`) or the bare backend id (the PipeWire or PulseAudio
/// node name, as `wpctl inspect` or `pactl list short sources` print it).
fn matches_name(d: &Device, wanted: &str) -> bool {
    device_name(d) == wanted || d.id().is_ok_and(|id| id_matches(&id, wanted))
}

fn id_matches(id: &cpal::DeviceId, wanted: &str) -> bool {
    id.to_string() == wanted || id.id() == wanted
}

fn pick(host: &cpal::Host, wanted: Option<&str>, input: bool) -> anyhow::Result<Device> {
    match wanted {
        None => if input {
            host.default_input_device()
        } else {
            host.default_output_device()
        }
        .ok_or_else(|| {
            anyhow!(
                "no default {} device",
                if input { "input" } else { "output" }
            )
        }),
        Some(name) => host
            .devices()?
            .filter(|d| {
                if input {
                    d.supports_input()
                } else {
                    d.supports_output()
                }
            })
            .find(|d| matches_name(d, name))
            .ok_or_else(|| {
                anyhow!(
                    "{} device {name:?} not found",
                    if input { "input" } else { "output" }
                )
            }),
    }
}

pub(super) fn list_devices() -> anyhow::Result<DeviceList> {
    let _quiet = silence_alsa();
    let host = cpal::default_host();
    let inputs = host.input_devices()?.map(|d| device_name(&d)).collect();
    let outputs = host.output_devices()?.map(|d| device_name(&d)).collect();
    Ok(DeviceList {
        inputs,
        outputs,
        default_input: host.default_input_device().map(|d| device_name(&d)),
        default_output: host.default_output_device().map(|d| device_name(&d)),
    })
}

fn log_err(what: &'static str) -> impl FnMut(cpal::Error) + Send + 'static {
    move |e| tracing::warn!("{what} stream: {e}")
}

/// Builds an input stream that hands mono f32 chunks to `on_mono`.
fn build_input(
    device: &Device,
    supported: &SupportedStreamConfig,
    on_mono: impl FnMut(&[f32]) + Send + 'static,
) -> anyhow::Result<Stream> {
    let config: StreamConfig = (*supported).into();
    let ch = usize::from(config.channels.max(1));
    let mut on_mono = on_mono;
    macro_rules! build {
        ($t:ty) => {{
            let mut mono: Vec<f32> = Vec::new();
            device.build_input_stream(
                config,
                move |data: &[$t], _: &_| {
                    mono.clear();
                    use cpal::Sample;
                    for frame in data.chunks(ch) {
                        let sum: f32 = frame.iter().map(|s| s.to_sample::<f32>()).sum();
                        mono.push(sum / ch as f32);
                    }
                    on_mono(&mono);
                },
                log_err("input"),
                None,
            )?
        }};
    }
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build!(f32),
        SampleFormat::F64 => build!(f64),
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I24 => build!(cpal::I24),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U24 => build!(cpal::U24),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        other => bail!("unsupported input sample format {other}"),
    };
    Ok(stream)
}

/// Builds an output stream that pulls mono audio from `queue`.
fn build_output(
    device: &Device,
    supported: &SupportedStreamConfig,
    queue: Arc<PlaybackQueue>,
) -> anyhow::Result<Stream> {
    let config: StreamConfig = (*supported).into();
    let ch = usize::from(config.channels.max(1));
    macro_rules! build {
        ($t:ty) => {{
            let mut mono: Vec<f32> = Vec::new();
            device.build_output_stream(
                config,
                move |data: &mut [$t], _: &_| {
                    use cpal::Sample;
                    mono.resize(data.len() / ch, 0.0);
                    queue.fill(&mut mono);
                    for (frame, v) in data.chunks_mut(ch).zip(&mono) {
                        frame.fill(v.to_sample::<$t>());
                    }
                },
                log_err("output"),
                None,
            )?
        }};
    }
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build!(f32),
        SampleFormat::F64 => build!(f64),
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I24 => build!(cpal::I24),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U24 => build!(cpal::U24),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        other => bail!("unsupported output sample format {other}"),
    };
    Ok(stream)
}

pub(super) fn record_clip(audio: &config::Audio, seconds: f32) -> anyhow::Result<AudioClip> {
    let _quiet = silence_alsa();
    let host = cpal::default_host();
    let device = pick(&host, audio.input_device.as_deref(), true)?;
    let supported = device.default_input_config()?;
    let rate = supported.sample_rate();
    let buf = Arc::new(Mutex::new(Vec::<f32>::new()));
    let sink = Arc::clone(&buf);
    let stream = build_input(&device, &supported, move |m| {
        sink.lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend_from_slice(m)
    })?;
    stream.play()?;
    std::thread::sleep(Duration::from_secs_f32(seconds.max(0.0)));
    drop(stream);
    let raw = std::mem::take(&mut *buf.lock().unwrap_or_else(|e| e.into_inner()));
    Ok(AudioClip {
        sample_rate: SPEECH_RATE,
        samples: resample_clip(&raw, rate, SPEECH_RATE)?,
    })
}

pub(super) fn play_clip(audio: &config::Audio, clip: &AudioClip) -> anyhow::Result<()> {
    let _quiet = silence_alsa();
    let host = cpal::default_host();
    let device = pick(&host, audio.output_device.as_deref(), false)?;
    let supported = device.default_output_config()?;
    let queue = Arc::new(PlaybackQueue::new(supported.sample_rate()));
    queue.push(clip.clone());
    let stream = build_output(&device, &supported, Arc::clone(&queue))?;
    stream.play()?;
    let deadline = Instant::now()
        + Duration::from_secs_f32(clip.samples.len() as f32 / clip.sample_rate.max(1) as f32 + 5.0);
    while queue.is_playing() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(stream);
    Ok(())
}

pub(super) fn level_meter(
    audio: &config::Audio,
    on_level: &mut dyn FnMut(f32) -> bool,
) -> anyhow::Result<()> {
    let _quiet = silence_alsa();
    let host = cpal::default_host();
    let device = pick(&host, audio.input_device.as_deref(), true)?;
    let supported = device.default_input_config()?;
    let window = (supported.sample_rate() as usize / 20).max(1); // 50 ms
    let (tx, rx) = mpsc::sync_channel::<f32>(64);
    let (mut acc, mut n) = (0.0f64, 0usize);
    let stream = build_input(&device, &supported, move |m| {
        for s in m {
            acc += f64::from(*s) * f64::from(*s);
            n += 1;
            if n >= window {
                let rms = (acc / n as f64).sqrt() as f32;
                let _ = tx.try_send(20.0 * rms.max(1e-6).log10());
                (acc, n) = (0.0, 0);
            }
        }
    })?;
    stream.play()?;
    loop {
        match rx.recv_timeout(Duration::from_secs(2)) {
            Ok(db) => {
                if !on_level(db) {
                    return Ok(());
                }
            }
            Err(_) => bail!("no audio from the input device for 2 s"),
        }
    }
}

/// Stops the stream-owning thread when dropped.
struct Guard {
    stop: Option<mpsc::Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

struct Opened {
    input_rate: u32,
    queue: Arc<PlaybackQueue>,
    chunks: Receiver<Vec<f32>>,
}

fn open_streams(audio: &config::Audio) -> anyhow::Result<((Stream, Stream), Opened)> {
    let host = cpal::default_host();
    let in_dev = pick(&host, audio.input_device.as_deref(), true)?;
    let out_dev = pick(&host, audio.output_device.as_deref(), false)?;
    let in_cfg = in_dev.default_input_config().context("input config")?;
    let out_cfg = out_dev.default_output_config().context("output config")?;
    tracing::info!(
        "audio in: {} @ {} Hz; out: {} @ {} Hz",
        device_name(&in_dev),
        in_cfg.sample_rate(),
        device_name(&out_dev),
        out_cfg.sample_rate()
    );
    let queue = Arc::new(PlaybackQueue::new(out_cfg.sample_rate()));
    let (tx, rx): (SyncSender<Vec<f32>>, _) = mpsc::sync_channel(128);
    let input = build_input(&in_dev, &in_cfg, move |m| {
        let _ = tx.try_send(m.to_vec());
    })?;
    let output = build_output(&out_dev, &out_cfg, Arc::clone(&queue))?;
    input.play()?;
    output.play()?;
    Ok((
        (input, output),
        Opened {
            input_rate: in_cfg.sample_rate(),
            queue,
            chunks: rx,
        },
    ))
}

pub(super) fn open(audio: &config::Audio) -> anyhow::Result<AudioIo> {
    let (ready_tx, ready_rx) = mpsc::channel::<Result<Opened, String>>();
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let cfg = audio.clone();
    let join = std::thread::Builder::new()
        .name("voicelink-audio-io".into())
        .spawn(move || {
            let _quiet = silence_alsa();
            match open_streams(&cfg) {
                Ok((streams, opened)) => {
                    let _ = ready_tx.send(Ok(opened));
                    let _ = stop_rx.recv(); // parks until the guard drops
                    drop(streams);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{e:#}")));
                }
            }
        })?;
    let guard = Guard {
        stop: Some(stop_tx),
        join: Some(join),
    };
    let opened = ready_rx
        .recv()
        .map_err(|_| anyhow!("audio thread exited during start-up"))?
        .map_err(|e| anyhow!(e))?;
    let muted = Arc::new(AtomicBool::new(false));
    let events = spawn_capture_worker(
        opened.input_rate,
        &opened.queue,
        audio,
        Arc::clone(&muted),
        opened.chunks,
    )?;
    Ok(AudioIo {
        events,
        sink: opened.queue,
        muted,
        _guard: Box::new(guard),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumeration_does_not_panic_without_sound_hardware() {
        // CI and containers may have no devices; either outcome is fine, a panic is not.
        let _ = list_devices();
    }

    #[test]
    fn ids_match_with_or_without_the_host_prefix() {
        let node = "alsa_input.pci-0000_c3_00.6.HiFi__Mic1__source";
        let id = cpal::DeviceId::new(cpal::default_host().id(), node);
        assert!(id_matches(&id, node));
        assert!(id_matches(&id, &id.to_string()));
        assert!(!id_matches(
            &id,
            "alsa_input.pci-0000_c3_00.6.HiFi__Mic2__source"
        ));
    }
}
