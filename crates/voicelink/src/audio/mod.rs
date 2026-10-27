//! Audio devices and the capture/playback pipeline (owned by the engine worker).
//!
//! The functions below are the interface the setup wizard calls; their
//! signatures are fixed. Without the `audio-io` feature they return an error
//! saying so.

use voicelink_backends::AudioClip;

use crate::config;

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

/// Lists audio devices.
pub fn list_devices() -> anyhow::Result<DeviceList> {
    anyhow::bail!("audio not built yet")
}

/// Records `seconds` of mono audio from the configured input (blocking).
pub fn record_clip(audio: &config::Audio, seconds: f32) -> anyhow::Result<AudioClip> {
    let _ = (audio, seconds);
    anyhow::bail!("audio not built yet")
}

/// Plays a clip on the configured output and waits for it to finish (blocking).
pub fn play_clip(audio: &config::Audio, clip: &AudioClip) -> anyhow::Result<()> {
    let _ = (audio, clip);
    anyhow::bail!("audio not built yet")
}

/// Streams input level in dBFS to `on_level` roughly every 50 ms until it
/// returns `false` (blocking).
pub fn level_meter(
    audio: &config::Audio,
    on_level: &mut dyn FnMut(f32) -> bool,
) -> anyhow::Result<()> {
    let _ = (audio, on_level);
    anyhow::bail!("audio not built yet")
}
