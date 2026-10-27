//! Client configuration (`~/.config/voicelink/config.toml` on Linux,
//! `~/Library/Application Support/voicelink/config.toml` on macOS).
//!
//! Written by `voicelink setup`, read by every other subcommand.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use voicelink_backends::{LlmConfig, SttConfig, TtsConfig};

/// Whole client configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// Where agentbox is and how to reach it.
    pub connection: Connection,
    /// Audio devices and the voice-activity pipeline.
    pub audio: Audio,
    /// Optional global mute hotkey.
    pub hotkey: Hotkey,
    /// Concierge behaviour.
    pub concierge: Concierge,
    /// Speech-to-text backend.
    pub stt: SttConfig,
    /// Concierge chat model.
    pub llm: LlmConfig,
    /// Text-to-speech backend.
    pub tts: TtsConfig,
    /// Directory holding downloaded models.
    pub models_dir: PathBuf,
}

/// Connection to the agentbox host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Connection {
    /// ssh destination of the Docker host, `user@host` (LAN address or tailnet name).
    pub ssh_destination: String,
    /// Extra ssh options, passed before the destination (for example `["-p", "2222"]`).
    #[serde(default)]
    pub ssh_options: Vec<String>,
    /// Container name.
    pub container: String,
    /// User inside the container (uid or name); must own the tmux server.
    pub container_user: String,
    /// The shared tmux session the operator's tabs live in.
    pub session: String,
    /// This client's grouped view session.
    pub view_session: String,
    /// Controller command inside the container.
    pub paned_command: String,
    /// Local port forwards opened on the ssh ControlMaster for remote backends.
    #[serde(default)]
    pub forwards: Vec<Forward>,
}

/// One `ssh -L` forward: `127.0.0.1:local_port` → `remote` (resolved on the ssh host).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Forward {
    /// Label for status output (for example `"loom"`).
    pub name: String,
    /// Local loopback port.
    pub local_port: u16,
    /// `host:port` as seen from the ssh host.
    pub remote: String,
}

/// Audio pipeline settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Audio {
    /// Input device name, or `None` for the system default.
    pub input_device: Option<String>,
    /// Output device name, or `None` for the system default.
    pub output_device: Option<String>,
    /// In-process acoustic echo cancellation against our own TTS output.
    pub echo_cancel: bool,
    /// Speech probability above which a frame counts as voice (0–1).
    pub vad_threshold: f32,
    /// Silence that ends an utterance, in milliseconds.
    pub end_silence_ms: u32,
    /// Shortest utterance passed to STT, in milliseconds.
    pub min_utterance_ms: u32,
    /// Stop TTS playback as soon as the operator starts speaking.
    pub barge_in: bool,
}

/// Global mute hotkey. Disabled by default: a hardware mic mute is the primary control.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hotkey {
    /// Register the hotkey.
    pub enabled: bool,
    /// Accelerator string, for example `"Ctrl+Alt+M"`.
    pub mute: String,
}

/// How the concierge speaks and confirms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Concierge {
    /// Speak a summary when the focused agent finishes a turn.
    pub speak_turn_end: bool,
    /// Announce turn ends and prompts from agents in other tabs.
    pub announce_background: bool,
    /// Say what will be typed or pressed before doing it.
    pub announce_actions: bool,
    /// Word budget for spoken summaries.
    pub summary_words: u16,
    /// Conversation turns kept in the concierge's context.
    pub history_turns: u16,
}

impl Default for Audio {
    fn default() -> Self {
        Audio {
            input_device: None,
            output_device: None,
            echo_cancel: true,
            vad_threshold: 0.5,
            end_silence_ms: 700,
            min_utterance_ms: 300,
            barge_in: true,
        }
    }
}

impl Default for Hotkey {
    fn default() -> Self {
        Hotkey { enabled: false, mute: "Ctrl+Alt+M".into() }
    }
}

impl Default for Concierge {
    fn default() -> Self {
        Concierge {
            speak_turn_end: true,
            announce_background: true,
            announce_actions: true,
            summary_words: 40,
            history_turns: 12,
        }
    }
}

impl Connection {
    /// Defaults for an agentbox host reached as `ssh_destination`, with a view
    /// session named after this machine.
    pub fn agentbox(ssh_destination: impl Into<String>, client_host: &str) -> Self {
        let slug: String = client_host
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c.to_ascii_lowercase() } else { '-' })
            .collect();
        Connection {
            ssh_destination: ssh_destination.into(),
            ssh_options: Vec::new(),
            container: "agentbox".into(),
            container_user: "1000".into(),
            session: "agentbox".into(),
            view_session: format!("remote-{slug}"),
            paned_command: "/opt/agentbox/bin/voice-paned".into(),
            forwards: Vec::new(),
        }
    }
}
