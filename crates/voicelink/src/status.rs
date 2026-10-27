//! Daemon status file, written atomically by `voicelink daemon` and read by
//! `voicelink status`. The daemon never writes to the attached terminal.

use serde::{Deserialize, Serialize};

/// Snapshot of the running daemon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonStatus {
    /// Daemon process id.
    pub pid: u32,
    /// Milliseconds since the Unix epoch of this snapshot.
    pub updated_ms: u64,
    /// Overall state.
    pub state: LinkState,
    /// Focus as last reported by the controller (`"2:Services codex"`), if linked.
    pub focus: Option<String>,
    /// Whether the microphone is muted by the hotkey.
    pub muted: bool,
    /// Backend names in use.
    pub backends: BackendNames,
    /// Last end-of-speech to transcript latency, in milliseconds.
    pub last_stt_ms: Option<u32>,
    /// Last transcript to first spoken audio latency, in milliseconds.
    pub last_reply_ms: Option<u32>,
    /// Most recent error, if any.
    pub last_error: Option<String>,
}

/// Link lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkState {
    /// Starting audio and backends.
    Starting,
    /// Connecting to the controller.
    Connecting,
    /// Linked and listening.
    Listening,
    /// Lost the controller; retrying.
    Reconnecting,
    /// Stopped.
    Stopped,
}

/// Names of the active backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendNames {
    /// Speech-to-text.
    pub stt: String,
    /// Chat model.
    pub llm: String,
    /// Text-to-speech.
    pub tts: String,
}
