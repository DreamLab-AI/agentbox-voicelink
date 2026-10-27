//! Per-user file locations.
//!
//! | File | Linux | macOS |
//! |---|---|---|
//! | config | `~/.config/voicelink/config.toml` | `~/Library/Application Support/voicelink/config.toml` |
//! | state (status, log, ssh control socket) | `~/.local/state/voicelink/` | `~/Library/Application Support/voicelink/state/` |
//! | models | `~/.local/share/voicelink/models/` | `~/Library/Application Support/voicelink/models/` |

use std::path::PathBuf;

use anyhow::Context;
use directories::ProjectDirs;

fn dirs() -> anyhow::Result<ProjectDirs> {
    ProjectDirs::from("uk", "DreamLab", "voicelink").context("cannot determine the home directory")
}

/// The config file.
pub fn config_file() -> anyhow::Result<PathBuf> {
    Ok(dirs()?.config_dir().join("config.toml"))
}

/// Directory for runtime state: status file, daemon log, ssh control socket.
pub fn state_dir() -> anyhow::Result<PathBuf> {
    let d = dirs()?;
    Ok(d.state_dir()
        .map(PathBuf::from)
        .unwrap_or_else(|| d.data_local_dir().join("state")))
}

/// The daemon status file.
pub fn status_file() -> anyhow::Result<PathBuf> {
    Ok(state_dir()?.join("status.json"))
}

/// The daemon log file.
pub fn log_file() -> anyhow::Result<PathBuf> {
    Ok(state_dir()?.join("daemon.log"))
}

/// Default models directory.
pub fn models_dir() -> anyhow::Result<PathBuf> {
    Ok(dirs()?.data_local_dir().join("models"))
}
