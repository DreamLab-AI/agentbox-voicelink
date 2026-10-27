//! Optional global mute hotkey (`[hotkey]` in the config).
//!
//! Platform limits, honestly stated: `global-hotkey` works on X11 and macOS
//! (and Windows). Wayland compositors generally do not let a client grab keys
//! globally, so on a Wayland session registration usually fails or never fires;
//! a warning says so. On macOS the underlying API wants a running `NSApplication`
//! event loop on the main thread, which this daemon does not run, so the
//! hotkey may not fire there (unverified, see the engine report). A hardware
//! mic mute is the primary control; this is a convenience.

use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

use super::status_file::StatusHandle;
use crate::config;

/// Parses an accelerator such as `"Ctrl+Alt+M"`.
pub(crate) fn parse(accelerator: &str) -> anyhow::Result<HotKey> {
    HotKey::from_str(accelerator)
        .map_err(|e| anyhow::anyhow!("invalid hotkey {accelerator:?}: {e}"))
}

fn wayland_session() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty())
        || std::env::var("XDG_SESSION_TYPE").is_ok_and(|v| v.eq_ignore_ascii_case("wayland"))
}

/// Registers the hotkey on a dedicated thread (the manager must live on the
/// thread that created it). Failure is logged, never fatal.
pub(crate) fn spawn(cfg: &config::Hotkey, muted: Arc<AtomicBool>, status: StatusHandle) {
    if !cfg.enabled {
        return;
    }
    let hotkey = match parse(&cfg.mute) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!("mute hotkey disabled: {e:#}");
            return;
        }
    };
    if wayland_session() {
        tracing::warn!(
            "mute hotkey: this looks like a Wayland session; compositors generally do not allow global key grabs, so {} may not work (X11 and macOS are supported)",
            cfg.mute
        );
    }
    let accel = cfg.mute.clone();
    let spawned = std::thread::Builder::new()
        .name("voicelink-hotkey".into())
        .spawn(move || {
            let manager = match GlobalHotKeyManager::new() {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!("mute hotkey unavailable: {e}");
                    return;
                }
            };
            if let Err(e) = manager.register(hotkey) {
                tracing::warn!("cannot register mute hotkey {accel}: {e}");
                return;
            }
            tracing::info!("mute hotkey {accel} registered");
            let rx = GlobalHotKeyEvent::receiver();
            while let Ok(ev) = rx.recv() {
                if ev.id == hotkey.id() && ev.state == HotKeyState::Pressed {
                    let now_muted = !muted.fetch_xor(true, Ordering::SeqCst);
                    tracing::info!("microphone {}", if now_muted { "muted" } else { "unmuted" });
                    status.update(|s| s.muted = now_muted);
                }
            }
            drop(manager);
        });
    if let Err(e) = spawned {
        tracing::warn!("cannot start hotkey thread: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_accelerator_parses() {
        let default = config::Hotkey::default();
        parse(&default.mute).expect("default accelerator must parse with global-hotkey");
    }

    #[test]
    fn nonsense_is_rejected() {
        assert!(parse("Banana+Pie").is_err());
    }
}
