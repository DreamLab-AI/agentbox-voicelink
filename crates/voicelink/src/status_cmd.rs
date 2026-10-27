//! `voicelink status`: a concise summary of the daemon's status file.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use nix::sys::signal::kill;
use nix::unistd::Pid;

use crate::status::{DaemonStatus, LinkState};

/// Whether `pid` names a live process we may signal.
pub fn pid_alive(pid: u32) -> bool {
    let Ok(raw) = i32::try_from(pid) else {
        return false;
    };
    if raw <= 0 {
        return false;
    }
    match kill(Pid::from_raw(raw), None) {
        Ok(()) => true,
        Err(nix::errno::Errno::EPERM) => true,
        Err(_) => false,
    }
}

fn state_name(s: LinkState) -> &'static str {
    match s {
        LinkState::Starting => "starting",
        LinkState::Connecting => "connecting",
        LinkState::Listening => "listening",
        LinkState::Reconnecting => "reconnecting",
        LinkState::Stopped => "stopped",
    }
}

fn ms(v: Option<u32>) -> String {
    v.map_or("-".into(), |v| format!("{v} ms"))
}

fn age(now_ms: u64, then_ms: u64) -> String {
    let s = now_ms.saturating_sub(then_ms) / 1000;
    match s {
        0..=59 => format!("{s} s ago"),
        60..=3599 => format!("{} min ago", s / 60),
        _ => format!("{} h ago", s / 3600),
    }
}

/// Renders the summary. `alive` says whether the recorded pid still exists.
pub fn render(status: &DaemonStatus, alive: bool, now_ms: u64, log: &Path) -> String {
    let state = if alive {
        state_name(status.state).to_string()
    } else if status.state == LinkState::Stopped {
        "stopped".into()
    } else {
        format!(
            "not running (stale status from pid {}, last state {})",
            status.pid,
            state_name(status.state)
        )
    };
    let mut out = String::new();
    out.push_str(&format!("state:     {state}\n"));
    if alive {
        out.push_str(&format!(
            "pid:       {} (updated {})\n",
            status.pid,
            age(now_ms, status.updated_ms)
        ));
        out.push_str(&format!(
            "focus:     {}\n",
            status.focus.as_deref().unwrap_or("-")
        ));
        out.push_str(&format!(
            "mic:       {}\n",
            if status.muted { "muted" } else { "live" }
        ));
    }
    out.push_str(&format!(
        "backends:  stt {} | llm {} | tts {}\n",
        status.backends.stt, status.backends.llm, status.backends.tts
    ));
    out.push_str(&format!(
        "latency:   speech-to-text {} | reply {}\n",
        ms(status.last_stt_ms),
        ms(status.last_reply_ms)
    ));
    out.push_str(&format!(
        "last error: {}\n",
        status.last_error.as_deref().unwrap_or("none")
    ));
    out.push_str(&format!("log:       {}\n", log.display()));
    out
}

/// Prints the daemon status.
pub fn run() -> anyhow::Result<()> {
    let file = crate::paths::status_file()?;
    let log = crate::paths::log_file()?;
    let text = match std::fs::read_to_string(&file) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!(
                "voicelink is not running (no status file at {}).\nlog: {}",
                file.display(),
                log.display()
            );
            return Ok(());
        }
        Err(e) => return Err(e).with_context(|| format!("reading {}", file.display())),
    };
    let status: DaemonStatus = serde_json::from_str(&text)
        .with_context(|| format!("{} is not a valid status file", file.display()))?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    print!("{}", render(&status, pid_alive(status.pid), now, &log));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::status::BackendNames;

    fn status() -> DaemonStatus {
        DaemonStatus {
            pid: 4242,
            updated_ms: 1_000_000,
            state: LinkState::Listening,
            focus: Some("2:Services codex".into()),
            muted: false,
            backends: BackendNames {
                stt: "openai-stt:whisper-1".into(),
                llm: "openai http://127.0.0.1:8080/v1".into(),
                tts: "openai-tts:kokoro:af_heart".into(),
            },
            last_stt_ms: Some(310),
            last_reply_ms: None,
            last_error: None,
        }
    }

    #[test]
    fn live_summary() {
        let out = render(&status(), true, 1_005_000, Path::new("/l/daemon.log"));
        assert!(out.contains("state:     listening"));
        assert!(out.contains("focus:     2:Services codex"));
        assert!(out.contains("updated 5 s ago"));
        assert!(out.contains("speech-to-text 310 ms | reply -"));
        assert!(out.contains("last error: none"));
        assert!(out.contains("log:       /l/daemon.log"));
    }

    #[test]
    fn stale_pid_is_reported() {
        let out = render(&status(), false, 2_000_000, Path::new("/l"));
        assert!(
            out.contains("not running (stale status from pid 4242, last state listening)"),
            "{out}"
        );
        assert!(!out.contains("focus:"));
    }

    #[test]
    fn pid_probe() {
        assert!(pid_alive(std::process::id()));
        assert!(!pid_alive(0));
        assert!(!pid_alive(u32::MAX));
        let mut c = std::process::Command::new("true").spawn().unwrap();
        let pid = c.id();
        c.wait().unwrap();
        assert!(!pid_alive(pid));
    }

    #[test]
    fn errors_and_age() {
        let mut s = status();
        s.last_error = Some("boom".into());
        s.muted = true;
        let out = render(&s, true, 1_000_000 + 7_300_000, Path::new("/l"));
        assert!(
            out.contains("last error: boom") && out.contains("muted") && out.contains("2 h ago")
        );
    }
}
