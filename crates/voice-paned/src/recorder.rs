//! `voice-paned record`: the hook recorder.
//!
//! Runs inside an agent's hook, so it must print **nothing** to stdout (hook
//! stdout reaches the model), always exit 0, and finish quickly. Every failure
//! is swallowed.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use voicelink_proto::{AgentEvent, AgentKind, EventKind, SpoolRecord};

use crate::spool;

/// Which agent invoked the recorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Source {
    /// Claude Code hook: JSON on stdin.
    Claude,
    /// Codex `notify` program: JSON as the last argument.
    Codex,
}

/// Arguments of `voice-paned record`.
#[derive(Debug, clap::Args)]
pub struct RecordArgs {
    /// Agent that invoked the recorder.
    #[arg(long, value_enum)]
    pub source: Source,
    /// Spool file (default: `$HOME/.local/state/voicelink/events.jsonl`).
    #[arg(long)]
    pub spool: Option<PathBuf>,
    /// Payload JSON (Codex passes it as the last argument).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub json: Vec<String>,
}

/// Longest wait for `tmux display-message` before giving up on the window index.
const TMUX_TIMEOUT: Duration = Duration::from_millis(500);
/// Cap on the hook payload read from stdin.
const MAX_STDIN: u64 = 4 << 20;

/// Entry point; never fails and never prints.
pub fn run(args: RecordArgs) {
    let _ = std::panic::catch_unwind(|| record(args));
}

fn record(args: RecordArgs) -> Option<()> {
    let pane = std::env::var("TMUX_PANE").ok().filter(|p| !p.is_empty())?;
    let payload = match args.source {
        Source::Claude => {
            let mut s = String::new();
            std::io::stdin()
                .take(MAX_STDIN)
                .read_to_string(&mut s)
                .ok()?;
            s
        }
        Source::Codex => args.json.last()?.clone(),
    };
    let window = window_index(&pane);
    let event = event_from_payload(args.source, &payload, pane, window, now_ms())?;
    let path = args.spool.or_else(spool::default_path)?;
    spool::append(&path, &SpoolRecord::new(event)).ok()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Window index of `pane` on the server named by `$TMUX`, with a timeout.
fn window_index(pane: &str) -> Option<u32> {
    let mut child = Command::new("tmux")
        .args(["display-message", "-p", "-t", pane, "#{window_index}"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait().ok()? {
            Some(status) if status.success() => break,
            Some(_) => return None,
            None if start.elapsed() > TMUX_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            None => std::thread::sleep(Duration::from_millis(5)),
        }
    }
    let mut out = String::new();
    child.stdout.take()?.read_to_string(&mut out).ok()?;
    out.trim().parse().ok()
}

/// Maps a hook payload to an event, or `None` for payloads to ignore.
pub fn event_from_payload(
    source: Source,
    payload: &str,
    pane: String,
    window: Option<u32>,
    ts_ms: u64,
) -> Option<AgentEvent> {
    let v: Value = serde_json::from_str(payload).ok()?;
    let str_field = |name: &str| {
        v.get(name)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    let (agent, kind, text) = match source {
        Source::Claude => {
            let (kind, text) = match v.get("hook_event_name").and_then(Value::as_str)? {
                "Stop" => (EventKind::TurnEnd, str_field("last_assistant_message")),
                "Notification" => {
                    let kind = match v.get("notification_type").and_then(Value::as_str) {
                        Some("permission_prompt" | "elicitation_dialog") => EventKind::Prompt,
                        // The Stop hook already announces the idle state.
                        Some("idle_prompt") => return None,
                        _ => EventKind::Notice,
                    };
                    (kind, str_field("message"))
                }
                _ => return None,
            };
            (AgentKind::Claude, kind, text)
        }
        Source::Codex => {
            if v.get("type").and_then(Value::as_str)? != "agent-turn-complete" {
                return None;
            }
            (
                AgentKind::Codex,
                EventKind::TurnEnd,
                str_field("last-assistant-message"),
            )
        }
    };
    Some(AgentEvent {
        ts_ms,
        pane,
        window,
        agent,
        kind,
        text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(source: Source, payload: &str) -> Option<AgentEvent> {
        event_from_payload(source, payload, "%4".into(), Some(2), 99)
    }

    #[test]
    fn claude_stop_from_a_real_payload() {
        let payload = r#"{"session_id":"s","transcript_path":"/t.jsonl","cwd":"/w","prompt_id":"p","permission_mode":"default","effort":{"level":"medium"},"hook_event_name":"Stop","stop_hook_active":false,"last_assistant_message":"All done.","background_tasks":[],"session_crons":[]}"#;
        let e = ev(Source::Claude, payload).unwrap();
        assert_eq!(e.kind, EventKind::TurnEnd);
        assert_eq!(e.agent, AgentKind::Claude);
        assert_eq!(e.text.as_deref(), Some("All done."));
        assert_eq!((e.pane.as_str(), e.window, e.ts_ms), ("%4", Some(2), 99));
    }

    #[test]
    fn claude_notifications() {
        let n = |t: &str| {
            format!(
                r#"{{"hook_event_name":"Notification","message":"m","notification_type":"{t}"}}"#
            )
        };
        assert_eq!(
            ev(Source::Claude, &n("permission_prompt")).unwrap().kind,
            EventKind::Prompt
        );
        assert_eq!(
            ev(Source::Claude, &n("elicitation_dialog")).unwrap().kind,
            EventKind::Prompt
        );
        assert!(ev(Source::Claude, &n("idle_prompt")).is_none());
        assert_eq!(
            ev(Source::Claude, &n("auth_success")).unwrap().kind,
            EventKind::Notice
        );
        let untyped = ev(
            Source::Claude,
            r#"{"hook_event_name":"Notification","message":"hey"}"#,
        )
        .unwrap();
        assert_eq!(
            (untyped.kind, untyped.text.as_deref()),
            (EventKind::Notice, Some("hey"))
        );
    }

    #[test]
    fn claude_ignores_unknown_and_garbage() {
        assert!(ev(Source::Claude, r#"{"hook_event_name":"PreToolUse"}"#).is_none());
        assert!(ev(Source::Claude, r#"{"no":"event"}"#).is_none());
        assert!(ev(Source::Claude, "not json").is_none());
        assert!(ev(Source::Claude, "").is_none());
    }

    #[test]
    fn claude_stop_without_message_has_no_text() {
        let e = ev(
            Source::Claude,
            r#"{"hook_event_name":"Stop","last_assistant_message":""}"#,
        )
        .unwrap();
        assert_eq!(e.text, None);
    }

    #[test]
    fn codex_turn_complete_from_a_real_payload() {
        let payload = r#"{"type":"agent-turn-complete","thread-id":"t","turn-id":"u","cwd":"/w","client":"codex_exec","input-messages":["hi"],"last-assistant-message":"probe ok"}"#;
        let e = ev(Source::Codex, payload).unwrap();
        assert_eq!(
            (e.kind, e.agent, e.text.as_deref()),
            (EventKind::TurnEnd, AgentKind::Codex, Some("probe ok"))
        );
        assert!(ev(Source::Codex, r#"{"type":"other"}"#).is_none());
        assert!(ev(Source::Codex, r#"{"nothing":1}"#).is_none());
    }
}
