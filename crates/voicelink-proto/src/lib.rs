//! Wire contract between a voice client and a tmux pane controller.
//!
//! A *client* (the `voicelink` daemon on a laptop) talks to a *controller*
//! (`voice-paned`, running beside a tmux server) over one bidirectional byte
//! stream, typically `ssh -T host … voice-paned --stdio`. Every message is one
//! JSON object on one line ([JSON Lines](https://jsonlines.org/)), tagged by a
//! `"type"` field.
//!
//! # Session shape
//!
//! 1. The client sends [`ClientMessage::Hello`] carrying [`PROTOCOL_VERSION`].
//! 2. The controller answers [`ServerMessage::Welcome`] (or an
//!    [`ErrorCode::ProtocolMismatch`] error and closes).
//! 3. Requests carry a client-chosen [`RequestId`]; each is answered by exactly
//!    one reply bearing the same id ([`ServerMessage::Ack`],
//!    [`ServerMessage::Panes`], [`ServerMessage::PaneText`],
//!    [`ServerMessage::Pong`] or [`ServerMessage::Error`]).
//! 4. Unsolicited [`ServerMessage::Focus`] and [`ServerMessage::Event`]
//!    messages may arrive at any time.
//!
//! # Focus-only writes
//!
//! Requests that type into a pane ([`ClientMessage::Send`],
//! [`ClientMessage::Keys`], [`ClientMessage::Interrupt`]) carry **no pane
//! argument**: they always act on the pane currently focused in the client's
//! view session. Acting on a background pane is only possible by first
//! moving focus to it with [`ClientMessage::SelectPane`], which the operator
//! can see. Reads ([`ClientMessage::ReadPane`], [`ClientMessage::ListPanes`]) may
//! address any pane.
//!
//! # Event spool
//!
//! Agent hooks (a Claude Code `Stop` hook, a Codex `notify` program) append
//! [`SpoolRecord`] lines to a bounded file; the controller tails it and relays
//! each record as [`ServerMessage::Event`].
//!
//! # Example
//!
//! ```
//! use voicelink_proto::{ClientMessage, ServerMessage, PROTOCOL_VERSION, encode_line, decode_line};
//!
//! let hello = ClientMessage::Hello {
//!     proto: PROTOCOL_VERSION,
//!     client: "voicelink/0.1.0".into(),
//!     view_session: "remote-laptop".into(),
//! };
//! let line = encode_line(&hello).unwrap();
//! assert!(line.ends_with('\n'));
//! let back: ClientMessage = decode_line(&line).unwrap();
//! assert_eq!(back, hello);
//!
//! let pong: ServerMessage = decode_line(r#"{"type":"pong","id":7}"#).unwrap();
//! assert_eq!(pong, ServerMessage::Pong { id: 7 });
//! ```

use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// Version of this wire contract. A controller refuses a [`ClientMessage::Hello`]
/// whose `proto` differs, answering [`ErrorCode::ProtocolMismatch`].
pub const PROTOCOL_VERSION: u32 = 1;

/// Upper bound on one encoded line, in bytes. Readers should reject longer
/// lines instead of buffering without limit.
pub const MAX_LINE_BYTES: usize = 1 << 20;

/// Upper bound on [`ReadPane`](ClientMessage::ReadPane) `lines`.
pub const MAX_READ_LINES: u32 = 2000;

/// Client-chosen correlation id, echoed in the reply.
pub type RequestId = u64;

/// A tmux pane id, including its `%` prefix (for example `"%12"`).
pub type PaneId = String;

/// Messages sent by the client to the controller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Opens the session. Must be the first message.
    Hello {
        /// Must equal [`PROTOCOL_VERSION`].
        proto: u32,
        /// Free-form client identifier, for logs (for example `"voicelink/0.1.0"`).
        client: String,
        /// Name of the tmux session whose focus the controller follows. The
        /// controller creates it as a grouped session if it does not exist.
        view_session: String,
    },
    /// Lists every pane of the shared window group.
    ListPanes {
        /// Correlation id.
        id: RequestId,
    },
    /// Reads the visible text of any pane (read-only).
    ReadPane {
        /// Correlation id.
        id: RequestId,
        /// Pane to read.
        pane: PaneId,
        /// Number of trailing lines to capture, at most [`MAX_READ_LINES`].
        lines: u32,
    },
    /// Types literal text into the focused pane.
    Send {
        /// Correlation id.
        id: RequestId,
        /// Text typed literally (no key-name interpretation).
        text: String,
        /// Press Enter after the text.
        submit: bool,
        /// Type even if the pane's input line already holds text. When false
        /// the controller answers [`ErrorCode::InputNotEmpty`] instead.
        force: bool,
    },
    /// Presses named keys in the focused pane, in order.
    Keys {
        /// Correlation id.
        id: RequestId,
        /// Keys to press.
        keys: Vec<Key>,
    },
    /// Sends Escape to the focused pane, interrupting a running agent turn.
    Interrupt {
        /// Correlation id.
        id: RequestId,
    },
    /// Moves the view session's focus to a pane: selects its window, then
    /// makes it that window's active pane. The active pane is per-window
    /// state, so other sessions viewing the same window see the change too.
    SelectPane {
        /// Correlation id.
        id: RequestId,
        /// Pane to focus (pane ids survive window renumbering).
        pane: PaneId,
    },
    /// Liveness probe.
    Ping {
        /// Correlation id.
        id: RequestId,
    },
    /// Ends the session; the controller exits cleanly.
    Bye,
}

/// Messages sent by the controller to the client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Accepts a [`ClientMessage::Hello`].
    Welcome {
        /// The controller's [`PROTOCOL_VERSION`].
        proto: u32,
        /// Free-form controller identifier (for example `"voice-paned/0.1.0"`).
        server: String,
        /// The view session being followed.
        view_session: String,
        /// Focus at the moment of the handshake.
        focus: Focus,
    },
    /// Focus changed in the view session (pushed, unsolicited).
    Focus(Focus),
    /// Reply to [`ClientMessage::ListPanes`].
    Panes {
        /// Correlation id.
        id: RequestId,
        /// Every pane, in window then pane order.
        panes: Vec<PaneInfo>,
    },
    /// Reply to [`ClientMessage::ReadPane`].
    PaneText {
        /// Correlation id.
        id: RequestId,
        /// Pane that was read.
        pane: PaneId,
        /// Captured text, lines joined with `\n`, trailing blank lines removed.
        text: String,
    },
    /// Success reply to a write or [`ClientMessage::SelectPane`].
    Ack {
        /// Correlation id.
        id: RequestId,
    },
    /// Reply to [`ClientMessage::Ping`].
    Pong {
        /// Correlation id.
        id: RequestId,
    },
    /// Failure reply, or a session-level error when `id` is `None`.
    Error {
        /// Correlation id of the failed request, if any.
        id: Option<RequestId>,
        /// Machine-readable reason.
        code: ErrorCode,
        /// Human-readable detail.
        message: String,
    },
    /// An agent event read from the spool (pushed, unsolicited).
    Event(AgentEvent),
}

/// The focused pane of the view session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Focus {
    /// Current window index.
    pub window: u32,
    /// Current window name.
    pub window_name: String,
    /// Active pane in that window.
    pub pane: PaneId,
    /// The pane's foreground command (`pane_current_command`).
    pub command: String,
    /// Agent recognised from `command`.
    pub agent: AgentKind,
}

/// One pane of the shared window group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaneInfo {
    /// Pane id.
    pub pane: PaneId,
    /// Window index.
    pub window: u32,
    /// Window name.
    pub window_name: String,
    /// The pane's foreground command.
    pub command: String,
    /// Agent recognised from `command`.
    pub agent: AgentKind,
    /// Whether this is the active pane of its window.
    pub active: bool,
    /// Whether this pane is the view session's focus.
    pub focused: bool,
}

/// Agent running in a pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    /// Claude Code.
    Claude,
    /// OpenAI Codex CLI.
    Codex,
    /// Anything else (a shell, an editor, a TUI).
    Other,
}

impl AgentKind {
    /// Classifies a tmux `pane_current_command`.
    ///
    /// ```
    /// use voicelink_proto::AgentKind;
    /// assert_eq!(AgentKind::from_command("claude"), AgentKind::Claude);
    /// assert_eq!(AgentKind::from_command("codex"), AgentKind::Codex);
    /// assert_eq!(AgentKind::from_command("fish"), AgentKind::Other);
    /// ```
    pub fn from_command(command: &str) -> Self {
        let base = command.rsplit('/').next().unwrap_or(command);
        match base {
            "claude" => AgentKind::Claude,
            "codex" | "codex-cli" => AgentKind::Codex,
            _ => AgentKind::Other,
        }
    }
}

/// A key for [`ClientMessage::Keys`]. Deliberately small: free text goes
/// through [`ClientMessage::Send`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "key", content = "value", rename_all = "snake_case")]
pub enum Key {
    /// Return.
    Enter,
    /// Escape.
    Escape,
    /// Tab.
    Tab,
    /// Shift-Tab.
    BackTab,
    /// Space bar.
    Space,
    /// Cursor up.
    Up,
    /// Cursor down.
    Down,
    /// Cursor left.
    Left,
    /// Cursor right.
    Right,
    /// A digit `0`–`9` (menu selection).
    Digit(u8),
    /// An ASCII letter `a`–`z` or `A`–`Z` (single-key answers such as `y`).
    Letter(char),
}

impl Key {
    /// The tmux `send-keys` token for this key, or `None` when the payload is
    /// out of range (a digit above 9, a non-ASCII-letter character).
    ///
    /// ```
    /// use voicelink_proto::Key;
    /// assert_eq!(Key::Enter.tmux_name().as_deref(), Some("Enter"));
    /// assert_eq!(Key::Digit(3).tmux_name().as_deref(), Some("3"));
    /// assert_eq!(Key::Digit(12).tmux_name(), None);
    /// assert_eq!(Key::Letter('y').tmux_name().as_deref(), Some("y"));
    /// assert_eq!(Key::Letter(';').tmux_name(), None);
    /// ```
    pub fn tmux_name(&self) -> Option<String> {
        Some(match self {
            Key::Enter => "Enter".into(),
            Key::Escape => "Escape".into(),
            Key::Tab => "Tab".into(),
            Key::BackTab => "BTab".into(),
            Key::Space => "Space".into(),
            Key::Up => "Up".into(),
            Key::Down => "Down".into(),
            Key::Left => "Left".into(),
            Key::Right => "Right".into(),
            Key::Digit(d) if *d <= 9 => d.to_string(),
            Key::Letter(c) if c.is_ascii_alphabetic() => c.to_string(),
            Key::Digit(_) | Key::Letter(_) => return None,
        })
    }
}

/// Machine-readable error reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// `Hello.proto` differs from the controller's [`PROTOCOL_VERSION`].
    ProtocolMismatch,
    /// A request arrived before `Hello`, or a second `Hello` arrived.
    BadHandshake,
    /// The line was not a valid message, or a field was out of range.
    BadRequest,
    /// The view session has no focused pane.
    NoFocus,
    /// `Send` without `force` while the focused pane's input line holds text.
    InputNotEmpty,
    /// The named pane or window does not exist.
    NotFound,
    /// tmux reported an error.
    Tmux,
}

/// What happened in an agent pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// The agent finished a turn and is waiting for input.
    TurnEnd,
    /// The agent is blocked on a question, menu or permission prompt.
    Prompt,
    /// Any other notification the agent raised.
    Notice,
}

/// An agent event, as relayed in [`ServerMessage::Event`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentEvent {
    /// Milliseconds since the Unix epoch when the event was recorded.
    pub ts_ms: u64,
    /// Pane the agent runs in (from `$TMUX_PANE`).
    pub pane: PaneId,
    /// Window index of that pane, when the recorder could resolve it.
    pub window: Option<u32>,
    /// Which agent raised it.
    pub agent: AgentKind,
    /// What happened.
    pub kind: EventKind,
    /// Source text for a spoken summary: the agent's last message, or the
    /// prompt's question. Recorders truncate it to [`MAX_EVENT_TEXT_CHARS`].
    pub text: Option<String>,
}

/// Upper bound on [`AgentEvent::text`], in characters.
pub const MAX_EVENT_TEXT_CHARS: usize = 4000;

/// One line of the event spool: an [`AgentEvent`] plus a record version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpoolRecord {
    /// Record format version; currently `1`.
    pub v: u32,
    /// The event.
    #[serde(flatten)]
    pub event: AgentEvent,
}

/// Current [`SpoolRecord::v`].
pub const SPOOL_RECORD_VERSION: u32 = 1;

/// Spool file location relative to the controller user's home directory.
/// Recorders rotate it to `events.jsonl.1` beyond [`SPOOL_MAX_BYTES`].
pub const SPOOL_RELATIVE_PATH: &str = ".local/state/voicelink/events.jsonl";

/// Size at which recorders rotate the spool.
pub const SPOOL_MAX_BYTES: u64 = 1 << 20;

impl SpoolRecord {
    /// Wraps an event at the current record version, truncating its text to
    /// [`MAX_EVENT_TEXT_CHARS`].
    ///
    /// ```
    /// use voicelink_proto::{AgentEvent, AgentKind, EventKind, SpoolRecord};
    /// let rec = SpoolRecord::new(AgentEvent {
    ///     ts_ms: 1, pane: "%3".into(), window: Some(2), agent: AgentKind::Codex,
    ///     kind: EventKind::TurnEnd, text: Some("x".repeat(5000)),
    /// });
    /// assert_eq!(rec.event.text.unwrap().chars().count(), 4000);
    /// ```
    pub fn new(mut event: AgentEvent) -> Self {
        if let Some(text) = event.text.as_mut()
            && let Some((cut, _)) = text.char_indices().nth(MAX_EVENT_TEXT_CHARS)
        {
            text.truncate(cut);
        }
        SpoolRecord {
            v: SPOOL_RECORD_VERSION,
            event,
        }
    }
}

/// Errors from [`encode_line`] and [`decode_line`].
#[derive(Debug, thiserror::Error)]
pub enum WireError {
    /// The line exceeds [`MAX_LINE_BYTES`].
    #[error("line of {0} bytes exceeds the {MAX_LINE_BYTES}-byte limit")]
    TooLong(usize),
    /// The JSON was malformed or did not match the message schema.
    #[error("invalid message: {0}")]
    Json(#[from] serde_json::Error),
}

/// Encodes a message as one JSON line, newline included.
///
/// # Errors
///
/// [`WireError::TooLong`] when the encoding exceeds [`MAX_LINE_BYTES`].
pub fn encode_line<T: Serialize>(message: &T) -> Result<String, WireError> {
    let mut line = serde_json::to_string(message)?;
    line.push('\n');
    if line.len() > MAX_LINE_BYTES {
        return Err(WireError::TooLong(line.len()));
    }
    Ok(line)
}

/// Decodes one JSON line (a trailing `\n` or `\r\n` is ignored).
///
/// # Errors
///
/// [`WireError::TooLong`] for oversized input, [`WireError::Json`] for
/// anything that is not a valid message.
pub fn decode_line<T: DeserializeOwned>(line: &str) -> Result<T, WireError> {
    if line.len() > MAX_LINE_BYTES {
        return Err(WireError::TooLong(line.len()));
    }
    Ok(serde_json::from_str(line.trim_end_matches(['\n', '\r']))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn focus() -> Focus {
        Focus {
            window: 2,
            window_name: "Services".into(),
            pane: "%7".into(),
            command: "codex".into(),
            agent: AgentKind::Codex,
        }
    }

    #[test]
    fn every_client_message_round_trips() {
        let all = vec![
            ClientMessage::Hello {
                proto: 1,
                client: "c".into(),
                view_session: "remote-x".into(),
            },
            ClientMessage::ListPanes { id: 1 },
            ClientMessage::ReadPane {
                id: 2,
                pane: "%1".into(),
                lines: 40,
            },
            ClientMessage::Send {
                id: 3,
                text: "fix the build\n".into(),
                submit: true,
                force: false,
            },
            ClientMessage::Keys {
                id: 4,
                keys: vec![Key::Digit(2), Key::Enter, Key::Letter('y')],
            },
            ClientMessage::Interrupt { id: 5 },
            ClientMessage::SelectPane {
                id: 6,
                pane: "%12".into(),
            },
            ClientMessage::Ping { id: 7 },
            ClientMessage::Bye,
        ];
        for msg in all {
            let line = encode_line(&msg).unwrap();
            assert_eq!(
                line.matches('\n').count(),
                1,
                "embedded newline must be escaped: {line}"
            );
            assert_eq!(decode_line::<ClientMessage>(&line).unwrap(), msg);
        }
    }

    #[test]
    fn every_server_message_round_trips() {
        let event = AgentEvent {
            ts_ms: 1_761_000_000_000,
            pane: "%7".into(),
            window: Some(2),
            agent: AgentKind::Codex,
            kind: EventKind::Prompt,
            text: Some("Allow command?".into()),
        };
        let all = vec![
            ServerMessage::Welcome {
                proto: 1,
                server: "s".into(),
                view_session: "remote-x".into(),
                focus: focus(),
            },
            ServerMessage::Focus(focus()),
            ServerMessage::Panes {
                id: 1,
                panes: vec![PaneInfo {
                    pane: "%7".into(),
                    window: 2,
                    window_name: "Services".into(),
                    command: "codex".into(),
                    agent: AgentKind::Codex,
                    active: true,
                    focused: true,
                }],
            },
            ServerMessage::PaneText {
                id: 2,
                pane: "%7".into(),
                text: "a\nb".into(),
            },
            ServerMessage::Ack { id: 3 },
            ServerMessage::Pong { id: 4 },
            ServerMessage::Error {
                id: None,
                code: ErrorCode::BadHandshake,
                message: "hello first".into(),
            },
            ServerMessage::Error {
                id: Some(5),
                code: ErrorCode::InputNotEmpty,
                message: "busy".into(),
            },
            ServerMessage::Event(event),
        ];
        for msg in all {
            let line = encode_line(&msg).unwrap();
            assert_eq!(decode_line::<ServerMessage>(&line).unwrap(), msg);
        }
    }

    #[test]
    fn wire_shape_is_stable() {
        // Pin the JSON shape other implementations rely on.
        let line = encode_line(&ClientMessage::Keys {
            id: 9,
            keys: vec![Key::Digit(1), Key::Enter],
        })
        .unwrap();
        assert_eq!(
            line,
            "{\"type\":\"keys\",\"id\":9,\"keys\":[{\"key\":\"digit\",\"value\":1},{\"key\":\"enter\"}]}\n"
        );
        let focus_line = encode_line(&ServerMessage::Focus(focus())).unwrap();
        let select = encode_line(&ClientMessage::SelectPane {
            id: 2,
            pane: "%12".into(),
        })
        .unwrap();
        assert_eq!(
            select,
            "{\"type\":\"select_pane\",\"id\":2,\"pane\":\"%12\"}\n"
        );
        assert_eq!(
            focus_line,
            "{\"type\":\"focus\",\"window\":2,\"window_name\":\"Services\",\"pane\":\"%7\",\"command\":\"codex\",\"agent\":\"codex\"}\n"
        );
    }

    #[test]
    fn spool_record_is_flat() {
        let rec = SpoolRecord::new(AgentEvent {
            ts_ms: 5,
            pane: "%1".into(),
            window: None,
            agent: AgentKind::Claude,
            kind: EventKind::TurnEnd,
            text: None,
        });
        let line = encode_line(&rec).unwrap();
        assert_eq!(
            line,
            "{\"v\":1,\"ts_ms\":5,\"pane\":\"%1\",\"window\":null,\"agent\":\"claude\",\"kind\":\"turn_end\",\"text\":null}\n"
        );
        assert_eq!(decode_line::<SpoolRecord>(&line).unwrap(), rec);
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        let rec = SpoolRecord::new(AgentEvent {
            ts_ms: 0,
            pane: "%1".into(),
            window: None,
            agent: AgentKind::Claude,
            kind: EventKind::TurnEnd,
            text: Some("é".repeat(MAX_EVENT_TEXT_CHARS + 10)),
        });
        assert_eq!(
            rec.event.text.unwrap().chars().count(),
            MAX_EVENT_TEXT_CHARS
        );
    }

    #[test]
    fn rejects_oversized_and_malformed_lines() {
        let huge = format!(
            "{{\"type\":\"ping\",\"id\":1,\"pad\":\"{}\"}}",
            "x".repeat(MAX_LINE_BYTES)
        );
        assert!(matches!(
            decode_line::<ClientMessage>(&huge),
            Err(WireError::TooLong(_))
        ));
        assert!(matches!(
            decode_line::<ClientMessage>("{\"type\":\"nope\"}"),
            Err(WireError::Json(_))
        ));
        assert!(matches!(
            decode_line::<ClientMessage>("not json"),
            Err(WireError::Json(_))
        ));
        let big = ServerMessage::PaneText {
            id: 1,
            pane: "%1".into(),
            text: "x".repeat(MAX_LINE_BYTES),
        };
        assert!(matches!(encode_line(&big), Err(WireError::TooLong(_))));
    }

    #[test]
    fn crlf_is_tolerated() {
        assert_eq!(
            decode_line::<ClientMessage>("{\"type\":\"bye\"}\r\n").unwrap(),
            ClientMessage::Bye
        );
    }

    #[test]
    fn agent_kind_handles_paths() {
        assert_eq!(
            AgentKind::from_command("/usr/bin/claude"),
            AgentKind::Claude
        );
        assert_eq!(AgentKind::from_command("node"), AgentKind::Other);
        assert_eq!(AgentKind::from_command(""), AgentKind::Other);
    }
}
