//! The stdio controller: one process per client connection.
//!
//! # View session lifecycle (verified on tmux 3.8, see `tests/tmux_semantics.rs`)
//!
//! * The view session is a *grouped* session of the base session: same
//!   windows, independent current window. Killing or detaching it never closes
//!   the shared windows.
//! * The controller never sets `destroy-unattached`. tmux destroys a session
//!   the moment that option is set on it while no client is attached, so the
//!   launcher sets it in the same invocation as its own attach
//!   (`tmux attach -t =V \; set-option -t =V: destroy-unattached on`). Once the
//!   launcher has armed it, a session survives while any client (terminal or
//!   control) is attached, including when the terminal detaches and the
//!   control client remains; it is destroyed when the last client leaves.
//!   Without the option (a session the controller created itself) it simply
//!   persists after the controller exits.
//! * The controller never runs `kill-session`.
//! * The control client uses `ignore-size,no-output`: it neither resizes the
//!   operator's windows (even if it declared a size) nor receives pane output.
//!   The controller never sends `refresh-client -C`.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tmux_control::{
    Client, ClientOptions, Error as TmuxError, Notification, Notifications, Socket,
};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use voicelink_proto::{
    AgentEvent, AgentKind, ClientMessage, ErrorCode, EventKind, Focus, MAX_LINE_BYTES,
    MAX_READ_LINES, PROTOCOL_VERSION, PaneInfo, RequestId, ServerMessage, decode_line, encode_line,
};

use crate::input::{self, InputState};
use crate::prompt;
use crate::spool::SpoolTail;
use crate::tmuxq::{Tmux, agent_of, valid_pane_id};

/// Controller settings.
pub struct Config {
    /// Which tmux server to use.
    pub socket: Socket,
    /// The shared base session whose windows the view session mirrors.
    pub session: String,
    /// The event spool file.
    pub spool: PathBuf,
}

const SPOOL_POLL: Duration = Duration::from_millis(200);
const SCAN_INTERVAL: Duration = Duration::from_millis(1500);
/// A screen-detected prompt is suppressed if a hook announced one this recently.
const HOOK_PROMPT_GRACE: Duration = Duration::from_secs(10);
const ATTACH_ATTEMPTS: usize = 5;
const NOT_EMPTY_PROMPT: &str =
    "the agent is showing a prompt; answer it with keys, or send with force";

/// What a request handler can fail with.
enum Fail {
    /// Reply with an error and carry on.
    Reply(ErrorCode, String),
    /// The tmux connection is gone; end the session.
    Fatal(String),
}

impl From<TmuxError> for Fail {
    fn from(e: TmuxError) -> Self {
        match e {
            TmuxError::Closed(_) | TmuxError::Io(_) => Fail::Fatal(e.to_string()),
            other => Fail::Reply(ErrorCode::Tmux, other.to_string()),
        }
    }
}

fn reply_err(code: ErrorCode, msg: impl Into<String>) -> Fail {
    Fail::Reply(code, msg.into())
}

enum Line {
    Text(String),
    TooLong,
    NotUtf8,
}

/// Reads newline-terminated lines of at most [`MAX_LINE_BYTES`], reporting
/// over-long ones without buffering them.
async fn read_lines<R: AsyncBufRead + Unpin>(mut input: R, tx: mpsc::Sender<Line>) {
    let mut buf: Vec<u8> = Vec::new();
    let mut overflow = false;
    loop {
        let chunk = match input.fill_buf().await {
            Ok(c) if !c.is_empty() => c,
            _ => break,
        };
        let newline = chunk.iter().position(|&b| b == b'\n');
        let take = newline.map_or(chunk.len(), |p| p + 1);
        if !overflow {
            buf.extend_from_slice(&chunk[..take]);
            if buf.len() > MAX_LINE_BYTES + 1 {
                overflow = true;
                buf.clear();
            }
        }
        input.consume(take);
        if newline.is_some() {
            let item = if overflow {
                Line::TooLong
            } else {
                buf.pop();
                match String::from_utf8(std::mem::take(&mut buf)) {
                    Ok(s) => Line::Text(s),
                    Err(_) => Line::NotUtf8,
                }
            };
            overflow = false;
            buf.clear();
            if tx.send(item).await.is_err() {
                return;
            }
        }
    }
    // A final unterminated line counts as a line.
    if !overflow
        && !buf.is_empty()
        && let Ok(s) = String::from_utf8(buf)
    {
        let _ = tx.send(Line::Text(s)).await;
    }
}

struct Out<W> {
    w: W,
}

impl<W: AsyncWrite + Unpin> Out<W> {
    async fn send(&mut self, msg: &ServerMessage) -> std::io::Result<()> {
        let line = match encode_line(msg) {
            Ok(l) => l,
            Err(e) => {
                let id = match msg {
                    ServerMessage::Panes { id, .. } | ServerMessage::PaneText { id, .. } => {
                        Some(*id)
                    }
                    _ => None,
                };
                encode_line(&ServerMessage::Error {
                    id,
                    code: ErrorCode::BadRequest,
                    message: format!("reply too large: {e}"),
                })
                .map_err(std::io::Error::other)?
            }
        };
        self.w.write_all(line.as_bytes()).await?;
        self.w.flush().await
    }

    async fn error(
        &mut self,
        id: Option<RequestId>,
        code: ErrorCode,
        message: impl Into<String>,
    ) -> std::io::Result<()> {
        self.send(&ServerMessage::Error {
            id,
            code,
            message: message.into(),
        })
        .await
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn view_name_problem(view: &str, base: &str) -> Option<&'static str> {
    if view.is_empty() {
        Some("view_session is empty")
    } else if view.contains(['.', ':'])
        || view.chars().any(char::is_control)
        || view.starts_with('=')
    {
        Some("view_session must not contain '.', ':' or control characters, or start with '='")
    } else if view == base {
        Some("view_session must differ from the base session")
    } else {
        None
    }
}

/// Replaces line breaks and other control characters with spaces: a newline
/// would submit early in an agent CLI.
pub fn sanitise(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_control() || c == '\u{2028}' || c == '\u{2029}' {
                ' '
            } else {
                c
            }
        })
        .collect()
}

fn hash_of(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Screen-based prompt detection state.
#[derive(Default)]
struct PromptScanner {
    /// Hash of the prompt already announced, per pane.
    announced: HashMap<String, u64>,
    /// Hash seen on the previous scan, per pane (a prompt must be stable for
    /// two scans before it is announced, which lets a hook-raised event win
    /// the race and avoids announcing half-drawn menus).
    candidate: HashMap<String, u64>,
    /// When a hook last raised a prompt for a pane.
    hook_prompt_at: HashMap<String, Instant>,
}

impl PromptScanner {
    async fn scan(&mut self, tmux: &Tmux) -> Result<Vec<AgentEvent>, Fail> {
        let mut events = Vec::new();
        let mut live = HashSet::new();
        for row in tmux.panes().await? {
            let agent = agent_of(&row.command);
            if agent == AgentKind::Other {
                continue;
            }
            live.insert(row.pane.clone());
            let screen = tmux.capture(&row.pane, 0, false).await?.join("\n");
            let Some(text) = prompt::detect(agent, &screen) else {
                self.announced.remove(&row.pane);
                self.candidate.remove(&row.pane);
                continue;
            };
            let h = hash_of(&text);
            if self.announced.get(&row.pane) == Some(&h) {
                continue;
            }
            if self.candidate.get(&row.pane) != Some(&h) {
                self.candidate.insert(row.pane.clone(), h);
                continue;
            }
            self.announced.insert(row.pane.clone(), h);
            let hooked = self
                .hook_prompt_at
                .get(&row.pane)
                .is_some_and(|t| t.elapsed() < HOOK_PROMPT_GRACE);
            if !hooked {
                events.push(
                    voicelink_proto::SpoolRecord::new(AgentEvent {
                        ts_ms: now_ms(),
                        pane: row.pane.clone(),
                        window: Some(row.window),
                        agent,
                        kind: EventKind::Prompt,
                        text: Some(text),
                    })
                    .event,
                );
            }
        }
        self.announced.retain(|p, _| live.contains(p));
        self.candidate.retain(|p, _| live.contains(p));
        self.hook_prompt_at
            .retain(|_, t| t.elapsed() < HOOK_PROMPT_GRACE);
        Ok(events)
    }
}

/// Runs one controller session. Returns the process exit code.
pub async fn run<R, W>(config: Config, input: R, output: W) -> anyhow::Result<i32>
where
    R: AsyncBufRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin,
{
    let mut out = Out { w: output };
    // Start tailing now so events recorded between start-up and Hello are kept.
    let mut tail = SpoolTail::new(config.spool.clone());
    let (line_tx, mut lines) = mpsc::channel(64);
    tokio::spawn(read_lines(input, line_tx));

    // Handshake.
    let (client_name, view) = {
        let Some(first) = lines.recv().await else {
            return Ok(0);
        };
        match first {
            Line::Text(t) => match decode_line::<ClientMessage>(&t) {
                Ok(ClientMessage::Hello {
                    proto,
                    client,
                    view_session,
                }) => {
                    if proto != PROTOCOL_VERSION {
                        out.error(
                            None,
                            ErrorCode::ProtocolMismatch,
                            format!("client speaks protocol {proto}, controller speaks {PROTOCOL_VERSION}"),
                        )
                        .await?;
                        return Ok(2);
                    }
                    if let Some(problem) = view_name_problem(&view_session, &config.session) {
                        out.error(None, ErrorCode::BadRequest, problem).await?;
                        return Ok(2);
                    }
                    (client, view_session)
                }
                Ok(ClientMessage::Bye) => return Ok(0),
                Ok(_) => {
                    out.error(
                        None,
                        ErrorCode::BadHandshake,
                        "the first message must be hello",
                    )
                    .await?;
                    return Ok(2);
                }
                Err(e) => {
                    out.error(
                        None,
                        ErrorCode::BadHandshake,
                        format!("the first message must be hello: {e}"),
                    )
                    .await?;
                    return Ok(2);
                }
            },
            Line::TooLong | Line::NotUtf8 => {
                out.error(
                    None,
                    ErrorCode::BadHandshake,
                    "the first message must be a valid hello",
                )
                .await?;
                return Ok(2);
            }
        }
    };
    eprintln!("voice-paned: client {client_name:?} view {view:?}");

    let (tmux, mut notes) = match attach(&config, &view).await {
        Ok(v) => v,
        Err(msg) => {
            out.error(None, ErrorCode::Tmux, msg).await?;
            return Ok(1);
        }
    };

    let code = session(&mut out, &mut lines, &mut tail, &tmux, &mut notes).await;
    let Tmux { client, .. } = tmux;
    client.detach().await;
    code
}

/// Ensures the grouped view session and attaches the control client. Does not
/// touch `destroy-unattached` (see the module documentation).
async fn attach(config: &Config, view: &str) -> Result<(Tmux, Notifications), String> {
    let options = ClientOptions::new(view)
        .flag("ignore-size")
        .flag("no-output");
    let mut last = String::from("could not attach");
    for attempt in 0..ATTACH_ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        tmux_control::ensure_grouped_session(&config.socket, &config.session, view)
            .await
            .map_err(|e| e.to_string())?;
        let (client, notes) =
            Client::attach(&config.socket, &options).map_err(|e| e.to_string())?;
        match client.command("display-message -p ok").await {
            Ok(_) => {
                let tmux = Tmux {
                    client,
                    view: view.to_owned(),
                };
                return Ok((tmux, notes));
            }
            // The launcher may have created and lost the session in the
            // same instant; ensure and attach again.
            Err(TmuxError::Closed(reason)) => last = reason.unwrap_or(last),
            Err(e) => return Err(e.to_string()),
        }
    }
    Err(last)
}

async fn session<W: AsyncWrite + Unpin>(
    out: &mut Out<W>,
    lines: &mut mpsc::Receiver<Line>,
    tail: &mut SpoolTail,
    tmux: &Tmux,
    notes: &mut Notifications,
) -> anyhow::Result<i32> {
    let focus = match tmux.focus().await {
        Ok(Some(f)) => f,
        Ok(None) => {
            out.error(
                None,
                ErrorCode::NoFocus,
                "the view session has no focused pane",
            )
            .await?;
            return Ok(1);
        }
        Err(e) => {
            out.error(None, ErrorCode::Tmux, e.to_string()).await?;
            return Ok(1);
        }
    };
    let mut last_focus = Some(focus.clone());
    out.send(&ServerMessage::Welcome {
        proto: PROTOCOL_VERSION,
        server: format!("voice-paned/{}", env!("CARGO_PKG_VERSION")),
        view_session: tmux.view.clone(),
        focus: focus.clone(),
    })
    .await?;

    let mut scanner = PromptScanner::default();
    let mut spool_tick = tokio::time::interval(SPOOL_POLL);
    let mut scan_tick =
        tokio::time::interval_at(tokio::time::Instant::now() + SCAN_INTERVAL, SCAN_INTERVAL);
    spool_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    scan_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        let step: Result<(), Fail> = tokio::select! {
            line = lines.recv() => match line {
                None => return Ok(0),
                Some(line) => match handle_line(out, tmux, line, &mut last_focus).await? {
                    Flow::Continue => Ok(()),
                    Flow::Bye => return Ok(0),
                },
            },
            note = notes.recv() => match note {
                None => Err(Fail::Fatal("tmux control connection closed".into())),
                Some(Notification::Exit { reason }) => {
                    Err(Fail::Fatal(format!("tmux control client exited{}", reason.map(|r| format!(": {r}")).unwrap_or_default())))
                }
                Some(n) if affects_focus(&n) => refresh_focus(out, tmux, &mut last_focus).await,
                Some(_) => Ok(()),
            },
            _ = spool_tick.tick() => {
                for rec in tail.poll() {
                    if rec.event.kind == EventKind::Prompt {
                        scanner.hook_prompt_at.insert(rec.event.pane.clone(), Instant::now());
                    }
                    out.send(&ServerMessage::Event(rec.event)).await?;
                }
                Ok(())
            },
            _ = scan_tick.tick() => {
                async {
                    refresh_focus(out, tmux, &mut last_focus).await?;
                    for event in scanner.scan(tmux).await? {
                        out.send(&ServerMessage::Event(event)).await.map_err(|e| Fail::Fatal(e.to_string()))?;
                    }
                    Ok(())
                }.await
            },
        };
        if let Err(fail) = step {
            match fail {
                Fail::Fatal(msg) => {
                    out.error(None, ErrorCode::Tmux, msg).await?;
                    return Ok(1);
                }
                Fail::Reply(code, msg) => out.error(None, code, msg).await?,
            }
        }
    }
}

fn affects_focus(n: &Notification) -> bool {
    matches!(
        n,
        Notification::SessionWindowChanged { .. }
            | Notification::WindowPaneChanged { .. }
            | Notification::SessionChanged { .. }
            | Notification::WindowRenamed { .. }
            | Notification::WindowClose { .. }
            | Notification::WindowAdd { .. }
            | Notification::LayoutChange { .. }
    )
}

/// Re-reads the focus and pushes it if it changed.
async fn refresh_focus<W: AsyncWrite + Unpin>(
    out: &mut Out<W>,
    tmux: &Tmux,
    last: &mut Option<Focus>,
) -> Result<(), Fail> {
    let Some(current) = tmux.focus().await? else {
        return Ok(());
    };
    if last.as_ref() != Some(&current) {
        *last = Some(current.clone());
        out.send(&ServerMessage::Focus(current))
            .await
            .map_err(|e| Fail::Fatal(e.to_string()))?;
    }
    Ok(())
}

enum Flow {
    Continue,
    Bye,
}

async fn handle_line<W: AsyncWrite + Unpin>(
    out: &mut Out<W>,
    tmux: &Tmux,
    line: Line,
    last_focus: &mut Option<Focus>,
) -> std::io::Result<Flow> {
    let text = match line {
        Line::Text(t) => t,
        Line::TooLong => {
            out.error(
                None,
                ErrorCode::BadRequest,
                format!("line exceeds {MAX_LINE_BYTES} bytes"),
            )
            .await?;
            return Ok(Flow::Continue);
        }
        Line::NotUtf8 => {
            out.error(None, ErrorCode::BadRequest, "line is not valid UTF-8")
                .await?;
            return Ok(Flow::Continue);
        }
    };
    if text.trim().is_empty() {
        return Ok(Flow::Continue);
    }
    let msg = match decode_line::<ClientMessage>(&text) {
        Ok(m) => m,
        Err(e) => {
            // Echo the id when the line is JSON carrying one.
            let id = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v.get("id").and_then(serde_json::Value::as_u64));
            out.error(id, ErrorCode::BadRequest, e.to_string()).await?;
            return Ok(Flow::Continue);
        }
    };
    let id = match &msg {
        ClientMessage::Bye => return Ok(Flow::Bye),
        ClientMessage::Hello { .. } => {
            out.error(None, ErrorCode::BadHandshake, "hello was already received")
                .await?;
            return Ok(Flow::Continue);
        }
        ClientMessage::ListPanes { id }
        | ClientMessage::ReadPane { id, .. }
        | ClientMessage::Send { id, .. }
        | ClientMessage::Keys { id, .. }
        | ClientMessage::Interrupt { id }
        | ClientMessage::SelectPane { id, .. }
        | ClientMessage::Ping { id } => *id,
    };
    match handle_request(tmux, msg).await {
        Ok(reply) => {
            out.send(&reply).await?;
            // A selection moves focus: push it right behind the ack.
            if matches!(reply, ServerMessage::Ack { .. })
                && let Err(Fail::Fatal(m)) = refresh_focus(out, tmux, last_focus).await
            {
                out.error(None, ErrorCode::Tmux, m).await?;
            }
        }
        Err(Fail::Reply(code, message)) => out.error(Some(id), code, message).await?,
        Err(Fail::Fatal(message)) => out.error(Some(id), ErrorCode::Tmux, message).await?,
    }
    Ok(Flow::Continue)
}

async fn handle_request(tmux: &Tmux, msg: ClientMessage) -> Result<ServerMessage, Fail> {
    match msg {
        ClientMessage::Ping { id } => Ok(ServerMessage::Pong { id }),
        ClientMessage::ListPanes { id } => {
            let focus = tmux.focus().await?;
            let panes = tmux
                .panes()
                .await?
                .into_iter()
                .map(|r| PaneInfo {
                    focused: focus.as_ref().is_some_and(|f| f.pane == r.pane),
                    agent: agent_of(&r.command),
                    pane: r.pane,
                    window: r.window,
                    window_name: r.window_name,
                    command: r.command,
                    active: r.active,
                })
                .collect();
            Ok(ServerMessage::Panes { id, panes })
        }
        ClientMessage::ReadPane { id, pane, lines } => {
            if !valid_pane_id(&pane) || !tmux.panes().await?.iter().any(|r| r.pane == pane) {
                return Err(reply_err(
                    ErrorCode::NotFound,
                    format!("no such pane {pane}"),
                ));
            }
            let want = lines.clamp(1, MAX_READ_LINES);
            let mut captured = tmux.capture(&pane, want, false).await?;
            while captured.last().is_some_and(|l| l.trim().is_empty()) {
                captured.pop();
            }
            let keep_from = captured.len().saturating_sub(want as usize);
            Ok(ServerMessage::PaneText {
                id,
                pane,
                text: captured[keep_from..].join("\n"),
            })
        }
        ClientMessage::Send {
            id,
            text,
            submit,
            force,
        } => {
            let focus = tmux
                .focus()
                .await?
                .ok_or_else(|| reply_err(ErrorCode::NoFocus, "no focused pane"))?;
            let text = sanitise(&text);
            if text.is_empty() && !submit {
                return Err(reply_err(ErrorCode::BadRequest, "nothing to send"));
            }
            if !force && !text.is_empty() {
                check_input(tmux, &focus).await?;
            }
            if !text.is_empty() {
                tmux.cmd(["send-keys", "-t", &focus.pane, "-l", "--", &text])
                    .await?;
            }
            if submit {
                tmux.cmd(["send-keys", "-t", &focus.pane, "Enter"]).await?;
            }
            Ok(ServerMessage::Ack { id })
        }
        ClientMessage::Keys { id, keys } => {
            let names: Option<Vec<String>> =
                keys.iter().map(voicelink_proto::Key::tmux_name).collect();
            let names =
                names.ok_or_else(|| reply_err(ErrorCode::BadRequest, "a key is out of range"))?;
            if names.is_empty() {
                return Err(reply_err(ErrorCode::BadRequest, "no keys"));
            }
            let focus = tmux
                .focus()
                .await?
                .ok_or_else(|| reply_err(ErrorCode::NoFocus, "no focused pane"))?;
            let mut args = vec!["send-keys".to_owned(), "-t".into(), focus.pane];
            args.extend(names);
            tmux.cmd(args).await?;
            Ok(ServerMessage::Ack { id })
        }
        ClientMessage::Interrupt { id } => {
            let focus = tmux
                .focus()
                .await?
                .ok_or_else(|| reply_err(ErrorCode::NoFocus, "no focused pane"))?;
            tmux.cmd(["send-keys", "-t", &focus.pane, "Escape"]).await?;
            Ok(ServerMessage::Ack { id })
        }
        ClientMessage::SelectPane { id, pane } => {
            let rows = tmux.panes().await?;
            let row = rows
                .iter()
                .find(|r| valid_pane_id(&pane) && r.pane == pane)
                .ok_or_else(|| reply_err(ErrorCode::NotFound, format!("no such pane {pane}")))?;
            // Window ids are stable across renumbering; the view session's
            // own current window changes, the base session's does not.
            tmux.cmd([
                "select-window",
                "-t",
                &format!("={}:{}", tmux.view, row.window_id),
            ])
            .await?;
            tmux.cmd(["select-pane", "-t", &pane]).await?;
            Ok(ServerMessage::Ack { id })
        }
        ClientMessage::Hello { .. } | ClientMessage::Bye => unreachable_request(),
    }
}

fn unreachable_request() -> Result<ServerMessage, Fail> {
    Err(reply_err(ErrorCode::BadRequest, "unexpected message"))
}

/// `InputNotEmpty` unless the focused pane's input line is empty.
async fn check_input(tmux: &Tmux, focus: &Focus) -> Result<(), Fail> {
    let not_empty = |msg: &str| reply_err(ErrorCode::InputNotEmpty, msg);
    match focus.agent {
        AgentKind::Claude | AgentKind::Codex => {
            let ansi = tmux.capture(&focus.pane, 0, true).await?.join("\n");
            let plain = input::strip_ansi(&ansi);
            if prompt::detect(focus.agent, &plain).is_some() {
                return Err(not_empty(NOT_EMPTY_PROMPT));
            }
            if input::input_state(focus.agent, &ansi, "") == InputState::HasText {
                return Err(not_empty("the input line already holds text"));
            }
        }
        AgentKind::Other => {
            let left = tmux.cursor_row_text(&focus.pane).await?;
            if input::input_state(AgentKind::Other, "", &left) == InputState::HasText {
                return Err(not_empty("the input line already holds text"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitise_replaces_control_characters() {
        assert_eq!(sanitise("a\nb\r\nc\td\u{1b}[31m"), "a b  c d [31m");
        assert_eq!(
            sanitise("keep unicode \u{e9}\u{65e5}"),
            "keep unicode \u{e9}\u{65e5}"
        );
        assert_eq!(sanitise("sep\u{2028}ara\u{85}tor"), "sep ara tor");
    }

    #[test]
    fn view_names() {
        assert!(view_name_problem("remote-laptop", "agentbox").is_none());
        for bad in ["", "a.b", "a:b", "=x", "agentbox", "a\nb"] {
            assert!(view_name_problem(bad, "agentbox").is_some(), "{bad:?}");
        }
    }

    #[tokio::test]
    async fn line_reader_limits_and_splits() {
        let mut data = b"one\ntwo\r\n".to_vec();
        data.extend(std::iter::repeat_n(b'x', MAX_LINE_BYTES + 10));
        data.extend_from_slice(b"\nafter\n\xff\xfe\nlast");
        let (tx, mut rx) = mpsc::channel(16);
        tokio::spawn(read_lines(
            tokio::io::BufReader::new(std::io::Cursor::new(data)),
            tx,
        ));
        let mut got = Vec::new();
        while let Some(l) = rx.recv().await {
            got.push(match l {
                Line::Text(t) => format!("T:{}", t.trim_end()),
                Line::TooLong => "LONG".into(),
                Line::NotUtf8 => "BAD".into(),
            });
        }
        assert_eq!(got, ["T:one", "T:two", "LONG", "T:after", "BAD", "T:last"]);
    }
}
