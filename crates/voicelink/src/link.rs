//! The controller connection: one JSON-lines session to `voice-paned`.
//!
//! The transport is abstracted behind [`Connector`] (any `AsyncRead` +
//! `AsyncWrite` pair), so production uses an `ssh` child multiplexed over the
//! operator's ControlMaster (I2: no other ingress) and tests use an in-memory
//! duplex. A supervisor task owns the session: handshake, request/reply
//! correlation, pushes (`Focus`, `Event`), and reconnection with backoff.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};
use voicelink_proto::{
    AgentEvent, ClientMessage, ErrorCode, Focus, Key, MAX_LINE_BYTES, PROTOCOL_VERSION, PaneId,
    PaneInfo, RequestId, ServerMessage, decode_line, encode_line,
};

use crate::config::Connection;

/// Failure of a single request.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LinkError {
    /// No live session (lost, or not yet established).
    #[error("not connected to agentbox")]
    Disconnected,
    /// The controller did not answer in time.
    #[error("the controller did not answer in time")]
    Timeout,
    /// The controller answered with an error.
    #[error("controller error ({code:?}): {message}")]
    Controller {
        /// Machine-readable code.
        code: ErrorCode,
        /// Human-readable detail.
        message: String,
    },
    /// The controller sent a reply of an unexpected shape.
    #[error("unexpected controller reply: {0}")]
    Unexpected(String),
}

/// Something the link tells the engine about, outside any request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkNotice {
    /// A session was established (also carries the initial focus).
    Connected(Focus),
    /// The focused pane changed.
    Focus(Focus),
    /// An agent event from the spool.
    Event(AgentEvent),
    /// The session was lost; the link will retry.
    Disconnected(String),
    /// A condition retrying cannot fix (for example a protocol mismatch).
    Fatal(String),
}

/// A connected byte stream pair plus whatever keeps it alive (the ssh child).
pub struct Transport {
    /// Bytes from the controller.
    pub reader: Box<dyn AsyncRead + Send + Unpin>,
    /// Bytes to the controller.
    pub writer: Box<dyn AsyncWrite + Send + Unpin>,
    /// Dropped when the session ends.
    pub keepalive: Option<Box<dyn Send>>,
}

/// Opens a fresh transport; called again on every reconnect.
#[async_trait]
pub trait Connector: Send + Sync + 'static {
    /// Establishes one connection attempt.
    async fn connect(&self) -> io::Result<Transport>;
}

/// Timing and identity knobs.
#[derive(Debug, Clone)]
pub struct LinkConfig {
    /// `client` string sent in `Hello`.
    pub client: String,
    /// The grouped view session this client drives.
    pub view_session: String,
    /// Per-request reply deadline.
    pub request_timeout: Duration,
    /// Deadline for connect plus handshake.
    pub hello_timeout: Duration,
    /// First reconnect delay.
    pub backoff_min: Duration,
    /// Reconnect delay ceiling.
    pub backoff_max: Duration,
}

impl LinkConfig {
    /// Production timings for `view_session`.
    pub fn production(view_session: &str) -> Self {
        LinkConfig {
            client: format!("voicelink/{}", env!("CARGO_PKG_VERSION")),
            view_session: view_session.to_string(),
            request_timeout: Duration::from_secs(10),
            hello_timeout: Duration::from_secs(20),
            backoff_min: Duration::from_millis(500),
            backoff_max: Duration::from_secs(15),
        }
    }
}

struct Cmd {
    id: RequestId,
    msg: ClientMessage,
    reply: oneshot::Sender<Result<ServerMessage, LinkError>>,
}

/// Cheap, cloneable handle for issuing requests.
///
/// None of the writing calls takes a pane argument (I1): they act on the pane
/// the controller reports as focused.
#[derive(Clone)]
pub struct LinkHandle {
    tx: mpsc::UnboundedSender<Cmd>,
    next_id: Arc<AtomicU64>,
    timeout: Duration,
}

impl LinkHandle {
    async fn request(
        &self,
        build: impl FnOnce(RequestId) -> ClientMessage,
    ) -> Result<ServerMessage, LinkError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Cmd {
                id,
                msg: build(id),
                reply,
            })
            .map_err(|_| LinkError::Disconnected)?;
        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(r)) => r,
            Ok(Err(_)) => Err(LinkError::Disconnected),
            Err(_) => Err(LinkError::Timeout),
        }
    }

    async fn expect_ack(
        &self,
        build: impl FnOnce(RequestId) -> ClientMessage,
    ) -> Result<(), LinkError> {
        match self.request(build).await? {
            ServerMessage::Ack { .. } => Ok(()),
            other => Err(LinkError::Unexpected(format!("{other:?}"))),
        }
    }

    /// Lists every pane in the shared session.
    pub async fn list_panes(&self) -> Result<Vec<PaneInfo>, LinkError> {
        match self.request(|id| ClientMessage::ListPanes { id }).await? {
            ServerMessage::Panes { panes, .. } => Ok(panes),
            other => Err(LinkError::Unexpected(format!("{other:?}"))),
        }
    }

    /// Reads the last `lines` lines of any pane (reads may address any pane).
    pub async fn read_pane(&self, pane: &str, lines: u32) -> Result<String, LinkError> {
        let pane = pane.to_string();
        match self
            .request(|id| ClientMessage::ReadPane { id, pane, lines })
            .await?
        {
            ServerMessage::PaneText { text, .. } => Ok(text),
            other => Err(LinkError::Unexpected(format!("{other:?}"))),
        }
    }

    /// Types `text` into the focused pane.
    pub async fn send(&self, text: &str, submit: bool, force: bool) -> Result<(), LinkError> {
        let text = text.to_string();
        self.expect_ack(|id| ClientMessage::Send {
            id,
            text,
            submit,
            force,
        })
        .await
    }

    /// Presses keys in the focused pane.
    pub async fn keys(&self, keys: Vec<Key>) -> Result<(), LinkError> {
        self.expect_ack(|id| ClientMessage::Keys { id, keys }).await
    }

    /// Interrupts the focused pane.
    pub async fn interrupt(&self) -> Result<(), LinkError> {
        self.expect_ack(|id| ClientMessage::Interrupt { id }).await
    }

    /// Moves focus to `pane`; the operator sees the tab switch.
    pub async fn select_pane(&self, pane: &str) -> Result<(), LinkError> {
        let pane: PaneId = pane.to_string();
        self.expect_ack(|id| ClientMessage::SelectPane { id, pane })
            .await
    }

    /// Round-trip liveness check.
    pub async fn ping(&self) -> Result<(), LinkError> {
        match self.request(|id| ClientMessage::Ping { id }).await? {
            ServerMessage::Pong { .. } => Ok(()),
            other => Err(LinkError::Unexpected(format!("{other:?}"))),
        }
    }
}

/// Starts the supervisor. Returns the request handle, the notice stream and the
/// supervisor task (dropping every handle ends the session with `Bye`).
pub fn spawn(
    connector: impl Connector,
    cfg: LinkConfig,
) -> (
    LinkHandle,
    mpsc::UnboundedReceiver<LinkNotice>,
    tokio::task::JoinHandle<()>,
) {
    let (tx, cmd_rx) = mpsc::unbounded_channel();
    let (notice_tx, notice_rx) = mpsc::unbounded_channel();
    let handle = LinkHandle {
        tx,
        next_id: Arc::new(AtomicU64::new(1)),
        timeout: cfg.request_timeout,
    };
    let task = tokio::spawn(supervise(Box::new(connector), cfg, cmd_rx, notice_tx));
    (handle, notice_rx, task)
}

fn reject(cmd: Cmd) {
    let _ = cmd.reply.send(Err(LinkError::Disconnected));
}

/// Awaits `fut` while answering any request that arrives with `Disconnected`.
async fn rejecting<T>(
    cmd_rx: &mut mpsc::UnboundedReceiver<Cmd>,
    fut: impl std::future::Future<Output = T>,
) -> T {
    tokio::pin!(fut);
    loop {
        tokio::select! {
            out = &mut fut => return out,
            Some(cmd) = cmd_rx.recv() => reject(cmd),
        }
    }
}

enum Attempt {
    Fatal(String),
    Retry(String),
}

async fn supervise(
    connector: Box<dyn Connector>,
    cfg: LinkConfig,
    mut cmd_rx: mpsc::UnboundedReceiver<Cmd>,
    notices: mpsc::UnboundedSender<LinkNotice>,
) {
    let mut backoff = cfg.backoff_min;
    loop {
        let attempt = rejecting(
            &mut cmd_rx,
            tokio::time::timeout(cfg.hello_timeout, open(&*connector, &cfg)),
        )
        .await;
        let attempt = match attempt {
            Ok(r) => r,
            Err(_) => Err(Attempt::Retry("connection attempt timed out".into())),
        };
        match attempt {
            Ok((session, focus)) => {
                backoff = cfg.backoff_min;
                let _ = notices.send(LinkNotice::Connected(focus));
                let (reason, closing) = run_session(session, &mut cmd_rx, &notices).await;
                if closing {
                    return;
                }
                tracing::warn!("controller link lost: {reason}");
                let _ = notices.send(LinkNotice::Disconnected(reason));
            }
            Err(Attempt::Fatal(reason)) => {
                tracing::error!("controller link fatal: {reason}");
                let _ = notices.send(LinkNotice::Fatal(reason));
                // Keep answering requests so callers fail fast instead of hanging.
                while let Some(cmd) = cmd_rx.recv().await {
                    reject(cmd);
                }
                return;
            }
            Err(Attempt::Retry(reason)) => {
                tracing::warn!("controller connect failed: {reason}");
                let _ = notices.send(LinkNotice::Disconnected(reason));
            }
        }
        if cmd_rx.is_closed() && cmd_rx.is_empty() {
            return;
        }
        rejecting(&mut cmd_rx, tokio::time::sleep(backoff)).await;
        backoff = (backoff * 2).min(cfg.backoff_max);
    }
}

struct Session {
    reader: mpsc::UnboundedReceiver<ReadEvent>,
    writer: Box<dyn AsyncWrite + Send + Unpin>,
    reader_task: tokio::task::JoinHandle<()>,
    _keepalive: Option<Box<dyn Send>>,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.reader_task.abort();
    }
}

enum ReadEvent {
    Message(ServerMessage),
    Closed(String),
}

/// Reads one `\n`-terminated line, refusing lines over [`MAX_LINE_BYTES`].
/// `Ok(false)` means clean EOF.
async fn read_line_limited<R: AsyncBufRead + Unpin>(
    r: &mut R,
    buf: &mut Vec<u8>,
) -> io::Result<bool> {
    buf.clear();
    loop {
        let avail = r.fill_buf().await?;
        if avail.is_empty() {
            return Ok(!buf.is_empty());
        }
        if let Some(i) = avail.iter().position(|b| *b == b'\n') {
            buf.extend_from_slice(&avail[..i]);
            r.consume(i + 1);
            return Ok(true);
        }
        let n = avail.len();
        buf.extend_from_slice(avail);
        r.consume(n);
        if buf.len() > MAX_LINE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "controller line too long",
            ));
        }
    }
}

async fn read_loop(
    reader: Box<dyn AsyncRead + Send + Unpin>,
    tx: mpsc::UnboundedSender<ReadEvent>,
) {
    let mut reader = BufReader::new(reader);
    let mut buf = Vec::new();
    loop {
        match read_line_limited(&mut reader, &mut buf).await {
            Ok(true) => {
                let line = String::from_utf8_lossy(&buf);
                if line.trim().is_empty() {
                    continue;
                }
                match decode_line::<ServerMessage>(&line) {
                    Ok(ServerMessage::Unknown) => {
                        // A newer controller's message: not ours to interpret.
                        tracing::info!(
                            "ignoring a controller message of unknown type: {}",
                            line.chars().take(120).collect::<String>()
                        );
                    }
                    Ok(m) => {
                        if tx.send(ReadEvent::Message(m)).is_err() {
                            return;
                        }
                    }
                    Err(e) => tracing::warn!("undecodable controller line: {e}"),
                }
            }
            Ok(false) => {
                let _ = tx.send(ReadEvent::Closed("controller closed the connection".into()));
                return;
            }
            Err(e) => {
                let _ = tx.send(ReadEvent::Closed(format!("read error: {e}")));
                return;
            }
        }
    }
}

async fn write_msg(w: &mut (dyn AsyncWrite + Send + Unpin), msg: &ClientMessage) -> io::Result<()> {
    let line =
        encode_line(msg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    w.write_all(line.as_bytes()).await?;
    w.flush().await
}

async fn open(connector: &dyn Connector, cfg: &LinkConfig) -> Result<(Session, Focus), Attempt> {
    let t = connector
        .connect()
        .await
        .map_err(|e| Attempt::Retry(format!("connect: {e}")))?;
    let (tx, rx) = mpsc::unbounded_channel();
    let reader_task = tokio::spawn(read_loop(t.reader, tx));
    let mut session = Session {
        reader: rx,
        writer: t.writer,
        reader_task,
        _keepalive: t.keepalive,
    };
    let hello = ClientMessage::Hello {
        proto: PROTOCOL_VERSION,
        client: cfg.client.clone(),
        view_session: cfg.view_session.clone(),
    };
    write_msg(&mut *session.writer, &hello)
        .await
        .map_err(|e| Attempt::Retry(format!("hello: {e}")))?;
    loop {
        match session.reader.recv().await {
            Some(ReadEvent::Message(ServerMessage::Welcome {
                proto,
                server,
                focus,
                ..
            })) => {
                if proto != PROTOCOL_VERSION {
                    return Err(Attempt::Fatal(format!(
                        "protocol mismatch: client speaks {PROTOCOL_VERSION}, controller {server} speaks {proto}"
                    )));
                }
                return Ok((session, focus));
            }
            Some(ReadEvent::Message(ServerMessage::Error {
                code: ErrorCode::ProtocolMismatch,
                message,
                ..
            })) => {
                return Err(Attempt::Fatal(format!(
                    "protocol mismatch: client speaks {PROTOCOL_VERSION}; controller says: {message}. Update voicelink or agentbox so both speak the same protocol."
                )));
            }
            Some(ReadEvent::Message(ServerMessage::Error { code, message, .. })) => {
                return Err(Attempt::Retry(format!(
                    "handshake refused ({code:?}): {message}"
                )));
            }
            Some(ReadEvent::Message(_)) => {}
            Some(ReadEvent::Closed(why)) => {
                return Err(Attempt::Retry(format!("handshake: {why}")));
            }
            None => return Err(Attempt::Retry("handshake: reader ended".into())),
        }
    }
}

type Pending = HashMap<RequestId, oneshot::Sender<Result<ServerMessage, LinkError>>>;

/// Runs one session; returns why it ended and whether the whole link is closing.
async fn run_session(
    mut s: Session,
    cmd_rx: &mut mpsc::UnboundedReceiver<Cmd>,
    notices: &mpsc::UnboundedSender<LinkNotice>,
) -> (String, bool) {
    let mut pending: Pending = HashMap::new();
    let mut closing = false;
    let reason = loop {
        tokio::select! {
            ev = s.reader.recv() => match ev {
                Some(ReadEvent::Message(m)) => route(m, &mut pending, notices),
                Some(ReadEvent::Closed(why)) => break why,
                None => break "reader ended".to_string(),
            },
            cmd = cmd_rx.recv() => match cmd {
                Some(Cmd { id, msg, reply }) => {
                    pending.retain(|_, r| !r.is_closed());
                    match write_msg(&mut *s.writer, &msg).await {
                        Ok(()) => { pending.insert(id, reply); }
                        Err(e) => {
                            let _ = reply.send(Err(LinkError::Disconnected));
                            break format!("write error: {e}");
                        }
                    }
                }
                None => {
                    let _ = write_msg(&mut *s.writer, &ClientMessage::Bye).await;
                    closing = true;
                    break "shutting down".to_string();
                }
            },
        }
    };
    // Fail every in-flight request immediately instead of letting it time out.
    for (_, r) in pending.drain() {
        let _ = r.send(Err(LinkError::Disconnected));
    }
    (reason, closing)
}

fn route(m: ServerMessage, pending: &mut Pending, notices: &mpsc::UnboundedSender<LinkNotice>) {
    let mut reply = |id: RequestId, r: Result<ServerMessage, LinkError>| {
        if let Some(tx) = pending.remove(&id) {
            let _ = tx.send(r);
        }
    };
    match m {
        ServerMessage::Focus(f) => {
            let _ = notices.send(LinkNotice::Focus(f));
        }
        ServerMessage::Event(e) => {
            let _ = notices.send(LinkNotice::Event(e));
        }
        ServerMessage::Welcome { .. } => {}
        ServerMessage::Error {
            id: Some(id),
            code,
            message,
        } => reply(id, Err(LinkError::Controller { code, message })),
        ServerMessage::Error {
            id: None,
            code,
            message,
        } => tracing::warn!("controller error ({code:?}): {message}"),
        ServerMessage::Panes { id, .. }
        | ServerMessage::PaneText { id, .. }
        | ServerMessage::Ack { id }
        | ServerMessage::Pong { id } => reply(id, Ok(m)),
        // A newer controller's message (the read loop already drops Unknown).
        _ => tracing::info!("ignoring an unrecognised controller message"),
    }
}

/// Connects through `ssh` over the operator's ControlMaster socket (I2).
pub struct SshConnector {
    control_path: std::path::PathBuf,
    conn: Connection,
}

impl SshConnector {
    /// Builds a connector for `conn`, multiplexed over `control_path`.
    pub fn new(control_path: &Path, conn: &Connection) -> Self {
        SshConnector {
            control_path: control_path.to_path_buf(),
            conn: conn.clone(),
        }
    }
}

/// POSIX single-quote escaping, so the remote shell that ssh hands the joined
/// command line to cannot split or expand a value.
fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:=@%+,".contains(c))
    {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// The ssh argument vector (without the program name).
pub(crate) fn ssh_args(control_path: &Path, c: &Connection) -> Vec<String> {
    let mut args = vec![
        "-S".to_string(),
        control_path.display().to_string(),
        "-o".into(),
        "ControlMaster=no".into(),
        // Never prompt: a password prompt would be written to the operator's tty (I5).
        "-o".into(),
        "BatchMode=yes".into(),
        "-T".into(),
    ];
    args.extend(c.ssh_options.iter().cloned());
    args.push(c.ssh_destination.clone());
    let mut remote: Vec<String> = ["docker", "exec", "-i", "--user"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    remote.push(c.container_user.clone());
    remote.push(c.container.clone());
    remote.extend(c.paned_command.split_whitespace().map(str::to_string));
    remote.push("--stdio".into());
    remote.push("--session".into());
    remote.push(c.session.clone());
    args.push(
        remote
            .iter()
            .map(|a| shell_quote(a))
            .collect::<Vec<_>>()
            .join(" "),
    );
    args
}

#[async_trait]
impl Connector for SshConnector {
    async fn connect(&self) -> io::Result<Transport> {
        let mut child = tokio::process::Command::new("ssh")
            .args(ssh_args(&self.control_path, &self.conn))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("no ssh stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("no ssh stdout"))?;
        Ok(Transport {
            reader: Box::new(stdout),
            writer: Box::new(stdin),
            keepalive: Some(Box::new(child)),
        })
    }
}

#[cfg(test)]
pub(crate) mod fake {
    //! In-process fake controller for tests.

    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;

    use super::*;

    /// Shared state of the fake controller.
    #[derive(Default)]
    pub(crate) struct Script {
        /// Every client message received, across sessions.
        pub log: Vec<ClientMessage>,
        /// `ReadPane` answers by pane.
        pub pane_text: HashMap<String, String>,
        /// `ListPanes` answer.
        pub panes: Vec<PaneInfo>,
        /// Reply `InputNotEmpty` to non-forced `Send`s.
        pub input_not_empty: bool,
        /// Protocol number answered in `Welcome`.
        pub welcome_proto: Option<u32>,
        /// Send `ProtocolMismatch` instead of `Welcome`.
        pub mismatch: bool,
        /// Close the current session right after the next message.
        pub drop_next: bool,
        /// Reply `PromptActive` to every `Send`, forced or not.
        pub prompt_active: bool,
    }

    /// Handle to a fake controller plus the connector that reaches it.
    #[derive(Clone)]
    pub(crate) struct FakeController {
        pub script: Arc<Mutex<Script>>,
        push: Arc<Mutex<Option<mpsc::UnboundedSender<String>>>>,
        pub connects: Arc<AtomicU64>,
        pub refuse: Arc<AtomicBool>,
        pub initial_focus: Focus,
    }

    pub(crate) fn focus(window: u32, name: &str, pane: &str, command: &str) -> Focus {
        Focus {
            window,
            window_name: name.into(),
            pane: pane.into(),
            command: command.into(),
            agent: voicelink_proto::AgentKind::from_command(command),
        }
    }

    impl FakeController {
        pub(crate) fn new(initial_focus: Focus) -> Self {
            FakeController {
                script: Arc::default(),
                push: Arc::default(),
                connects: Arc::default(),
                refuse: Arc::default(),
                initial_focus,
            }
        }

        /// Messages the client has sent so far.
        pub(crate) fn log(&self) -> Vec<ClientMessage> {
            self.script.lock().unwrap().log.clone()
        }

        /// Pushes a raw line (possibly from a newer controller) to the live session.
        pub(crate) fn push_raw(&self, line: &str) {
            if let Some(tx) = self.push.lock().unwrap().as_ref() {
                let _ = tx.send(format!("{line}\n"));
            }
        }

        /// Pushes an unsolicited message to the live session.
        pub(crate) fn push(&self, m: ServerMessage) {
            if let Some(tx) = self.push.lock().unwrap().as_ref() {
                let _ = tx.send(encode_line(&m).unwrap());
            }
        }

        async fn serve(self, server: tokio::io::DuplexStream) {
            let (r, mut w) = tokio::io::split(server);
            let mut lines = BufReader::new(r).lines();
            let (ptx, mut prx) = mpsc::unbounded_channel();
            *self.push.lock().unwrap() = Some(ptx);
            loop {
                tokio::select! {
                    line = lines.next_line() => {
                        let Ok(Some(line)) = line else { return };
                        let Ok(msg) = decode_line::<ClientMessage>(&line) else { continue };
                        let replies = self.respond(&msg);
                        let drop_now = std::mem::take(&mut self.script.lock().unwrap().drop_next);
                        if drop_now { return; }
                        for r in replies {
                            if w.write_all(encode_line(&r).unwrap().as_bytes()).await.is_err() { return; }
                        }
                    }
                    pushed = prx.recv() => {
                        let Some(m) = pushed else { return };
                        if w.write_all(m.as_bytes()).await.is_err() { return; }
                    }
                }
            }
        }

        fn respond(&self, msg: &ClientMessage) -> Vec<ServerMessage> {
            let mut s = self.script.lock().unwrap();
            s.log.push(msg.clone());
            match msg {
                ClientMessage::Hello { .. } => {
                    if s.mismatch {
                        return vec![ServerMessage::Error {
                            id: None,
                            code: ErrorCode::ProtocolMismatch,
                            message: "controller speaks protocol 99".into(),
                        }];
                    }
                    vec![ServerMessage::Welcome {
                        proto: s.welcome_proto.unwrap_or(PROTOCOL_VERSION),
                        server: "fake-paned/0".into(),
                        view_session: "remote-test".into(),
                        focus: self.initial_focus.clone(),
                    }]
                }
                ClientMessage::ListPanes { id } => vec![ServerMessage::Panes {
                    id: *id,
                    panes: s.panes.clone(),
                }],
                ClientMessage::ReadPane { id, pane, .. } => vec![ServerMessage::PaneText {
                    id: *id,
                    pane: pane.clone(),
                    text: s.pane_text.get(pane).cloned().unwrap_or_default(),
                }],
                ClientMessage::Send { id, .. } if s.prompt_active => {
                    vec![ServerMessage::Error {
                        id: Some(*id),
                        code: ErrorCode::PromptActive,
                        message: "a menu is showing".into(),
                    }]
                }
                ClientMessage::Send { id, force, .. } if s.input_not_empty && !force => {
                    vec![ServerMessage::Error {
                        id: Some(*id),
                        code: ErrorCode::InputNotEmpty,
                        message: "the prompt already holds text".into(),
                    }]
                }
                ClientMessage::Send { id, .. }
                | ClientMessage::Keys { id, .. }
                | ClientMessage::Interrupt { id }
                | ClientMessage::SelectPane { id, .. } => vec![ServerMessage::Ack { id: *id }],
                ClientMessage::Ping { id } => vec![ServerMessage::Pong { id: *id }],
                _ => vec![],
            }
        }
    }

    #[async_trait]
    impl Connector for FakeController {
        async fn connect(&self) -> io::Result<Transport> {
            if self.refuse.load(Ordering::Relaxed) {
                return Err(io::Error::new(io::ErrorKind::ConnectionRefused, "refused"));
            }
            self.connects.fetch_add(1, Ordering::Relaxed);
            let (client, server) = tokio::io::duplex(64 * 1024);
            tokio::spawn(self.clone().serve(server));
            let (r, w) = tokio::io::split(client);
            Ok(Transport {
                reader: Box::new(r),
                writer: Box::new(w),
                keepalive: None,
            })
        }
    }

    /// Quick timings for tests.
    pub(crate) fn fast_config() -> LinkConfig {
        LinkConfig {
            client: "voicelink/test".into(),
            view_session: "remote-test".into(),
            request_timeout: Duration::from_secs(2),
            hello_timeout: Duration::from_secs(2),
            backoff_min: Duration::from_millis(20),
            backoff_max: Duration::from_millis(80),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::*;
    use super::*;
    use voicelink_proto::AgentKind;

    async fn next_notice(rx: &mut mpsc::UnboundedReceiver<LinkNotice>) -> LinkNotice {
        tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("notice in time")
            .expect("open")
    }

    #[tokio::test]
    async fn handshake_then_requests_correlate_by_id() {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        fc.script
            .lock()
            .unwrap()
            .pane_text
            .insert("%3".into(), "hello pane".into());
        let (h, mut notices, _task) = spawn(fc.clone(), fast_config());
        assert!(
            matches!(next_notice(&mut notices).await, LinkNotice::Connected(f) if f.agent == AgentKind::Claude)
        );
        let (a, b, c) = tokio::join!(h.read_pane("%3", 20), h.ping(), h.send("hi", true, false));
        assert_eq!(a.unwrap(), "hello pane");
        b.unwrap();
        c.unwrap();
        let log = fc.log();
        assert!(
            matches!(&log[0], ClientMessage::Hello { proto, client, view_session }
            if *proto == PROTOCOL_VERSION && client.starts_with("voicelink/") && view_session == "remote-test")
        );
    }

    #[tokio::test]
    async fn unknown_server_messages_and_fields_are_ignored() {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        let (h, mut notices, _task) = spawn(fc.clone(), fast_config());
        next_notice(&mut notices).await;
        fc.push_raw(r#"{"type":"hologram","id":1,"payload":[1,2,3]}"#);
        fc.push_raw(
            r#"{"type":"focus","window":4,"window_name":"x","pane":"%4","command":"codex","agent":"codex","zoom":true}"#,
        );
        fc.push_raw(
            r#"{"type":"event","ts_ms":1,"pane":"%4","window":4,"agent":"aider","kind":"cost_alert","text":null}"#,
        );
        assert!(matches!(next_notice(&mut notices).await, LinkNotice::Focus(f) if f.window == 4));
        assert!(matches!(
            next_notice(&mut notices).await,
            LinkNotice::Event(e) if e.kind == voicelink_proto::EventKind::Unknown
        ));
        // The session survived all of it.
        h.ping().await.unwrap();
        assert_eq!(fc.connects.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn pushes_are_routed_as_notices() {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        let (_h, mut notices, _task) = spawn(fc.clone(), fast_config());
        next_notice(&mut notices).await;
        fc.push(ServerMessage::Focus(focus(2, "svc", "%2", "codex")));
        assert!(matches!(next_notice(&mut notices).await, LinkNotice::Focus(f) if f.window == 2));
    }

    #[tokio::test]
    async fn controller_error_is_reported_with_its_code() {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        fc.script.lock().unwrap().input_not_empty = true;
        let (h, mut notices, _task) = spawn(fc, fast_config());
        next_notice(&mut notices).await;
        let err = h.send("x", true, false).await.unwrap_err();
        assert!(matches!(
            err,
            LinkError::Controller {
                code: ErrorCode::InputNotEmpty,
                ..
            }
        ));
        h.send("x", true, true).await.unwrap();
    }

    #[tokio::test]
    async fn protocol_mismatch_is_fatal_and_does_not_hot_loop() {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        fc.script.lock().unwrap().mismatch = true;
        let (h, mut notices, _task) = spawn(fc.clone(), fast_config());
        let LinkNotice::Fatal(why) = next_notice(&mut notices).await else {
            panic!("expected fatal")
        };
        assert!(why.contains("protocol mismatch"), "{why}");
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(fc.connects.load(Ordering::Relaxed), 1);
        assert_eq!(h.ping().await.unwrap_err(), LinkError::Disconnected);
    }

    #[tokio::test]
    async fn welcome_with_other_protocol_is_fatal() {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        fc.script.lock().unwrap().welcome_proto = Some(PROTOCOL_VERSION + 1);
        let (_h, mut notices, _task) = spawn(fc, fast_config());
        assert!(matches!(
            next_notice(&mut notices).await,
            LinkNotice::Fatal(_)
        ));
    }

    #[tokio::test]
    async fn loss_fails_pending_immediately_then_reconnects_with_new_session() {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        let (h, mut notices, _task) = spawn(fc.clone(), fast_config());
        next_notice(&mut notices).await;
        h.ping().await.unwrap();
        fc.script.lock().unwrap().drop_next = true;
        let t0 = std::time::Instant::now();
        assert_eq!(h.ping().await.unwrap_err(), LinkError::Disconnected);
        assert!(
            t0.elapsed() < Duration::from_secs(1),
            "pending request waited for its timeout"
        );
        assert!(matches!(
            next_notice(&mut notices).await,
            LinkNotice::Disconnected(_)
        ));
        assert!(matches!(
            next_notice(&mut notices).await,
            LinkNotice::Connected(_)
        ));
        h.ping().await.unwrap();
        assert_eq!(fc.connects.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn requests_while_down_fail_fast_and_connect_retries_back_off() {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        fc.refuse.store(true, Ordering::Relaxed);
        let (h, mut notices, _task) = spawn(fc.clone(), fast_config());
        assert!(matches!(
            next_notice(&mut notices).await,
            LinkNotice::Disconnected(_)
        ));
        assert_eq!(h.ping().await.unwrap_err(), LinkError::Disconnected);
        fc.refuse.store(false, Ordering::Relaxed);
        loop {
            if matches!(next_notice(&mut notices).await, LinkNotice::Connected(_)) {
                break;
            }
        }
        h.ping().await.unwrap();
    }

    #[tokio::test]
    async fn an_unanswered_request_times_out() {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        let mut cfg = fast_config();
        cfg.request_timeout = Duration::from_millis(100);
        let (h, mut notices, _task) = spawn(fc.clone(), cfg);
        next_notice(&mut notices).await;
        // The fake answers with the id it was sent, so a reply to a wrong id is never delivered.
        let err = h
            .request(|id| ClientMessage::Ping { id: id + 1_000_000 })
            .await;
        assert_eq!(err.unwrap_err(), LinkError::Timeout);
    }

    #[tokio::test]
    async fn over_long_line_is_refused() {
        let (mut a, b) = tokio::io::duplex(4 * MAX_LINE_BYTES);
        tokio::spawn(async move {
            let junk = vec![b'x'; MAX_LINE_BYTES + 10];
            let _ = a.write_all(&junk).await;
        });
        let mut r = BufReader::new(b);
        let mut buf = Vec::new();
        assert!(read_line_limited(&mut r, &mut buf).await.is_err());
    }

    #[test]
    fn ssh_command_line_matches_the_spec_and_quotes_arguments() {
        let mut c = Connection::agentbox("john@10.0.0.5", "laptop");
        c.ssh_options = vec!["-p".into(), "2222".into()];
        c.session = "my session".into();
        let args = ssh_args(Path::new("/run/x/ctl.sock"), &c);
        assert_eq!(
            &args[..7],
            [
                "-S",
                "/run/x/ctl.sock",
                "-o",
                "ControlMaster=no",
                "-o",
                "BatchMode=yes",
                "-T"
            ]
        );
        assert_eq!(&args[7..10], ["-p", "2222", "john@10.0.0.5"]);
        assert_eq!(
            args[10],
            "docker exec -i --user 1000 agentbox /opt/agentbox/bin/voice-paned --stdio --session 'my session'"
        );
    }

    #[test]
    fn shell_quote_neutralises_metacharacters() {
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
        assert_eq!(shell_quote("$(touch pwned)"), "'$(touch pwned)'");
        assert_eq!(shell_quote("plain-1.2"), "plain-1.2");
    }
}
