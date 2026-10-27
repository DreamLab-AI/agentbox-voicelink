//! The asynchronous control-mode connection.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, Command};
use tokio::sync::{mpsc, oneshot};

use crate::error::Error;
use crate::parser::{Decoder, Event, Notification};
use crate::socket::Socket;

/// Receiving end of a connection's notifications. It yields `None` once the
/// connection has ended (after [`Notification::Exit`], if tmux sent one).
pub type Notifications = mpsc::UnboundedReceiver<Notification>;

type Reply = Result<Vec<String>, Error>;

/// How to attach a control-mode client.
///
/// ```
/// use tmux_control::ClientOptions;
/// let opts = ClientOptions::new("work").flag("ignore-size").flag("no-output");
/// assert_eq!(opts.session, "work");
/// assert_eq!(opts.flags, ["ignore-size", "no-output"]);
/// ```
#[derive(Debug, Clone)]
pub struct ClientOptions {
    /// Session to attach to (matched exactly, not by prefix).
    pub session: String,
    /// Client flags passed as `-f a,b,c`: for example `ignore-size`,
    /// `no-output`, `read-only`, `pause-after=5`.
    pub flags: Vec<String>,
    /// The tmux executable. Defaults to `tmux` on `PATH`.
    pub program: OsString,
}

impl ClientOptions {
    /// Options attaching to `session` with no client flags.
    pub fn new(session: impl Into<String>) -> Self {
        Self {
            session: session.into(),
            flags: Vec::new(),
            program: "tmux".into(),
        }
    }

    /// Adds one client flag.
    #[must_use]
    pub fn flag(mut self, flag: impl Into<String>) -> Self {
        self.flags.push(flag.into());
        self
    }

    /// Uses a specific tmux executable.
    #[must_use]
    pub fn program(mut self, program: impl Into<OsString>) -> Self {
        self.program = program.into();
        self
    }
}

#[derive(Default)]
struct Shared {
    /// Senders awaiting a reply, in command order. `None` once the connection
    /// has closed.
    pending: Mutex<Option<VecDeque<oneshot::Sender<Reply>>>>,
    /// Why the connection ended, as far as tmux said.
    reason: Mutex<Option<String>>,
}

impl Shared {
    fn closed_error(&self) -> Error {
        Error::Closed(self.reason.lock().ok().and_then(|r| r.clone()))
    }
}

enum Msg {
    Command {
        line: String,
        reply: oneshot::Sender<Reply>,
    },
    Detach {
        done: oneshot::Sender<()>,
    },
}

/// A live control-mode connection to one tmux session.
///
/// Created with [`Client::attach`]. Commands are answered in order; the
/// connection is cancellation-safe (dropping a [`command`](Client::command)
/// future never corrupts the stream). Dropping the `Client` closes the
/// connection; prefer [`detach`](Client::detach) for a clean goodbye.
#[derive(Debug)]
pub struct Client {
    tx: mpsc::UnboundedSender<Msg>,
    shared: Arc<Shared>,
    exited: Option<oneshot::Receiver<()>>,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Shared")
    }
}

impl Client {
    /// Spawns `tmux -u [-L|-S] -C attach -t =session [-f flags]` and starts
    /// the reader and writer tasks.
    ///
    /// `-u` makes tmux treat the client as UTF-8 whatever the locale: without
    /// it, a client spawned with no `LANG`/`LC_*` (as under a bare
    /// `docker exec`) gets tabs and other control characters in command
    /// output replaced with `_`, which breaks tab-separated formats.
    ///
    /// Returns immediately; if the session does not exist, the first
    /// [`command`](Client::command) fails with [`Error::Closed`] carrying
    /// tmux's message, and the notification channel ends.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if tmux cannot be spawned.
    pub fn attach(
        socket: &Socket,
        options: &ClientOptions,
    ) -> Result<(Client, Notifications), Error> {
        let mut cmd = Command::new(&options.program);
        cmd.arg("-u")
            .args(socket.args())
            .arg("-C")
            .arg("attach")
            .arg("-t")
            .arg(format!("={}", options.session));
        if !options.flags.is_empty() {
            cmd.arg("-f").arg(options.flags.join(","));
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Tmux("no stdin pipe".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Tmux("no stdout pipe".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| Error::Tmux("no stderr pipe".into()))?;

        let shared = Arc::new(Shared {
            pending: Mutex::new(Some(VecDeque::new())),
            reason: Mutex::new(None),
        });
        let (ntx, nrx) = mpsc::unbounded_channel();
        let (tx, rx) = mpsc::unbounded_channel();
        let (exit_tx, exit_rx) = oneshot::channel();

        tokio::spawn(reader(stdout, shared.clone(), ntx));
        tokio::spawn(stderr_reader(stderr, shared.clone()));
        tokio::spawn(writer(stdin, rx, shared.clone()));
        tokio::spawn(async move {
            let _ = child.wait().await;
            let _ = exit_tx.send(());
        });

        Ok((
            Client {
                tx,
                shared,
                exited: Some(exit_rx),
            },
            nrx,
        ))
    }

    /// Sends one tmux command line and returns its output lines.
    ///
    /// Build the line with [`command_line`](crate::command_line) when any
    /// part is untrusted.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidCommand`] for an empty line or one containing `\n` or
    ///   `\r`;
    /// * [`Error::Tmux`] with tmux's message when the command fails;
    /// * [`Error::Closed`] when the connection ends first.
    pub async fn command(&self, line: impl Into<String>) -> Result<Vec<String>, Error> {
        let line = line.into();
        if line.trim().is_empty() {
            return Err(Error::InvalidCommand("empty command"));
        }
        if line.contains(['\n', '\r']) {
            return Err(Error::InvalidCommand("raw line break in command"));
        }
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Msg::Command { line, reply })
            .map_err(|_| self.shared.closed_error())?;
        match rx.await {
            Ok(r) => r,
            Err(_) => Err(self.shared.closed_error()),
        }
    }

    /// Cleanly detaches: tells tmux to end the client, then waits (up to two
    /// seconds) for the process to exit. The tmux server and the session are
    /// untouched.
    pub async fn detach(mut self) {
        let (done, done_rx) = oneshot::channel();
        if self.tx.send(Msg::Detach { done }).is_ok() {
            let _ = done_rx.await;
        }
        if let Some(exited) = self.exited.take() {
            let _ = tokio::time::timeout(Duration::from_secs(2), exited).await;
        }
    }
}

async fn writer(mut stdin: ChildStdin, mut rx: mpsc::UnboundedReceiver<Msg>, shared: Arc<Shared>) {
    while let Some(msg) = rx.recv().await {
        match msg {
            Msg::Command { line, reply } => {
                // Register before writing so the reply cannot overtake us.
                {
                    let Ok(mut guard) = shared.pending.lock() else {
                        return;
                    };
                    match guard.as_mut() {
                        Some(q) => q.push_back(reply),
                        None => {
                            let _ = reply.send(Err(shared.closed_error()));
                            continue;
                        }
                    }
                }
                let mut bytes = line.into_bytes();
                bytes.push(b'\n');
                let res = async {
                    stdin.write_all(&bytes).await?;
                    stdin.flush().await
                }
                .await;
                if let Err(e) = res
                    && let Ok(mut guard) = shared.pending.lock()
                    && let Some(sender) = guard.as_mut().and_then(VecDeque::pop_back)
                {
                    let _ = sender.send(Err(Error::Io(e)));
                }
            }
            Msg::Detach { done } => {
                // An empty line makes tmux end a control client.
                let _ = stdin.write_all(b"\n").await;
                let _ = stdin.flush().await;
                drop(stdin);
                let _ = done.send(());
                return;
            }
        }
    }
    // All senders gone: dropping stdin (EOF) makes tmux exit the client.
}

async fn stderr_reader(stderr: tokio::process::ChildStderr, shared: Arc<Shared>) {
    let mut lines = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if let Ok(mut r) = shared.reason.lock() {
            r.get_or_insert(line);
        }
    }
}

async fn reader(
    stdout: tokio::process::ChildStdout,
    shared: Arc<Shared>,
    ntx: mpsc::UnboundedSender<Notification>,
) {
    let mut input = BufReader::new(stdout);
    let mut decoder = Decoder::default();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match input.read_until(b'\n', &mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if buf.last() == Some(&b'\n') {
            buf.pop();
        }
        match decoder.feed(&buf) {
            None => {}
            Some(Event::Notification(n)) => {
                if let Notification::Exit { reason: Some(r) } = &n
                    && let Ok(mut slot) = shared.reason.lock()
                {
                    slot.get_or_insert_with(|| r.clone());
                }
                let _ = ntx.send(n);
            }
            Some(Event::Reply(reply)) => {
                if !reply.is_ours() {
                    // The block answering our own `attach` (or any other
                    // server-originated block). An error here means we never
                    // attached; remember why.
                    if let Err(lines) = &reply.result
                        && let Ok(mut slot) = shared.reason.lock()
                    {
                        *slot = Some(lines.join("; "));
                    }
                    continue;
                }
                let sender = shared
                    .pending
                    .lock()
                    .ok()
                    .and_then(|mut g| g.as_mut().and_then(VecDeque::pop_front));
                if let Some(sender) = sender {
                    let out = reply.result.map_err(|lines| Error::Tmux(lines.join("\n")));
                    let _ = sender.send(out);
                }
            }
        }
    }
    // End of stream: fail everything still waiting, then close the channel.
    let drained = shared.pending.lock().ok().and_then(|mut g| g.take());
    if let Some(queue) = drained {
        for sender in queue {
            let _ = sender.send(Err(shared.closed_error()));
        }
    }
}
