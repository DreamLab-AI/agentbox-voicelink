//! Shared test harness: a private tmux server and a controller process.
//!
//! SAFETY: every tmux invocation here carries `-S <private socket>`, removes
//! `TMUX`/`TMUX_PANE` from its environment and points `TMUX_TMPDIR` at the
//! private directory, so even a missing `-S` could not reach the default server.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use voicelink_proto::{ClientMessage, PROTOCOL_VERSION, ServerMessage, decode_line, encode_line};

pub const TIMEOUT: Duration = Duration::from_secs(10);

/// A private tmux server, killed on drop.
pub struct Server {
    pub dir: tempfile::TempDir,
    pub sock: PathBuf,
}

impl Server {
    /// Base session `agentbox` with windows: 0 `one` (one pane), 1 `two` (two
    /// panes), 2 `three` (one pane), all running `cat`. Window 0 is current.
    /// `None` (with a message) when tmux is not installed.
    pub fn start() -> Option<Server> {
        if StdCommand::new("tmux")
            .arg("-V")
            .stdout(Stdio::null())
            .status()
            .is_err()
        {
            eprintln!("SKIP: tmux not found on PATH");
            return None;
        }
        let dir = tempfile::Builder::new()
            .prefix("vp-")
            .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
            .unwrap();
        let sock = dir.path().join("sock");
        let s = Server { dir, sock };
        s.ok(&[
            "-f",
            "/dev/null",
            "new-session",
            "-d",
            "-x",
            "100",
            "-y",
            "30",
            "-s",
            "agentbox",
            "-n",
            "one",
            "cat",
        ]);
        s.ok(&[
            "new-window",
            "-d",
            "-t",
            "agentbox:",
            "-n",
            "two words",
            "cat",
        ]);
        s.ok(&["split-window", "-d", "-t", "agentbox:1", "cat"]);
        s.ok(&["new-window", "-d", "-t", "agentbox:", "-n", "three", "cat"]);
        assert_eq!(
            s.out(&["display-message", "-p", "-t", "agentbox", "#{window_index}"]),
            "0"
        );
        Some(s)
    }

    pub fn cmd(&self) -> StdCommand {
        let mut c = StdCommand::new("tmux");
        c.arg("-S")
            .arg(&self.sock)
            .env_remove("TMUX")
            .env_remove("TMUX_PANE")
            .env("TMUX_TMPDIR", self.dir.path());
        c
    }

    pub fn ok(&self, args: &[&str]) {
        let o = self.cmd().args(args).output().unwrap();
        assert!(
            o.status.success(),
            "tmux {args:?}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
    }

    pub fn out(&self, args: &[&str]) -> String {
        let o = self.cmd().args(args).output().unwrap();
        String::from_utf8_lossy(&o.stdout).trim_end().to_owned()
    }

    pub fn try_out(&self, args: &[&str]) -> Option<String> {
        let o = self.cmd().args(args).output().unwrap();
        o.status
            .success()
            .then(|| String::from_utf8_lossy(&o.stdout).trim_end().to_owned())
    }

    pub fn sessions(&self) -> Vec<String> {
        self.out(&["list-sessions", "-F", "#{session_name}"])
            .lines()
            .map(str::to_owned)
            .collect()
    }

    /// Pane ids of a window of the base session, in order.
    pub fn panes_of(&self, window: u32) -> Vec<String> {
        let target = format!("agentbox:{window}");
        self.out(&["list-panes", "-t", &target, "-F", "#{pane_id}"])
            .lines()
            .map(str::to_owned)
            .collect()
    }

    pub fn base_window(&self) -> String {
        self.out(&["display-message", "-p", "-t", "agentbox", "#{window_index}"])
    }

    pub fn view_window(&self, view: &str) -> String {
        self.out(&[
            "display-message",
            "-p",
            "-t",
            &format!("={view}:"),
            "#{window_index}",
        ])
    }

    pub fn capture(&self, pane: &str) -> String {
        self.out(&["capture-pane", "-p", "-t", pane])
    }

    pub fn pane_sizes(&self) -> Vec<String> {
        self.out(&[
            "list-panes",
            "-a",
            "-F",
            "#{pane_id}=#{pane_width}x#{pane_height}",
        ])
        .lines()
        .map(str::to_owned)
        .collect()
    }

    pub fn tmux_env(&self) -> String {
        let pid = self.out(&["display-message", "-p", "#{pid}"]);
        format!("{},{pid},0", self.sock.display())
    }

    /// Polls `capture-pane` until `pred` holds.
    pub fn wait_capture(&self, pane: &str, what: &str, pred: impl Fn(&str) -> bool) -> String {
        let start = Instant::now();
        loop {
            let c = self.capture(pane);
            if pred(&c) {
                return c;
            }
            assert!(
                start.elapsed() < TIMEOUT,
                "timed out waiting for {what} in {pane}; screen:\n{c}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    pub fn spool_path(&self) -> PathBuf {
        self.dir.path().join("spool/events.jsonl")
    }

    /// Starts the controller binary against this server.
    pub fn controller(&self) -> Controller {
        self.controller_with(&["--session", "agentbox"])
    }

    pub fn controller_with(&self, extra: &[&str]) -> Controller {
        let mut child = Command::new(env!("CARGO_BIN_EXE_voice-paned"))
            .arg("--stdio")
            .arg("--socket-path")
            .arg(&self.sock)
            .arg("--spool")
            .arg(self.spool_path())
            .args(extra)
            .env_remove("TMUX")
            .env_remove("TMUX_PANE")
            .env("TMUX_TMPDIR", self.dir.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Controller {
            child,
            stdin,
            stdout,
        }
    }

    /// Runs the recorder binary as an agent hook would.
    pub fn record(
        &self,
        args: &[&str],
        stdin: Option<&str>,
        pane: Option<&str>,
    ) -> std::process::Output {
        use std::io::Write;
        let mut cmd = StdCommand::new(env!("CARGO_BIN_EXE_voice-paned"));
        cmd.arg("record")
            .args(args)
            .env_remove("TMUX")
            .env_remove("TMUX_PANE")
            .env("TMUX_TMPDIR", self.dir.path());
        if let Some(p) = pane {
            cmd.env("TMUX", self.tmux_env()).env("TMUX_PANE", p);
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut sin = child.stdin.take().unwrap();
        if let Some(s) = stdin {
            let _ = sin.write_all(s.as_bytes());
        }
        drop(sin);
        child.wait_with_output().unwrap()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.cmd().arg("kill-server").stderr(Stdio::null()).status();
    }
}

/// A running `voice-paned --stdio`.
pub struct Controller {
    pub child: Child,
    pub stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl Controller {
    pub async fn send(&mut self, msg: &ClientMessage) {
        let line = encode_line(msg).unwrap();
        self.send_raw(&line).await;
    }

    pub async fn send_raw(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin closed");
        stdin.write_all(line.as_bytes()).await.unwrap();
        stdin.flush().await.unwrap();
    }

    /// Next server message, or `None` at end of stream.
    pub async fn try_recv(&mut self) -> Option<ServerMessage> {
        let mut line = String::new();
        let n = tokio::time::timeout(TIMEOUT, self.stdout.read_line(&mut line))
            .await
            .expect("timed out reading controller");
        (n.unwrap() > 0)
            .then(|| decode_line(&line).unwrap_or_else(|e| panic!("bad server line {line:?}: {e}")))
    }

    pub async fn recv(&mut self) -> ServerMessage {
        self.try_recv().await.expect("controller closed stdout")
    }

    /// Reads until `pick` returns `Some`, ignoring other messages.
    pub async fn recv_until<T>(&mut self, mut pick: impl FnMut(&ServerMessage) -> Option<T>) -> T {
        loop {
            let m = self.recv().await;
            if let Some(v) = pick(&m) {
                return v;
            }
        }
    }

    /// Hello + Welcome; returns the Welcome focus.
    pub async fn handshake(&mut self, view: &str) -> voicelink_proto::Focus {
        self.send(&ClientMessage::Hello {
            proto: PROTOCOL_VERSION,
            client: "test".into(),
            view_session: view.into(),
        })
        .await;
        match self.recv().await {
            ServerMessage::Welcome {
                focus,
                view_session,
                ..
            } => {
                assert_eq!(view_session, view);
                focus
            }
            other => panic!("expected welcome, got {other:?}"),
        }
    }

    /// Sends a request and returns the reply bearing the same id, skipping pushes.
    pub async fn request(&mut self, msg: ClientMessage, id: u64) -> ServerMessage {
        self.send(&msg).await;
        self.recv_until(|m| match m {
            ServerMessage::Focus(_) | ServerMessage::Event(_) => None,
            ServerMessage::Error { id: Some(i), .. }
            | ServerMessage::Ack { id: i }
            | ServerMessage::Pong { id: i }
            | ServerMessage::Panes { id: i, .. }
            | ServerMessage::PaneText { id: i, .. }
                if *i == id =>
            {
                Some(m.clone())
            }
            _ => None,
        })
        .await
    }

    /// Closes stdin and waits for the process to exit.
    pub async fn close_and_wait(mut self) -> std::process::ExitStatus {
        self.stdin.take();
        tokio::time::timeout(TIMEOUT, self.child.wait())
            .await
            .expect("controller did not exit")
            .unwrap()
    }
}

/// Spawns a terminal-like tmux client on a pty (via `script`) of the given
/// size, attached to `target`. `None` when `script` is unavailable.
pub struct PtyClient {
    handle: Option<std::process::Child>,
}

impl PtyClient {
    pub fn attach(server: &Server, target: &str, cols: u32, rows: u32) -> Option<PtyClient> {
        if StdCommand::new("script")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            eprintln!("SKIP: `script` not found; cannot create a pty client");
            return None;
        }
        let inner = format!(
            "stty cols {cols} rows {rows}; exec tmux -S {} attach -t {target}",
            shell_quote(&server.sock)
        );
        let mut child = StdCommand::new("script");
        child
            .args(["-qec", &inner, "/dev/null"])
            .env_remove("TMUX")
            .env_remove("TMUX_PANE")
            .env("TMUX_TMPDIR", server.dir.path())
            .env("TERM", "xterm-256color")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let handle = child.spawn().ok()?;
        let me = PtyClient {
            handle: Some(handle),
        };
        // Wait for the client to register.
        let start = Instant::now();
        while server.terminal_clients().is_empty() {
            assert!(start.elapsed() < TIMEOUT, "pty client never attached");
            std::thread::sleep(Duration::from_millis(50));
        }
        Some(me)
    }
}

impl Drop for PtyClient {
    fn drop(&mut self) {
        if let Some(mut h) = self.handle.take() {
            drop(h.stdin.take());
            let _ = h.kill();
            let _ = h.wait();
        }
    }
}

fn shell_quote(p: &Path) -> String {
    format!("'{}'", p.display().to_string().replace('\'', "'\\''"))
}

impl Server {
    /// Names of attached clients that are not control-mode clients.
    pub fn terminal_clients(&self) -> Vec<String> {
        self.out(&["list-clients", "-F", "#{client_name}\t#{client_flags}"])
            .lines()
            .filter_map(|l| {
                let (name, flags) = l.split_once('\t')?;
                (!flags.contains("control-mode")).then(|| name.to_owned())
            })
            .collect()
    }

    pub fn control_clients(&self) -> usize {
        self.out(&["list-clients", "-F", "#{client_flags}"])
            .lines()
            .filter(|f| f.contains("control-mode"))
            .count()
    }
}
