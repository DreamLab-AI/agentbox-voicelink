//! Launcher: ssh ControlMaster with forwards, the voice daemon, and the tmux
//! attach, in that order (I2, I5, I6).
//!
//! The command builders are pure functions returning argument vectors and the
//! quoted remote script so they can be tested without running ssh.

use std::io::IsTerminal;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use sha2::{Digest, Sha256};

use crate::config::{Config, Connection};
use crate::setup;

/// Longest control-socket path we accept; macOS `sun_path` holds 104 bytes.
const MAX_SOCKET_PATH: usize = 100;

/// Single-quotes `s` for a POSIX shell.
pub fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Eight hex characters of SHA-256 over the destination and ssh options.
pub fn short_hash(conn: &Connection) -> String {
    let mut h = Sha256::new();
    h.update(conn.ssh_destination.as_bytes());
    for o in &conn.ssh_options {
        h.update([0]);
        h.update(o.as_bytes());
    }
    setup::net::hex(&h.finalize())[..8].to_string()
}

/// The ControlPath socket for `conn`, under `state_dir`, or under the system
/// temp directory when that would exceed the Unix socket path limit.
pub fn control_socket(state_dir: &Path, conn: &Connection) -> PathBuf {
    let name = format!("cm-{}.sock", short_hash(conn));
    let p = state_dir.join(&name);
    if p.as_os_str().len() <= MAX_SOCKET_PATH {
        p
    } else {
        std::env::temp_dir().join(format!("voicelink-{name}"))
    }
}

fn s(v: &str) -> String {
    v.to_string()
}

/// `ssh` arguments that start the background master with the configured forwards.
pub fn master_argv(conn: &Connection, sock: &Path) -> Vec<String> {
    let mut a = vec![
        s("-f"),
        s("-N"),
        s("-M"),
        s("-S"),
        sock.to_string_lossy().into_owned(),
    ];
    for o in [
        "ControlPersist=yes",
        "ExitOnForwardFailure=yes",
        "ServerAliveInterval=30",
        "ServerAliveCountMax=3",
    ] {
        a.push(s("-o"));
        a.push(s(o));
    }
    for f in &conn.forwards {
        a.push(s("-L"));
        a.push(format!("127.0.0.1:{}:{}", f.local_port, f.remote));
    }
    a.extend(conn.ssh_options.iter().cloned());
    a.push(conn.ssh_destination.clone());
    a
}

/// `ssh -S sock -O <op> dest`.
pub fn control_argv(conn: &Connection, sock: &Path, op: &str) -> Vec<String> {
    let mut a = vec![s("-S"), sock.to_string_lossy().into_owned(), s("-O"), s(op)];
    a.extend(conn.ssh_options.iter().cloned());
    a.push(conn.ssh_destination.clone());
    a
}

/// The script run by `bash -c` inside the container: agentbox's own autostart,
/// then the grouped view session, then attach.
///
/// `destroy-unattached` is set in the same tmux invocation as the attach: set
/// on a detached session it would destroy the session on the spot.
pub fn remote_script(conn: &Connection) -> String {
    let view = sh_quote(&format!("={}", conn.view_session));
    let view_colon = sh_quote(&format!("={}:", conn.view_session));
    let view_name = sh_quote(&conn.view_session);
    let base = sh_quote(&format!("={}", conn.session));
    format!(
        "bash /opt/agentbox/config/tmux-autostart.sh 2>/dev/null; \
         tmux has-session -t {view} 2>/dev/null || tmux new-session -d -t {base} -s {view_name} 2>/dev/null || tmux has-session -t {view} 2>/dev/null; \
         tmux -u attach -t {view} \\; set-option -t {view_colon} destroy-unattached on 2>/dev/null || exec fish"
    )
}

/// The `docker exec` words (unquoted) that run the attach script.
pub fn docker_words(conn: &Connection, term: &str, colorterm: &str) -> Vec<String> {
    vec![
        s("docker"),
        s("exec"),
        s("-it"),
        s("--user"),
        conn.container_user.clone(),
        s("-e"),
        format!("TERM={term}"),
        s("-e"),
        format!("COLORTERM={colorterm}"),
        s("-e"),
        s("LANG=C.UTF-8"),
        s("-e"),
        s("LC_ALL=C.UTF-8"),
        conn.container.clone(),
        s("bash"),
        s("-c"),
        remote_script(conn),
    ]
}

/// The single remote command string handed to ssh: every word quoted.
pub fn remote_command(words: &[String]) -> String {
    words
        .iter()
        .map(|w| sh_quote(w))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `ssh` arguments that attach the terminal through the master.
pub fn attach_argv(conn: &Connection, sock: &Path, term: &str, colorterm: &str) -> Vec<String> {
    let mut a = vec![s("-S"), sock.to_string_lossy().into_owned(), s("-t")];
    a.extend(conn.ssh_options.iter().cloned());
    a.push(conn.ssh_destination.clone());
    a.push(remote_command(&docker_words(conn, term, colorterm)));
    a
}

fn ssh(args: &[String]) -> Command {
    let mut c = Command::new("ssh");
    c.args(args);
    c
}

fn master_alive(conn: &Connection, sock: &Path) -> bool {
    sock.exists()
        && ssh(&control_argv(conn, sock, "check"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
}

fn ensure_master(conn: &Connection, sock: &Path) -> anyhow::Result<()> {
    if master_alive(conn, sock) {
        return Ok(());
    }
    let _ = std::fs::remove_file(sock);
    // Inherit the terminal: password or passphrase prompts happen here, before tmux owns it (I6).
    let status = ssh(&master_argv(conn, sock))
        .status()
        .context("running ssh (is OpenSSH installed?)")?;
    if !status.success() {
        bail!(
            "ssh could not open the control connection to {} ({status}); check `ssh {}` works, or run `voicelink setup`",
            conn.ssh_destination,
            conn.ssh_destination
        );
    }
    Ok(())
}

/// The background voice daemon, stopped (and the master closed) on drop.
struct Link<'a> {
    conn: &'a Connection,
    sock: PathBuf,
    daemon: Option<Child>,
}

impl Link<'_> {
    fn spawn_daemon(&mut self) -> anyhow::Result<()> {
        let log = crate::paths::log_file()?;
        std::fs::create_dir_all(log.parent().context("log path has no parent")?)?;
        let out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .with_context(|| format!("opening {}", log.display()))?;
        let err = out.try_clone()?;
        let exe = std::env::current_exe().context("locating the voicelink executable")?;
        let child = Command::new(exe)
            .arg("daemon")
            .arg("--control-path")
            .arg(&self.sock)
            .process_group(0)
            .stdin(Stdio::null())
            .stdout(out)
            .stderr(err)
            .spawn()
            .context("starting the voice daemon")?;
        self.daemon = Some(child);
        std::thread::sleep(Duration::from_millis(400));
        if let Some(child) = self.daemon.as_mut()
            && let Ok(Some(status)) = child.try_wait()
        {
            eprintln!(
                "voicelink: the voice daemon exited early ({status}); voice is off for this session. See {}",
                log.display()
            );
            self.daemon = None;
        }
        Ok(())
    }

    fn stop_daemon(&mut self) {
        let Some(mut child) = self.daemon.take() else {
            return;
        };
        let _ = kill(Pid::from_raw(child.id() as i32), Signal::SIGTERM);
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        let _ = child.wait();
    }

    fn close_master(&self) {
        let _ = ssh(&control_argv(self.conn, &self.sock, "exit"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

impl Drop for Link<'_> {
    fn drop(&mut self) {
        self.stop_daemon();
        self.close_master();
    }
}

fn load_or_setup() -> anyhow::Result<Option<Config>> {
    let path = crate::paths::config_file()?;
    if !path.exists() {
        setup::run(setup::SetupArgs::default())?;
        if !path.exists() {
            return Ok(None);
        }
    }
    setup::store::load(&path).map(Some)
}

/// Attaches the terminal and starts the daemon.
pub fn run() -> anyhow::Result<()> {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        bail!("`voicelink connect` needs an interactive terminal");
    }
    let Some(cfg) = load_or_setup()? else {
        eprintln!("voicelink: setup was not completed; nothing written.");
        return Ok(());
    };
    let conn = &cfg.connection;
    if conn.ssh_destination.starts_with('-') {
        bail!(
            "ssh destination {:?} looks like an option",
            conn.ssh_destination
        );
    }
    let state = crate::paths::state_dir()?;
    std::fs::create_dir_all(&state)?;
    let sock = control_socket(&state, conn);

    ensure_master(conn, &sock)?;
    let mut link = Link {
        conn,
        sock: sock.clone(),
        daemon: None,
    };
    link.spawn_daemon()?;

    let term = std::env::var("TERM").unwrap_or_else(|_| "xterm-256color".into());
    let colorterm = std::env::var("COLORTERM").unwrap_or_default();
    let status = ssh(&attach_argv(conn, &sock, &term, &colorterm))
        .status()
        .context("attaching through ssh")?;
    drop(link);
    if !status.success() {
        eprintln!("voicelink: the attach ended with {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Forward;

    fn conn() -> Connection {
        let mut c = Connection::agentbox("machinelearn@192.168.2.132", "laptop");
        c.forwards.push(Forward {
            name: "loom".into(),
            local_port: 18084,
            remote: "127.0.0.1:8084".into(),
        });
        c.ssh_options = vec!["-p".into(), "2222".into()];
        c
    }

    #[test]
    fn quoting() {
        assert_eq!(sh_quote("abc"), "'abc'");
        assert_eq!(sh_quote("a'b"), r"'a'\''b'");
        assert_eq!(sh_quote(""), "''");
    }

    #[test]
    fn master_command_line() {
        let a = master_argv(&conn(), Path::new("/s/cm-x.sock"));
        assert_eq!(&a[..5], ["-f", "-N", "-M", "-S", "/s/cm-x.sock"]);
        assert!(a.windows(2).any(|w| w == ["-o", "ControlPersist=yes"]));
        assert!(
            a.windows(2)
                .any(|w| w == ["-o", "ExitOnForwardFailure=yes"])
        );
        assert!(
            a.windows(2)
                .any(|w| w == ["-L", "127.0.0.1:18084:127.0.0.1:8084"])
        );
        assert_eq!(
            &a[a.len() - 3..],
            ["-p", "2222", "machinelearn@192.168.2.132"].map(String::from)
        );
        assert_eq!(
            control_argv(&conn(), Path::new("/s"), "check")[..4],
            ["-S", "/s", "-O", "check"]
        );
    }

    #[test]
    fn socket_path_is_stable_and_short() {
        let c = conn();
        assert_eq!(short_hash(&c).len(), 8);
        assert_eq!(short_hash(&c), short_hash(&c.clone()));
        let mut other = c.clone();
        other.ssh_destination = "x@y".into();
        assert_ne!(short_hash(&c), short_hash(&other));
        let mac = Path::new("/Users/someone.long/Library/Application Support/voicelink/state");
        let p = control_socket(mac, &c);
        assert!(p.as_os_str().len() < 104, "{}", p.display());
        let deep = PathBuf::from(format!("/{}", "d".repeat(120)));
        let p = control_socket(&deep, &c);
        assert!(p.as_os_str().len() < 104 && p.starts_with(std::env::temp_dir()));
    }

    #[test]
    fn attach_matches_agentbox_shell() {
        let c = conn();
        let a = attach_argv(&c, Path::new("/s"), "xterm-256color", "truecolor");
        assert_eq!(&a[..3], ["-S", "/s", "-t"]);
        let cmd = a.last().unwrap();
        assert!(cmd.starts_with("'docker' 'exec' '-it' '--user' '1000' '-e' 'TERM=xterm-256color' '-e' 'COLORTERM=truecolor' '-e' 'LANG=C.UTF-8' '-e' 'LC_ALL=C.UTF-8' 'agentbox' 'bash' '-c' "), "{cmd}");
        let script = remote_script(&c);
        assert!(script.starts_with("bash /opt/agentbox/config/tmux-autostart.sh 2>/dev/null; "));
        assert!(script.contains("tmux has-session -t '=remote-laptop'"));
        assert!(script.contains("tmux new-session -d -t '=agentbox' -s 'remote-laptop'"));
        assert!(script.contains("tmux -u attach -t '=remote-laptop' \\; set-option -t '=remote-laptop:' destroy-unattached on"));
        assert!(script.ends_with("|| exec fish"));
    }

    /// Runs `command` through a local POSIX shell with `docker` replaced by a
    /// function that prints its arguments NUL-separated.
    fn argv_after_shell(command: &str) -> Vec<String> {
        let out = Command::new("sh")
            .arg("-c")
            .arg(format!("docker() {{ printf '%s\\0' \"$@\"; }}; {command}"))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout)
            .unwrap()
            .split('\0')
            .filter(|p| !p.is_empty())
            .map(String::from)
            .collect()
    }

    #[test]
    fn hostile_names_survive_both_shells() {
        let mut c = conn();
        c.view_session = "a'b $x `id` \"q\" ;rm".into();
        c.session = "s p'ace".into();
        c.container = "ctr;touch /tmp/pwn".into();
        c.container_user = "u$(id)".into();
        let words = docker_words(&c, "xterm'x", "$COLORTERM");
        let got = argv_after_shell(&remote_command(&words));
        assert_eq!(got, words[1..]);
        // The script, as bash -c receives it, still contains the names verbatim inside quotes.
        let script = got.last().unwrap();
        assert!(script.contains(r"'=a'\''b $x `id` "), "{script}");
    }

    #[test]
    fn view_session_prelude_works_on_a_private_tmux_server() {
        if Command::new("tmux").arg("-V").output().is_err() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let run = |args: &[&str]| {
            Command::new("tmux")
                .args(args)
                .env("TMUX_TMPDIR", dir.path())
                .env_remove("TMUX")
                .output()
                .unwrap()
        };
        assert!(
            run(&["new-session", "-d", "-s", "agentbox"])
                .status
                .success()
        );
        let mut c = conn();
        c.view_session = "remote-test".into();
        // Everything up to the attach: autostart (absent here, silenced), has-or-create.
        let script = remote_script(&c);
        let prelude: String = script.split(" tmux -u attach").next().unwrap().to_string();
        for _ in 0..2 {
            let st = Command::new("sh")
                .arg("-c")
                .arg(&prelude)
                .env("TMUX_TMPDIR", dir.path())
                .env_remove("TMUX")
                .status()
                .unwrap();
            assert!(st.success());
        }
        let ls =
            String::from_utf8(run(&["list-sessions", "-F", "#{session_name}"]).stdout).unwrap();
        assert!(ls.lines().any(|l| l == "remote-test"), "{ls}");
        assert_eq!(ls.lines().count(), 2);
        run(&["kill-server"]);
    }
}
