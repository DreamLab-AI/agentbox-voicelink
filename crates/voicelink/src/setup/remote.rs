//! Reachability checks for the agentbox host: the `voice-paned --version`
//! probe and a real Hello/Welcome handshake over ssh.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use voicelink_proto::{
    ClientMessage, ErrorCode, PROTOCOL_VERSION, ServerMessage, decode_line, encode_line,
};

use crate::config::Connection;
use crate::connect::remote_command;

fn docker_exec_words(conn: &Connection, tail: &[&str]) -> Vec<String> {
    let mut w: Vec<String> = [
        "docker",
        "exec",
        "-i",
        "--user",
        &conn.container_user,
        &conn.container,
        &conn.paned_command,
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    w.extend(tail.iter().map(|x| x.to_string()));
    w
}

fn batch_argv(conn: &Connection, words: &[String]) -> Vec<String> {
    let mut a: Vec<String> = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
        .iter()
        .map(|x| x.to_string())
        .collect();
    a.extend(conn.ssh_options.iter().cloned());
    a.push(conn.ssh_destination.clone());
    a.push(remote_command(words));
    a
}

/// `ssh` arguments for the `voice-paned --version` probe.
pub fn version_argv(conn: &Connection) -> Vec<String> {
    batch_argv(conn, &docker_exec_words(conn, &["--version"]))
}

/// `ssh` arguments for the controller's stdio session.
pub fn stdio_argv(conn: &Connection) -> Vec<String> {
    batch_argv(
        conn,
        &docker_exec_words(conn, &["--stdio", "--session", &conn.session]),
    )
}

/// Result of the version probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The controller answered with its version line.
    Ready(String),
    /// Not usable, with an explanation for the operator.
    Problem(String),
}

/// Interprets the exit status and output of the version probe.
pub fn classify_version(code: Option<i32>, stdout: &str, stderr: &str) -> Verdict {
    let err = stderr.trim();
    let low = err.to_ascii_lowercase();
    if code == Some(0) && stdout.contains("voice-paned") {
        return Verdict::Ready(stdout.trim().to_string());
    }
    let first = err.lines().next().unwrap_or("");
    Verdict::Problem(
        if code == Some(255)
            || low.contains("permission denied (publickey")
            || low.contains("could not resolve")
            || low.contains("connection refused")
            || low.contains("timed out")
        {
            format!(
                "ssh could not connect without a prompt ({first}). Load your key into ssh-agent and check that `ssh <destination>` works; \
             over a tailnet use the tailnet name."
            )
        } else if low.contains("no such container") || low.contains("is not running") {
            format!("the container is not running or is named differently ({first}).")
        } else if code == Some(127)
            || low.contains("executable file not found")
            || low.contains("no such file")
            || low.contains("not found")
            || code == Some(126)
        {
            "voice-paned is not installed in the container yet. It arrives with the agentbox `[voicelink]` slot: enable it in agentbox.toml and \
         rebuild, or run the controller manually, then re-run this check."
            .to_string()
        } else if code == Some(0) {
            format!(
                "the command succeeded but did not look like voice-paned (output: {:?}).",
                stdout.trim()
            )
        } else {
            format!("unexpected failure (exit {code:?}): {first}")
        },
    )
}

/// Runs the version probe.
pub fn check_version(conn: &Connection) -> Verdict {
    match Command::new("ssh")
        .args(version_argv(conn))
        .stdin(Stdio::null())
        .output()
    {
        Ok(o) => classify_version(
            o.status.code(),
            &String::from_utf8_lossy(&o.stdout),
            &String::from_utf8_lossy(&o.stderr),
        ),
        Err(e) => Verdict::Problem(format!("could not run ssh: {e}")),
    }
}

/// What the controller said in its Welcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Welcome {
    /// Controller identifier.
    pub server: String,
    /// Focus at handshake, `window:name command`.
    pub focus: String,
}

/// Turns the controller's first line into a result.
pub fn interpret_first_line(line: &str) -> Result<Welcome, String> {
    match decode_line::<ServerMessage>(line) {
        Ok(ServerMessage::Welcome {
            proto,
            server,
            focus,
            ..
        }) if proto == PROTOCOL_VERSION => Ok(Welcome {
            server,
            focus: format!("{}:{} {}", focus.window, focus.window_name, focus.command),
        }),
        Ok(ServerMessage::Welcome { proto, server, .. }) => Err(format!(
            "protocol mismatch: {server} speaks version {proto}, this client speaks {PROTOCOL_VERSION}. Update voicelink or agentbox so they match."
        )),
        Ok(ServerMessage::Error {
            code: ErrorCode::ProtocolMismatch,
            message,
            ..
        }) => Err(format!(
            "protocol mismatch reported by the controller: {message}. This client speaks version {PROTOCOL_VERSION}; update voicelink or agentbox."
        )),
        Ok(ServerMessage::Error { code, message, .. }) => Err(format!(
            "the controller refused the session ({code:?}): {message}"
        )),
        Ok(other) => Err(format!(
            "unexpected first message from the controller: {other:?}"
        )),
        Err(e) => Err(format!(
            "the remote end did not speak the voicelink protocol ({e}); first line: {:?}",
            line.chars().take(120).collect::<String>()
        )),
    }
}

/// Performs Hello/Welcome over ssh, then says Bye.
pub fn handshake(conn: &Connection) -> Result<Welcome, String> {
    let mut ssh = Command::new("ssh");
    ssh.args(stdio_argv(conn));
    handshake_with(ssh, &conn.view_session)
}

/// Performs Hello/Welcome with the controller started by `command` (whose
/// stdin and stdout become the session), then says Bye. Production passes
/// `ssh`; tests pass the `voice-paned` binary itself.
pub fn handshake_with(mut command: Command, view_session: &str) -> Result<Welcome, String> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run ssh: {e}"))?;
    let mut stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let hello = ClientMessage::Hello {
        proto: PROTOCOL_VERSION,
        client: format!("voicelink/{}", env!("CARGO_PKG_VERSION")),
        view_session: view_session.to_string(),
    };
    let line = encode_line(&hello).map_err(|e| e.to_string())?;
    stdin
        .write_all(line.as_bytes())
        .and_then(|()| stdin.flush())
        .map_err(|e| format!("could not write to ssh: {e}"))?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut l = String::new();
        let _ = BufReader::new(stdout).read_line(&mut l);
        let _ = tx.send(l);
    });
    let first = rx.recv_timeout(Duration::from_secs(15));
    if let Ok(bye) = encode_line(&ClientMessage::Bye) {
        let _ = stdin.write_all(bye.as_bytes());
    }
    drop(stdin);
    let result = match first {
        Ok(l) if !l.trim().is_empty() => interpret_first_line(&l),
        _ => {
            let _ = child.kill();
            let mut err = String::new();
            if let Some(mut e) = child.stderr.take() {
                let _ = std::io::Read::read_to_string(&mut e, &mut err);
            }
            match classify_version(None, "", &err) {
                Verdict::Problem(p) if !err.trim().is_empty() => Err(p),
                _ => Err("no answer from the controller within 15 s".into()),
            }
        }
    };
    let _ = child.kill();
    let _ = child.wait();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conn() -> Connection {
        Connection::agentbox("machinelearn@192.168.2.132", "laptop")
    }

    #[test]
    fn version_probe_command() {
        let a = version_argv(&conn());
        assert_eq!(
            &a[..5],
            [
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "machinelearn@192.168.2.132"
            ]
        );
        assert_eq!(
            a[5],
            "'docker' 'exec' '-i' '--user' '1000' 'agentbox' '/opt/agentbox/bin/voice-paned' '--version'"
        );
        let s = stdio_argv(&conn());
        assert!(
            s[5].ends_with("'--stdio' '--session' 'agentbox'"),
            "{}",
            s[5]
        );
    }

    #[test]
    fn verdicts() {
        assert_eq!(
            classify_version(Some(0), "voice-paned 0.1.0\n", ""),
            Verdict::Ready("voice-paned 0.1.0".into())
        );
        let not_baked = classify_version(
            Some(127),
            "",
            "OCI runtime exec failed: exec failed: unable to start container process: exec: \"/opt/agentbox/bin/voice-paned\": stat /opt/agentbox/bin/voice-paned: no such file or directory",
        );
        assert!(matches!(&not_baked, Verdict::Problem(p) if p.contains("[voicelink]")));
        assert!(
            matches!(classify_version(Some(255), "", "ssh: connect to host x port 22: Connection refused"), Verdict::Problem(p) if p.contains("ssh could not connect"))
        );
        assert!(
            matches!(classify_version(Some(1), "", "Error response from daemon: No such container: agentbox"), Verdict::Problem(p) if p.contains("container"))
        );
        assert!(matches!(
            classify_version(Some(0), "hello", ""),
            Verdict::Problem(_)
        ));
    }

    #[test]
    fn handshake_lines() {
        let welcome = r#"{"type":"welcome","proto":1,"server":"voice-paned/0.1.0","view_session":"remote-x","focus":{"window":2,"window_name":"Services","pane":"%7","command":"codex","agent":"codex"}}"#;
        let w = interpret_first_line(welcome).unwrap();
        assert_eq!(
            w,
            Welcome {
                server: "voice-paned/0.1.0".into(),
                focus: "2:Services codex".into()
            }
        );
        let old = welcome.replace("\"proto\":1", "\"proto\":9");
        assert!(
            interpret_first_line(&old)
                .unwrap_err()
                .contains("protocol mismatch")
        );
        let err = r#"{"type":"error","id":null,"code":"protocol_mismatch","message":"want 2"}"#;
        assert!(
            interpret_first_line(err)
                .unwrap_err()
                .contains("protocol mismatch")
        );
        assert!(
            interpret_first_line("bash: voice-paned: command not found")
                .unwrap_err()
                .contains("did not speak")
        );
    }
}
