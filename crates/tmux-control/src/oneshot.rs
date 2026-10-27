//! One-shot (non-control) tmux invocations and session helpers.

use std::ffi::OsStr;
use std::process::Output;

use tokio::process::Command;

use crate::error::Error;
use crate::socket::Socket;

async fn raw<S: AsRef<OsStr>>(socket: &Socket, args: &[S]) -> Result<Output, Error> {
    // `-u`: see `Client::attach`; output must not depend on the locale.
    let out = Command::new("tmux")
        .arg("-u")
        .args(socket.args())
        .args(args)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .output()
        .await?;
    Ok(out)
}

fn failure(out: &Output) -> Error {
    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_owned();
    Error::Tmux(if stderr.is_empty() {
        format!("tmux exited with {}", out.status)
    } else {
        stderr
    })
}

/// Runs `tmux -u [-L|-S] args...` once, outside control mode, and returns its
/// standard output.
///
/// Arguments are passed as an argv vector, never through a shell, so no
/// quoting is needed here (unlike [`Client::command`](crate::Client::command)).
///
/// # Errors
///
/// [`Error::Io`] if tmux cannot be spawned; [`Error::Tmux`] with its standard
/// error when it exits non-zero.
///
/// ```no_run
/// # async fn demo() -> Result<(), tmux_control::Error> {
/// use tmux_control::{run, Socket};
/// let names = run(&Socket::Name("work".into()), &["list-sessions", "-F", "#{session_name}"]).await?;
/// println!("{names}");
/// # Ok(()) }
/// ```
pub async fn run<S: AsRef<OsStr>>(socket: &Socket, args: &[S]) -> Result<String, Error> {
    let out = raw(socket, args).await?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(failure(&out))
    }
}

fn check_name(name: &str) -> Result<(), Error> {
    if name.is_empty() || name.contains(['.', ':']) || name.chars().any(char::is_control) {
        return Err(Error::InvalidSessionName(name.to_owned()));
    }
    Ok(())
}

/// Whether a session named exactly `name` exists (no prefix matching).
///
/// # Errors
///
/// [`Error::InvalidSessionName`] for an unusable name; [`Error::Io`] if tmux
/// cannot be spawned. A missing server counts as "no such session".
pub async fn has_session(socket: &Socket, name: &str) -> Result<bool, Error> {
    check_name(name)?;
    let out = raw(socket, &["has-session", "-t", &format!("={name}")]).await?;
    Ok(out.status.success())
}

/// Ensures a detached session `new_name` exists in the same session group as
/// `target_session` (`new-session -d -t =target -s new_name`), so both see the
/// same windows but keep independent current-window selections.
///
/// Idempotent and race-tolerant: if the session already exists, or another
/// process creates it between the check and the creation (tmux answers
/// `duplicate session`), that is success. Returns `true` when this call
/// created it.
///
/// # Errors
///
/// [`Error::InvalidSessionName`], [`Error::Io`], or [`Error::Tmux`] when
/// creation fails for any reason other than a duplicate (for example the
/// target session does not exist).
///
/// ```no_run
/// # async fn demo() -> Result<(), tmux_control::Error> {
/// use tmux_control::{ensure_grouped_session, Socket};
/// let created = ensure_grouped_session(&Socket::Default, "agentbox", "remote-laptop").await?;
/// println!("created: {created}");
/// # Ok(()) }
/// ```
pub async fn ensure_grouped_session(
    socket: &Socket,
    target_session: &str,
    new_name: &str,
) -> Result<bool, Error> {
    check_name(new_name)?;
    if has_session(socket, new_name).await? {
        return Ok(false);
    }
    // `new-session -t` treats an unknown target as a *new group name* and
    // happily creates an unrelated session, so require the target to exist.
    if !has_session(socket, target_session).await? {
        return Err(Error::Tmux(format!("can't find session: {target_session}")));
    }
    let target = format!("={target_session}");
    let out = raw(
        socket,
        &["new-session", "-d", "-t", target.as_str(), "-s", new_name],
    )
    .await?;
    if out.status.success() {
        return Ok(true);
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stderr.contains("duplicate session") {
        return Ok(false);
    }
    Err(failure(&out))
}
