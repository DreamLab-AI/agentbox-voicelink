//! Error type shared by the whole crate.

use std::io;

/// Everything that can go wrong talking to tmux.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Spawning tmux or talking to its pipes failed.
    #[error("tmux I/O error: {0}")]
    Io(#[from] io::Error),
    /// tmux ran the command and reported an error (the text of a `%error`
    /// block, or the stderr of a one-shot invocation).
    #[error("tmux: {0}")]
    Tmux(String),
    /// The control connection ended before the command was answered. The
    /// payload carries tmux's explanation when it gave one (for example
    /// `can't find session: foo`).
    #[error("control connection closed{}", .0.as_deref().map(|r| format!(": {r}")).unwrap_or_default())]
    Closed(Option<String>),
    /// The command line was refused before sending: it was empty or held a
    /// raw line break (an empty line detaches a control client, and a break
    /// would split the command in two). Build commands with
    /// [`quote`](crate::quote) and [`command_line`](crate::command_line).
    #[error("invalid command: {0}")]
    InvalidCommand(&'static str),
    /// A session name tmux would silently rewrite (it turns `.` and `:` into
    /// `_`), which would make later lookups by that name miss.
    #[error("invalid session name {0:?}: must be non-empty with no '.', ':' or control characters")]
    InvalidSessionName(String),
}
