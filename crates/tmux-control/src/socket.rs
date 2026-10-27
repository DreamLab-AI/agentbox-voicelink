//! Selecting which tmux server to talk to.

use std::ffi::OsString;
use std::path::PathBuf;

/// Which tmux server a call addresses.
///
/// Every API in this crate takes a `Socket`, so a test or tool can run on a
/// private server and never touch the user's default one.
///
/// ```
/// use tmux_control::Socket;
/// assert!(Socket::Default.args().is_empty());
/// assert_eq!(Socket::Name("work".into()).args(), ["-L", "work"]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Socket {
    /// tmux's default server (`$TMUX`, else the per-user default socket).
    #[default]
    Default,
    /// A named server, `tmux -L name`.
    Name(String),
    /// An explicit socket path, `tmux -S path`.
    Path(PathBuf),
}

impl Socket {
    /// The leading tmux arguments that select this server.
    pub fn args(&self) -> Vec<OsString> {
        match self {
            Socket::Default => Vec::new(),
            Socket::Name(n) => vec!["-L".into(), n.into()],
            Socket::Path(p) => vec!["-S".into(), p.into()],
        }
    }
}
