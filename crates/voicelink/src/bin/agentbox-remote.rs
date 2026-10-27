//! `agentbox-remote`: an alias for `voicelink connect`.
//!
//! Executes the `voicelink` binary that sits next to this one (falling back
//! to `PATH`), passing every argument through after the `connect` subcommand.

use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let sibling = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("voicelink")))
        .filter(|p| p.exists());
    let program = sibling.unwrap_or_else(|| "voicelink".into());
    let err = Command::new(&program)
        .arg("connect")
        .args(std::env::args_os().skip(1))
        .exec();
    eprintln!("agentbox-remote: cannot run {}: {err}", program.display());
    std::process::exit(127);
}
