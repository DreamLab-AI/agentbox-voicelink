//! The voice daemon (owned by the engine worker).

/// Runs the daemon over the given ssh ControlMaster socket.
pub fn run(_control_path: &std::path::Path) -> anyhow::Result<()> {
    anyhow::bail!("daemon not built yet")
}
