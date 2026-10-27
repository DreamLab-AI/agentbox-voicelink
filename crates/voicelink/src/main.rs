//! `voicelink`: local voice for agent CLIs running in a remote tmux.

mod config;
mod status;

use clap::{Parser, Subcommand};

/// Local voice for agent CLIs running in a remote tmux.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the onboarding wizard (also runs automatically when no config exists).
    Setup,
    /// Attach this terminal to agentbox and start the voice link (the default).
    Connect,
    /// Run the voice daemon for an established ControlMaster (started by `connect`).
    #[command(hide = true)]
    Daemon {
        /// ssh ControlPath socket to multiplex over.
        #[arg(long)]
        control_path: std::path::PathBuf,
    },
    /// Show the daemon's state.
    Status,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let _ = cli.command;
    Ok(())
}
