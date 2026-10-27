//! `voicelink`: local voice for agent CLIs running in a remote tmux.

mod audio;
mod concierge;
mod config;
mod connect;
mod daemon;
mod link;
mod paths;
mod setup;
mod status;
mod status_cmd;

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
    /// Run the voice daemon over an established ssh ControlMaster (started by `connect`).
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
    match Cli::parse().command.unwrap_or(Command::Connect) {
        Command::Setup => setup::run(),
        Command::Connect => connect::run(),
        Command::Daemon { control_path } => daemon::run(&control_path),
        Command::Status => status_cmd::run(),
    }
}
