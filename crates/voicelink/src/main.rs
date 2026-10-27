//! `voicelink`: local voice for agent CLIs running in a remote tmux.
//!
//! The binary is a thin shell over the `voicelink_client` library crate target,
//! which exists so integration tests can drive the real link and handshake code.

use voicelink_client::{connect, daemon, setup, status_cmd};

use clap::{Parser, Subcommand};

/// Local voice for agent CLIs running in a remote tmux.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Run the onboarding wizard (also runs automatically when no config exists).
    Setup {
        #[command(flatten)]
        args: setup::SetupArgs,
    },
    /// Attach this terminal to agentbox and start the voice link (the default).
    Connect {
        #[command(flatten)]
        args: connect::ConnectArgs,
    },
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
    match Cli::parse().command.unwrap_or(Command::Connect {
        args: Default::default(),
    }) {
        Command::Setup { args } => setup::run(args),
        Command::Connect { args } => connect::run(args),
        Command::Daemon { control_path } => daemon::run(&control_path),
        Command::Status => status_cmd::run(),
    }
}
