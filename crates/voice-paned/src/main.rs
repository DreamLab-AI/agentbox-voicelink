//! `voice-paned`: the voicelink controller and hook recorder.
//!
//! * `voice-paned --stdio` follows a grouped tmux view session and speaks the
//!   `voicelink-proto` JSON-lines protocol on stdin/stdout.
//! * `voice-paned record --source claude|codex` is the agent-hook recorder: it
//!   appends one event to the spool, prints nothing and always exits 0.

mod controller;
mod input;
mod prompt;
mod recorder;
mod spool;
mod tmuxq;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tmux_control::Socket;

/// Voice controller for agent CLIs running in tmux.
#[derive(Debug, Parser)]
#[command(name = "voice-paned", version)]
struct Cli {
    /// Serve the controller protocol on stdin/stdout.
    #[arg(long)]
    stdio: bool,
    /// tmux server by socket name (`tmux -L`).
    #[arg(long, conflicts_with = "socket_path")]
    socket_name: Option<String>,
    /// tmux server by socket path (`tmux -S`).
    #[arg(long)]
    socket_path: Option<PathBuf>,
    /// Shared base session whose windows the view session mirrors.
    #[arg(long, default_value = "agentbox")]
    session: String,
    /// Event spool file (default: `$HOME/.local/state/voicelink/events.jsonl`).
    #[arg(long)]
    spool: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Hook recorder: append one agent event to the spool (silent, exit 0).
    Record(recorder::RecordArgs),
}

fn main() -> ExitCode {
    // The recorder runs inside agent hooks: no argument error, panic or
    // failure may reach the agent, so it is handled before anything can
    // exit non-zero.
    if std::env::args_os().nth(1).is_some_and(|a| a == "record") {
        if let Ok(Cli {
            command: Some(Command::Record(args)),
            ..
        }) = Cli::try_parse()
        {
            recorder::run(args);
        }
        return ExitCode::SUCCESS;
    }

    let cli = Cli::parse();
    if !cli.stdio {
        eprintln!("voice-paned: nothing to do (use --stdio, or the record subcommand); see --help");
        return ExitCode::from(2);
    }
    let Some(spool) = cli.spool.or_else(spool::default_path) else {
        eprintln!("voice-paned: $HOME is not set and no --spool was given");
        return ExitCode::from(2);
    };
    let socket = match (cli.socket_name, cli.socket_path) {
        (Some(name), _) => Socket::Name(name),
        (None, Some(path)) => Socket::Path(path),
        (None, None) => Socket::Default,
    };
    let config = controller::Config {
        socket,
        session: cli.session,
        spool,
    };

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("voice-paned: cannot start runtime: {e}");
            return ExitCode::from(1);
        }
    };
    let result = runtime.block_on(controller::run(
        config,
        tokio::io::BufReader::new(tokio::io::stdin()),
        tokio::io::stdout(),
    ));
    // The stdin reader may be parked in a blocking read; do not wait for it.
    runtime.shutdown_background();
    match result {
        Ok(0) => ExitCode::SUCCESS,
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
        Err(e) => {
            eprintln!("voice-paned: {e:#}");
            ExitCode::from(1)
        }
    }
}
