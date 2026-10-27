//! The voice daemon: `voicelink daemon --control-path <sock>`.
//!
//! Started by `connect` once the ssh ControlMaster exists. It never writes to
//! the terminal it was started from (I5): logs go to a file, state goes to the
//! status file, and fatal errors are recorded there before the process exits.

mod engine;
mod hotkey;
mod speaker;
mod status_file;

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;

use crate::config::Config;
use crate::link::{LinkConfig, SshConnector};
use crate::status::{BackendNames, LinkState};
use crate::{audio, paths};
use status_file::StatusHandle;

/// Largest log kept before it is rotated aside at start-up.
const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

/// Sends tracing output to the daemon log file; silent if it cannot be opened.
fn init_logging() {
    // A panic message would otherwise go to stderr, i.e. the operator's terminal.
    std::panic::set_hook(Box::new(|info| tracing::error!("panic: {info}")));
    let Ok(path) = paths::log_file() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if std::fs::metadata(&path).is_ok_and(|m| m.len() > MAX_LOG_BYTES) {
        let _ = std::fs::rename(&path, path.with_extension("log.1"));
    }
    let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    let filter = tracing_subscriber::EnvFilter::try_from_env("VOICELINK_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_writer(std::sync::Mutex::new(file))
        .with_ansi(false)
        .with_env_filter(filter)
        .try_init();
}

fn load_config() -> anyhow::Result<Config> {
    let path = paths::config_file()?;
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Runs the daemon over the given ssh ControlMaster socket.
///
/// On a start-up failure the error is logged and written to the status file,
/// and the process exits with status 1 without printing anything.
pub fn run(control_path: &Path) -> anyhow::Result<()> {
    init_logging();
    let status_path = paths::status_file().ok();
    match run_inner(control_path, status_path.clone()) {
        Ok(()) => Ok(()),
        Err(e) => {
            tracing::error!("daemon failed: {e:#}");
            let status = StatusHandle::new(
                status_path,
                BackendNames {
                    stt: String::new(),
                    llm: String::new(),
                    tts: String::new(),
                },
            );
            status.update(|s| {
                s.state = LinkState::Stopped;
                s.last_error = Some(format!("{e:#}"));
            });
            // Returning the error would make `main` print it to stderr (I5).
            std::process::exit(1);
        }
    }
}

fn run_inner(control_path: &Path, status_path: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    let cfg = load_config()?;
    tracing::info!("voicelink daemon {} starting", env!("CARGO_PKG_VERSION"));

    let stt: Arc<dyn voicelink_backends::SpeechToText> =
        Arc::from(voicelink_backends::build_stt(&cfg.stt).context("speech-to-text backend")?);
    let llm: Arc<dyn voicelink_backends::ChatModel> =
        Arc::from(voicelink_backends::build_llm(&cfg.llm).context("chat backend")?);
    let tts: Arc<dyn voicelink_backends::TextToSpeech> =
        Arc::from(voicelink_backends::build_tts(&cfg.tts).context("text-to-speech backend")?);
    let status = StatusHandle::new(
        status_path,
        BackendNames {
            stt: stt.name(),
            llm: llm.name(),
            tts: tts.name(),
        },
    );

    let io = audio::open(&cfg.audio).context("opening audio devices")?;
    hotkey::spawn(&cfg.hotkey, Arc::clone(&io.muted), status.clone());

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;
    let result = rt.block_on(async {
        let (link, notices, link_task) = crate::link::spawn(
            SshConnector::new(control_path, &cfg.connection),
            LinkConfig::production(&cfg.connection.view_session),
        );
        let parts = engine::Parts {
            stt,
            llm,
            tts,
            audio_events: io.events,
            sink: io.sink,
            link,
            notices,
            audio: cfg.audio.clone(),
            concierge: cfg.concierge.clone(),
            status: status.clone(),
        };
        engine::run(parts, shutdown_signal()).await;
        // Dropping the last handle makes the session say Bye; give it a moment.
        if tokio::time::timeout(std::time::Duration::from_millis(500), link_task)
            .await
            .is_err()
        {
            tracing::debug!("link task did not finish in time");
        }
        Ok::<(), anyhow::Error>(())
    });
    status.update(|s| {
        s.state = LinkState::Stopped;
        s.focus = None;
    });
    tracing::info!("voicelink daemon stopped");
    result
}

/// Resolves on SIGTERM or SIGINT (Ctrl-C elsewhere).
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match (
            signal(SignalKind::terminate()),
            signal(SignalKind::interrupt()),
        ) {
            (Ok(mut term), Ok(mut int)) => {
                tokio::select! {
                    _ = term.recv() => tracing::info!("SIGTERM"),
                    _ = int.recv() => tracing::info!("SIGINT"),
                }
            }
            _ => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
