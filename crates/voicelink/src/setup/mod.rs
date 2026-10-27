//! Onboarding wizard: `voicelink setup`.
//!
//! The interactive wizard (`ui`) drives a pure state machine ([`wizard`]);
//! `--non-interactive` builds the same configuration from flags. Either way
//! the terminal belongs to setup only until it returns, which is before the
//! launcher hands it to tmux (I6).

pub mod catalogue;
pub mod choices;
mod effects;
pub mod net;
pub mod probe;
pub mod remote;
pub mod store;
mod ui;
pub mod wizard;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, bail};
use clap::Args;

use choices::{Choices, LlmKind, SttKind, Tier, TtsKind};

/// Flags of `voicelink setup`. Every wizard choice has a flag so setup can run
/// headless with `--non-interactive`.
#[derive(Debug, Clone, Default, Args)]
pub struct SetupArgs {
    /// Do not open the wizard: build the configuration from flags and defaults.
    #[arg(long)]
    pub non_interactive: bool,
    /// ssh destination of the Docker host, `user@host` (LAN address or tailnet name).
    #[arg(long)]
    pub ssh: Option<String>,
    /// Extra ssh option (repeatable), for example `--ssh-option=-p --ssh-option=2222`.
    #[arg(long = "ssh-option", allow_hyphen_values = true)]
    pub ssh_options: Vec<String>,
    /// Hardware tier; probed when omitted.
    #[arg(long, value_enum)]
    pub tier: Option<Tier>,
    /// Speech-to-text provider.
    #[arg(long, value_enum)]
    pub stt: Option<SttKind>,
    /// Embedded whisper model id (see `voicelink setup --list-models`).
    #[arg(long)]
    pub whisper_model: Option<String>,
    /// Base URL of an HTTP STT server.
    #[arg(long)]
    pub stt_url: Option<String>,
    /// Model name sent to the HTTP STT server.
    #[arg(long)]
    pub stt_model: Option<String>,
    /// Concierge LLM provider.
    #[arg(long, value_enum)]
    pub llm: Option<LlmKind>,
    /// Base URL of a local LLM server (with `--llm local`).
    #[arg(long)]
    pub llm_url: Option<String>,
    /// LLM model name.
    #[arg(long)]
    pub llm_model: Option<String>,
    /// Text-to-speech provider.
    #[arg(long, value_enum)]
    pub tts: Option<TtsKind>,
    /// Base URL of an HTTP TTS server.
    #[arg(long)]
    pub tts_url: Option<String>,
    /// Model name sent to the HTTP TTS server.
    #[arg(long)]
    pub tts_model: Option<String>,
    /// TTS voice.
    #[arg(long)]
    pub tts_voice: Option<String>,
    /// Input device name (default: system default).
    #[arg(long)]
    pub input_device: Option<String>,
    /// Output device name (default: system default).
    #[arg(long)]
    pub output_device: Option<String>,
    /// Longest utterance passed to STT, in milliseconds (1000 to 30000).
    #[arg(long)]
    pub max_utterance_ms: Option<u32>,
    /// Enable the global mute hotkey with this accelerator, for example `Ctrl+Alt+M`.
    #[arg(long)]
    pub hotkey: Option<String>,
    /// Models directory.
    #[arg(long)]
    pub models_dir: Option<PathBuf>,
    /// Skip downloading models.
    #[arg(long)]
    pub no_download: bool,
    /// Also run the ssh and protocol checks and fail if they fail.
    #[arg(long)]
    pub verify: bool,
    /// Print the model catalogue and exit.
    #[arg(long)]
    pub list_models: bool,
    /// Config file to read and write instead of the default location.
    #[arg(long, hide = true)]
    pub config_path: Option<PathBuf>,
}

impl SetupArgs {
    /// Applies the flags over `c`.
    pub fn apply(&self, c: &mut Choices, host: &str) -> anyhow::Result<()> {
        if let Some(t) = self.tier {
            c.apply_tier(t, host);
        }
        if let Some(v) = &self.ssh {
            c.ssh_destination = v.clone();
        }
        if !self.ssh_options.is_empty() {
            c.connection_mut().ssh_options = self.ssh_options.clone();
        }
        if let Some(k) = self.stt {
            c.stt_kind = k;
        }
        if let Some(id) = &self.whisper_model {
            if catalogue::find(id).is_none_or(|e| e.kind != catalogue::Kind::Whisper) {
                bail!(
                    "unknown whisper model {id:?}; choose one of: {}",
                    catalogue::whisper_entries()
                        .map(|e| e.id)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            c.whisper_model = id.clone();
            c.whisper_path_override = None;
        }
        if let Some(v) = &self.stt_url {
            c.stt_url = v.clone();
            c.stt_kind = self.stt.unwrap_or(SttKind::Http);
        }
        if let Some(v) = &self.stt_model {
            c.stt_model = v.clone();
        }
        if let Some(k) = self.llm {
            c.llm_kind = k;
        }
        if let Some(v) = &self.llm_url {
            c.llm_url = v.clone();
            c.llm_kind = self.llm.unwrap_or(LlmKind::Local);
        }
        if let Some(v) = &self.llm_model {
            c.llm_model = v.clone();
        }
        if let Some(k) = self.tts {
            c.tts_kind = k;
        }
        if let Some(v) = &self.tts_url {
            c.tts_url = v.clone();
            c.tts_kind = self.tts.unwrap_or(TtsKind::Http);
        }
        if let Some(v) = &self.tts_model {
            c.tts_model = v.clone();
        }
        if let Some(v) = &self.tts_voice {
            c.tts_voice = v.clone();
            c.kokoro_voice = v.clone();
        }
        if let Some(v) = &self.input_device {
            c.input_device = Some(v.clone());
        }
        if let Some(v) = &self.output_device {
            c.output_device = Some(v.clone());
        }
        if let Some(v) = self.max_utterance_ms {
            c.max_utterance_ms = v;
        }
        if let Some(v) = &self.hotkey {
            c.hotkey_enabled = true;
            c.hotkey_accel = v.clone();
        }
        if let Some(d) = &self.models_dir {
            c.models_dir = d.clone();
        }
        Ok(())
    }
}

fn client_host() -> String {
    probe::capture("hostname", &["-s"], std::time::Duration::from_secs(2))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "client".into())
}

fn existing_config(path: &std::path::Path) -> anyhow::Result<Option<crate::config::Config>> {
    if path.exists() {
        store::load(path).map(Some)
    } else {
        Ok(None)
    }
}

/// Runs setup: the wizard on a terminal, or the flag-driven path with `--non-interactive`.
pub fn run(args: SetupArgs) -> anyhow::Result<()> {
    if args.list_models {
        for e in catalogue::CATALOGUE {
            println!(
                "{:<28} {:>10} B  sha256 {}  {}",
                e.id, e.size, e.sha256, e.url
            );
        }
        return Ok(());
    }
    let path = match &args.config_path {
        Some(p) => p.clone(),
        None => crate::paths::config_file()?,
    };
    let existing = existing_config(&path)
        .with_context(|| format!("reading the existing config {}", path.display()))?;
    let host = client_host();
    if args.non_interactive {
        return run_headless(&args, existing, &path, &host);
    }
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        bail!(
            "`voicelink setup` needs an interactive terminal; for scripts use `voicelink setup --non-interactive --ssh user@host --tier cpu ...`"
        );
    }
    if args.config_path.is_some() {
        bail!("--config-path is only supported with --non-interactive");
    }
    let mut wiz = wizard::Wizard::new(existing.as_ref(), &host);
    args.apply(&mut wiz.choices, &host)?;
    match ui::run(wiz)? {
        wizard::Finish::Saved(p) => {
            println!("Wrote {}", p.display());
            Ok(())
        }
        wizard::Finish::Cancelled => {
            println!("Setup cancelled; nothing was written.");
            Ok(())
        }
    }
}

fn run_headless(
    args: &SetupArgs,
    existing: Option<crate::config::Config>,
    path: &std::path::Path,
    host: &str,
) -> anyhow::Result<()> {
    let mut choices = match (&existing, args.tier) {
        (Some(cfg), _) => Choices::from_config(cfg),
        (None, Some(t)) => Choices::defaults_for(t, host),
        (None, None) => {
            let p = probe::probe().propose();
            for l in &p.evidence {
                eprintln!("{l}");
            }
            Choices::defaults_for(p.tier, host)
        }
    };
    args.apply(&mut choices, host)?;
    if existing.is_none() && args.ssh.is_none() {
        eprintln!(
            "note: using the default ssh destination {}",
            choices.ssh_destination
        );
    }
    let problems = choices.validate();
    if !problems.is_empty() {
        bail!("invalid configuration:\n  {}", problems.join("\n  "));
    }
    if !args.no_download {
        for e in choices.required_models() {
            let mut next = 0u64;
            let cancel = AtomicBool::new(false);
            eprintln!("fetching {} ...", e.file);
            net::download(e, &choices.models_dir, &cancel, &mut |d, t| {
                if t > 0 && d * 10 / t >= next {
                    eprintln!("  {}%", d * 100 / t);
                    next = d * 10 / t + 1;
                }
            })
            .with_context(|| format!("downloading {}", e.id))?;
        }
    }
    let cfg = choices.to_config();
    if args.verify {
        match remote::check_version(&cfg.connection) {
            remote::Verdict::Ready(v) => eprintln!("voice-paned: {v}"),
            remote::Verdict::Problem(p) => bail!("{p}"),
        }
        match remote::handshake(&cfg.connection) {
            Ok(w) => eprintln!("handshake ok: {} (focus {})", w.server, w.focus),
            Err(e) => bail!("{e}"),
        }
    }
    store::save(path, &cfg)?;
    println!("Wrote {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        args: SetupArgs,
    }

    fn parse(argv: &[&str]) -> SetupArgs {
        Wrap::try_parse_from(std::iter::once("setup").chain(argv.iter().copied()))
            .unwrap()
            .args
    }

    #[test]
    fn flags_override_defaults() {
        let a = parse(&[
            "--non-interactive",
            "--ssh",
            "me@tail",
            "--tier",
            "cpu",
            "--llm",
            "loom",
            "--tts",
            "pocket",
            "--hotkey",
            "Ctrl+Shift+M",
            "--ssh-option=-p",
            "--ssh-option=2222",
        ]);
        let mut c = Choices::defaults_with(
            crate::setup::choices::Engines {
                whisper: true,
                kokoro: true,
            },
            Tier::Nvidia,
            "h",
        );
        a.apply(&mut c, "h").unwrap();
        assert_eq!(c.tier, Tier::Cpu);
        assert_eq!(c.ssh_destination, "me@tail");
        assert_eq!(c.llm_kind, LlmKind::Loom);
        assert_eq!(c.tts_kind, TtsKind::Pocket);
        assert!(c.hotkey_enabled && c.hotkey_accel == "Ctrl+Shift+M");
        assert_eq!(c.to_config().connection.ssh_options, vec!["-p", "2222"]);
        assert_eq!(c.whisper_model, "whisper-base-en-q8");
    }

    #[test]
    fn explicit_flags_beat_tier_defaults() {
        let a = parse(&[
            "--tier",
            "thin",
            "--stt",
            "embedded",
            "--whisper-model",
            "whisper-small-q8",
            "--llm-url",
            "http://127.0.0.1:11434/v1",
            "--llm-model",
            "qwen",
        ]);
        let mut c = Choices::defaults_with(
            crate::setup::choices::Engines {
                whisper: true,
                kokoro: true,
            },
            Tier::Cpu,
            "h",
        );
        a.apply(&mut c, "h").unwrap();
        assert_eq!(c.stt_kind, SttKind::Embedded);
        assert_eq!(c.whisper_model, "whisper-small-q8");
        assert_eq!(c.llm_kind, LlmKind::Local);
        assert_eq!(c.llm_url, "http://127.0.0.1:11434/v1");
        assert!(
            parse(&["--whisper-model", "nope"])
                .apply(&mut c, "h")
                .is_err()
        );
    }

    #[test]
    fn headless_writes_config_without_network() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.toml");
        let a = parse(&[
            "--non-interactive",
            "--ssh",
            "machinelearn@192.168.2.132",
            "--tier",
            "thin",
            "--no-download",
            "--config-path",
            path.to_str().unwrap(),
        ]);
        run(a).unwrap();
        let cfg = store::load(&path).unwrap();
        assert_eq!(cfg.connection.forwards.len(), 3);
        // Re-running pre-fills from the file and keeps edits.
        let a = parse(&[
            "--non-interactive",
            "--no-download",
            "--ssh",
            "me@other",
            "--config-path",
            path.to_str().unwrap(),
        ]);
        run(a).unwrap();
        let again = store::load(&path).unwrap();
        assert_eq!(again.connection.ssh_destination, "me@other");
        assert_eq!(again.connection.forwards.len(), 3);
    }

    #[test]
    fn headless_rejects_bad_destination() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.toml");
        let a = parse(&[
            "--non-interactive",
            "--ssh=-oEvil",
            "--tier",
            "cpu",
            "--no-download",
            "--config-path",
            path.to_str().unwrap(),
        ]);
        assert!(run(a).is_err());
        assert!(!path.exists());
    }
}
