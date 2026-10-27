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

use crate::stack::ComposeProfile;
use choices::{Choices, Source, Tier};

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
    /// Compose profile for the slots served by the bundled stack (default: from the tier).
    #[arg(long, value_enum)]
    pub profile: Option<ComposeProfile>,
    /// Speech-to-text source: the compose stack, a local server, or agentbox Nemotron.
    #[arg(long, value_enum)]
    pub stt: Option<Source>,
    /// whisper model the compose stack serves (see `voicelink setup --list-models`).
    #[arg(long)]
    pub whisper_model: Option<String>,
    /// Base URL of a local STT server (implies `--stt local`).
    #[arg(long)]
    pub stt_url: Option<String>,
    /// Model name sent to the HTTP STT server.
    #[arg(long)]
    pub stt_model: Option<String>,
    /// Concierge model source: the compose stack, a local server, or the agentbox Loom.
    #[arg(long, value_enum)]
    pub llm: Option<Source>,
    /// GGUF file in the models directory that the compose llama.cpp server loads.
    #[arg(long)]
    pub llm_file: Option<String>,
    /// Base URL of a local LLM server (implies `--llm local`).
    #[arg(long)]
    pub llm_url: Option<String>,
    /// LLM model name.
    #[arg(long)]
    pub llm_model: Option<String>,
    /// Text-to-speech source: the compose stack, a local server, or agentbox Pocket TTS.
    #[arg(long, value_enum)]
    pub tts: Option<Source>,
    /// Base URL of a local TTS server (implies `--tts local`).
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
    /// After writing the config, run `docker compose ... up -d` for the stack
    /// (headless only; the interactive wizard asks for confirmation instead).
    #[arg(long)]
    pub start_stack: bool,
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
        if let Some(p) = self.profile {
            c.profile = p;
        }
        if let Some(k) = self.stt {
            c.stt_source = k;
        }
        if let Some(id) = &self.whisper_model {
            if catalogue::find(id).is_none() {
                bail!(
                    "unknown whisper model {id:?}; choose one of: {}",
                    catalogue::CATALOGUE
                        .iter()
                        .map(|e| e.id)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            c.whisper_model = id.clone();
            c.whisper_override = None;
        }
        if let Some(v) = &self.stt_url {
            c.stt_url = v.clone();
            c.stt_source = self.stt.unwrap_or(Source::Local);
        }
        if let Some(v) = &self.stt_model {
            c.stt_model = v.clone();
        }
        if let Some(k) = self.llm {
            c.llm_source = k;
        }
        if let Some(v) = &self.llm_file {
            c.llm_file = v.clone();
        }
        if let Some(v) = &self.llm_url {
            c.llm_url = v.clone();
            c.llm_source = self.llm.unwrap_or(Source::Local);
        }
        if let Some(v) = &self.llm_model {
            c.llm_model = v.clone();
        }
        if let Some(k) = self.tts {
            c.tts_source = k;
        }
        if let Some(v) = &self.tts_url {
            c.tts_url = v.clone();
            c.tts_source = self.tts.unwrap_or(Source::Local);
        }
        if let Some(v) = &self.tts_model {
            c.tts_model = v.clone();
        }
        if let Some(v) = &self.tts_voice {
            // Pocket TTS and Kokoro name their voices differently.
            if c.tts_source == Source::Remote {
                c.pocket_voice = v.clone();
            } else {
                c.tts_voice = v.clone();
            }
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

/// Loads the existing config. A config from an embedded-engine build is not an
/// error here: its connection, audio, hotkey and LLM settings are kept and the
/// speech slots restart from the defaults, so `voicelink setup` is the fix the
/// loader's error message points at.
fn existing_config(path: &std::path::Path) -> anyhow::Result<Option<crate::config::Config>> {
    if !path.exists() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    match store::parse(&text) {
        Ok(c) => Ok(Some(c)),
        Err(e) if e.downcast_ref::<store::LegacyEmbedded>().is_some() => {
            let defaults = Choices::defaults_for(Tier::Thin, &client_host()).to_config();
            match store::migrate_legacy(&text, &defaults) {
                Some(c) => {
                    eprintln!(
                        "note: {} selects an embedded speech engine, which this voicelink no longer includes. Keeping your ssh, audio, hotkey and LLM settings; choose the speech servers again.",
                        path.display()
                    );
                    Ok(Some(c))
                }
                None => Err(e).with_context(|| format!("parsing {}", path.display())),
            }
        }
        Err(e) => Err(e).with_context(|| format!("parsing {}", path.display())),
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
        (None, Some(t)) => {
            let mut c = Choices::defaults_for(t, host);
            c.apply_arch(std::env::consts::ARCH);
            c
        }
        (None, None) => {
            let p = probe::probe().propose();
            for l in &p.evidence {
                eprintln!("{l}");
            }
            let mut c = Choices::defaults_for(p.tier, host);
            c.profile = p.profile;
            c.apply_arch(std::env::consts::ARCH);
            c
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
    if args.start_stack {
        start_stack(&cfg)?;
    } else if let Some(stack) = &cfg.stack {
        match crate::stack::locate()
            .and_then(|f| crate::stack::ComposeCommand::from_config(&cfg, &f))
        {
            Some(cmd) => println!(
                "The {} compose stack is not started yet. Start it with:\n  {}",
                stack.profile.name(),
                cmd.display()
            ),
            None => println!(
                "The {} compose stack is not started yet and compose/compose.yml was not found next to the binary; see compose/README.md in the release tarball.",
                stack.profile.name()
            ),
        }
    }
    Ok(())
}

/// Headless `--start-stack`: the only place outside the wizard that runs docker.
fn start_stack(cfg: &crate::config::Config) -> anyhow::Result<()> {
    let file = crate::stack::locate().context(
        "--start-stack: compose/compose.yml was not found next to the binary (set VOICELINK_COMPOSE to its path)",
    )?;
    let cmd = crate::stack::ComposeCommand::from_config(cfg, &file)
        .context("--start-stack: no model slot uses the compose stack")?;
    eprintln!("running: {}", cmd.display());
    let status = cmd
        .run_inherit()
        .context("running docker (is Docker installed?)")?;
    if !status.success() {
        bail!("docker compose failed ({status})");
    }
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
            "remote",
            "--tts",
            "remote",
            "--profile",
            "vulkan",
            "--hotkey",
            "Ctrl+Shift+M",
            "--ssh-option=-p",
            "--ssh-option=2222",
        ]);
        let mut c = Choices::defaults_for(Tier::Nvidia, "h");
        a.apply(&mut c, "h").unwrap();
        assert_eq!(c.tier, Tier::Cpu);
        assert_eq!(c.profile, ComposeProfile::Vulkan);
        assert_eq!(c.ssh_destination, "me@tail");
        assert_eq!(c.llm_source, Source::Remote);
        assert_eq!(c.tts_source, Source::Remote);
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
            "compose",
            "--whisper-model",
            "whisper-small-q8",
            "--llm-url",
            "http://127.0.0.1:11434/v1",
            "--llm-model",
            "qwen",
            "--stt-url",
            "http://127.0.0.1:8080/v1",
        ]);
        let mut c = Choices::defaults_for(Tier::Cpu, "h");
        a.apply(&mut c, "h").unwrap();
        // A URL implies the local source unless a source was named explicitly.
        assert_eq!(c.stt_source, Source::Compose);
        assert_eq!(c.whisper_model, "whisper-small-q8");
        assert_eq!(c.llm_source, Source::Local);
        assert_eq!(c.llm_url, "http://127.0.0.1:11434/v1");
        assert_eq!(
            parse(&["--stt-url", "http://x/v1"]).stt,
            None,
            "no source flag"
        );
        let mut d = Choices::defaults_for(Tier::Cpu, "h");
        parse(&["--stt-url", "http://x/v1"])
            .apply(&mut d, "h")
            .unwrap();
        assert_eq!(d.stt_source, Source::Local);
        assert!(
            parse(&["--whisper-model", "nope"])
                .apply(&mut c, "h")
                .is_err()
        );
    }

    #[test]
    fn the_voice_flag_follows_the_tts_source() {
        let mut c = Choices::defaults_for(Tier::Nvidia, "h");
        parse(&["--tts-voice", "bf_emma"])
            .apply(&mut c, "h")
            .unwrap();
        assert_eq!(
            (c.tts_voice.as_str(), c.pocket_voice.as_str()),
            ("bf_emma", "alba")
        );
        parse(&["--tts", "remote", "--tts-voice", "jane"])
            .apply(&mut c, "h")
            .unwrap();
        assert_eq!(
            (c.tts_voice.as_str(), c.pocket_voice.as_str()),
            ("bf_emma", "jane")
        );
    }

    #[test]
    fn the_embedded_source_is_gone() {
        assert!(Wrap::try_parse_from(["setup", "--stt", "embedded"]).is_err());
        assert!(Wrap::try_parse_from(["setup", "--tts", "embedded"]).is_err());
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
        assert!(cfg.stack.is_none());
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
    fn rerunning_setup_over_an_embedded_config_migrates_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.toml");
        std::fs::write(&path, store::LEGACY).unwrap();
        // The loader names the fix, with a typed error...
        let e = store::load(&path).unwrap_err();
        assert!(format!("{e:#}").contains("voicelink setup"), "{e:#}");
        // ...and setup carries on: connection, audio and hotkey survive.
        let a = parse(&[
            "--non-interactive",
            "--no-download",
            "--config-path",
            path.to_str().unwrap(),
        ]);
        run(a).unwrap();
        let cfg = store::load(&path).unwrap();
        assert_eq!(cfg.connection.ssh_destination, "me@box");
        assert_eq!(cfg.audio.input_device.as_deref(), Some("USB Mic"));
        assert!(cfg.hotkey.enabled);
        assert!(matches!(
            cfg.stt,
            voicelink_backends::SttConfig::Nemotron { .. }
        ));
        // A file too broken to migrate still ends in the same advice.
        std::fs::write(&path, "[stt]\nkind = \"whisper\"\n").unwrap();
        let a = parse(&[
            "--non-interactive",
            "--no-download",
            "--tier",
            "thin",
            "--config-path",
            path.to_str().unwrap(),
        ]);
        let e = run(a).unwrap_err();
        assert!(format!("{e:#}").contains("voicelink setup"), "{e:#}");
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
