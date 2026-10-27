//! The wizard's editable model of a configuration, and its projection to and
//! from [`Config`].

use std::path::PathBuf;

use serde_json::{Map, Value, json};
use voicelink_backends::{LlmConfig, SttConfig, TtsConfig};

use super::catalogue::{self, Entry};
use crate::config::{Config, Connection, Forward, Slot};
use crate::stack::{ComposeProfile, DEFAULT_LLM_FILE, DEFAULT_WHISPER_FILE, Stack};

/// Local port of the forward to the agentbox Ontology Loom.
pub const LOOM_PORT: u16 = 18084;
/// Local port of the forward to agentbox Pocket TTS.
pub const POCKET_PORT: u16 = 18898;
/// Local port of the forward to remote Nemotron streaming ASR.
pub const NEMOTRON_PORT: u16 = 18897;
/// Hard ceiling for `audio.max_utterance_ms`: the remote Nemotron queue holds about 36 s.
pub const MAX_UTTERANCE_CAP_MS: u32 = 30_000;
/// Default ssh destination for the agentbox host.
pub const DEFAULT_SSH: &str = "machinelearn@192.168.2.132";
/// Base URL of the compose stack's llama.cpp server (`compose/compose.yml`).
pub const COMPOSE_LLM: &str = "http://127.0.0.1:8180/v1";
/// Base URL of the compose stack's whisper.cpp server.
pub const COMPOSE_STT: &str = "http://127.0.0.1:8181/v1";
/// Base URL of the compose stack's Kokoro-FastAPI server.
pub const COMPOSE_TTS: &str = "http://127.0.0.1:8182/v1";
/// Where a locally run llama-server listens by default.
pub const LOCAL_LLM: &str = "http://127.0.0.1:8080/v1";
/// Where a locally run whisper-server listens by default.
pub const LOCAL_STT: &str = "http://127.0.0.1:8080/v1";
/// Where a locally run Kokoro-FastAPI listens by default.
pub const LOCAL_TTS: &str = "http://127.0.0.1:8880/v1";

/// Hardware class, deciding the compose profile and the default model sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Tier {
    /// ROCm with at least 32 GiB unified or dedicated memory.
    #[value(name = "rocm-large")]
    RocmLarge,
    /// NVIDIA with at least 12 GiB VRAM.
    #[value(name = "nvidia")]
    Nvidia,
    /// Any smaller GPU (Vulkan).
    #[value(name = "small-gpu")]
    SmallGpu,
    /// Apple silicon: Docker has no GPU there, so native servers.
    #[value(name = "metal")]
    Metal,
    /// No usable GPU.
    #[value(name = "cpu")]
    Cpu,
    /// Thin client: everything is remote.
    #[value(name = "thin")]
    Thin,
}

impl Tier {
    /// Every tier, in menu order.
    pub const ALL: [Tier; 6] = [
        Tier::RocmLarge,
        Tier::Nvidia,
        Tier::SmallGpu,
        Tier::Metal,
        Tier::Cpu,
        Tier::Thin,
    ];

    /// Menu label.
    pub fn label(self) -> &'static str {
        match self {
            Tier::RocmLarge => "rocm-large",
            Tier::Nvidia => "nvidia",
            Tier::SmallGpu => "small-gpu (Vulkan)",
            Tier::Metal => "metal (Apple silicon, native servers)",
            Tier::Cpu => "cpu",
            Tier::Thin => "thin (remote speech)",
        }
    }

    /// Default whisper model id for the tier (`None` for thin clients).
    pub fn default_whisper(self) -> Option<&'static str> {
        Some(match self {
            Tier::RocmLarge => "whisper-large-v3-turbo",
            Tier::Nvidia => "whisper-medium-q8",
            Tier::SmallGpu => "whisper-small-q8",
            Tier::Metal => "whisper-large-v3-turbo-q8",
            Tier::Cpu => "whisper-base-en-q8",
            Tier::Thin => return None,
        })
    }

    /// The compose profile that matches the tier (Metal and thin clients do not
    /// use the stack by default; the profile is only a starting point there).
    pub fn profile(self) -> ComposeProfile {
        match self {
            Tier::RocmLarge => ComposeProfile::Rocm,
            Tier::Nvidia => ComposeProfile::Cuda,
            Tier::SmallGpu => ComposeProfile::Vulkan,
            Tier::Metal | Tier::Cpu | Tier::Thin => ComposeProfile::Cpu,
        }
    }

    /// The tier that a stored compose profile implies.
    pub fn from_profile(p: ComposeProfile) -> Tier {
        match p {
            ComposeProfile::Rocm => Tier::RocmLarge,
            ComposeProfile::Cuda => Tier::Nvidia,
            ComposeProfile::Vulkan => Tier::SmallGpu,
            ComposeProfile::Cpu => Tier::Cpu,
        }
    }

    /// Guidance on the concierge model size for the tier. The GGUF itself is
    /// the operator's choice: it must support tool calls.
    pub fn llm_hint(self) -> &'static str {
        match self {
            Tier::RocmLarge => {
                "a 27B-35B tool-calling model at Q4 (about 20 GB) fits in unified memory"
            }
            Tier::Nvidia => "an 8B-14B tool-calling model at Q4 (6-10 GB)",
            Tier::SmallGpu => "a 4B-8B tool-calling model at Q4 (3-5 GB)",
            Tier::Metal => "an 8B-14B tool-calling model through Ollama or LM Studio",
            Tier::Cpu => "CPU inference is slow: prefer the agentbox Loom, or a 4B model at Q4",
            Tier::Thin => "the agentbox Loom",
        }
    }
}

/// Where one model slot is served from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Source {
    /// The bundled Docker compose stack.
    #[value(name = "compose")]
    Compose,
    /// A server already running on this machine.
    #[value(name = "local")]
    Local,
    /// An agentbox service over an ssh forward (Nemotron, Pocket TTS, the Loom).
    #[value(name = "remote")]
    Remote,
}

impl Source {
    /// Every source, in menu order.
    pub const ALL: &'static [Source] = &[Source::Compose, Source::Local, Source::Remote];

    /// Menu label for `slot`.
    pub fn label(self, slot: Slot) -> &'static str {
        match (self, slot) {
            (Source::Compose, Slot::Stt) => "compose stack (whisper.cpp server)",
            (Source::Compose, Slot::Llm) => "compose stack (llama.cpp server)",
            (Source::Compose, Slot::Tts) => "compose stack (Kokoro-FastAPI)",
            (Source::Local, Slot::Stt) => "server already running (HTTP)",
            (Source::Local, Slot::Llm) => {
                "server already running (llama-server, Ollama, LM Studio)"
            }
            (Source::Local, Slot::Tts) => "server already running (HTTP)",
            (Source::Remote, Slot::Stt) => "agentbox Nemotron (ssh forward)",
            (Source::Remote, Slot::Llm) => "agentbox Ontology Loom (ssh forward)",
            (Source::Remote, Slot::Tts) => "agentbox Pocket TTS (ssh forward)",
        }
    }
}

/// Steps through `all` from `current` by `dir` (+1 or -1), wrapping.
pub fn cycle<T: Copy + PartialEq>(all: &[T], current: T, dir: i32) -> T {
    let i = all.iter().position(|x| *x == current).unwrap_or(0) as i32;
    all[(i + dir).rem_euclid(all.len() as i32) as usize]
}

/// Everything the wizard edits.
#[derive(Debug, Clone, PartialEq)]
pub struct Choices {
    /// Hardware tier.
    pub tier: Tier,
    /// ssh destination.
    pub ssh_destination: String,
    /// Compose profile for the slots served by the stack.
    pub profile: ComposeProfile,
    /// STT source.
    pub stt_source: Source,
    /// Catalogue id of the whisper model the stack serves.
    pub whisper_model: String,
    /// Explicit whisper file overriding the catalogue (from a hand-edited config).
    pub whisper_override: Option<String>,
    /// Local STT base URL.
    pub stt_url: String,
    /// STT model name sent in requests.
    pub stt_model: String,
    /// LLM source.
    pub llm_source: Source,
    /// Local LLM base URL.
    pub llm_url: String,
    /// LLM model name.
    pub llm_model: String,
    /// GGUF file (in the models directory) the stack's llama.cpp server loads.
    pub llm_file: String,
    /// TTS source.
    pub tts_source: Source,
    /// Local TTS base URL.
    pub tts_url: String,
    /// TTS model name.
    pub tts_model: String,
    /// TTS voice of a Kokoro server (compose or local).
    pub tts_voice: String,
    /// Voice sent to agentbox Pocket TTS, which has its own voice names.
    pub pocket_voice: String,
    /// Input device (`None` = system default).
    pub input_device: Option<String>,
    /// Output device (`None` = system default).
    pub output_device: Option<String>,
    /// Register the mute hotkey.
    pub hotkey_enabled: bool,
    /// Hotkey accelerator.
    pub hotkey_accel: String,
    /// Longest utterance passed to STT, in milliseconds (at most [`MAX_UTTERANCE_CAP_MS`]).
    pub max_utterance_ms: u32,
    /// Where models live; the compose stack mounts the same directory.
    pub models_dir: PathBuf,
    /// The configuration being edited; fields the wizard does not own pass through.
    base: Config,
}

impl Choices {
    /// Defaults for `tier` on a machine called `client_host`.
    pub fn defaults_for(tier: Tier, client_host: &str) -> Self {
        let models_dir = crate::paths::models_dir().unwrap_or_else(|_| PathBuf::from("models"));
        let base = Config {
            connection: Connection::agentbox(DEFAULT_SSH, client_host),
            audio: Default::default(),
            hotkey: Default::default(),
            concierge: Default::default(),
            stt: SttConfig::Nemotron {
                url: nemotron_url(),
            },
            llm: LlmConfig::OpenAi {
                base_url: LOCAL_LLM.into(),
                model: "default".into(),
                api_key_env: None,
                extra: Map::new(),
            },
            tts: TtsConfig::OpenAi {
                base_url: pocket_url(),
                model: "pocket-tts".into(),
                voice: "alba".into(),
                api_key_env: None,
            },
            models_dir: models_dir.clone(),
            stack: None,
        };
        let (stt_source, llm_source, tts_source) = tier_sources(tier);
        Choices {
            tier,
            ssh_destination: DEFAULT_SSH.into(),
            profile: tier.profile(),
            stt_source,
            whisper_model: tier
                .default_whisper()
                .unwrap_or("whisper-base-en-q8")
                .into(),
            whisper_override: None,
            stt_url: LOCAL_STT.into(),
            stt_model: "whisper-1".into(),
            llm_source,
            llm_url: LOCAL_LLM.into(),
            llm_model: "default".into(),
            llm_file: DEFAULT_LLM_FILE.into(),
            tts_source,
            tts_url: LOCAL_TTS.into(),
            tts_model: "kokoro".into(),
            tts_voice: "af_heart".into(),
            pocket_voice: "alba".into(),
            input_device: None,
            output_device: None,
            hotkey_enabled: false,
            hotkey_accel: "Ctrl+Alt+M".into(),
            max_utterance_ms: MAX_UTTERANCE_CAP_MS,
            models_dir,
            base,
        }
    }

    /// Resets the tier-dependent choices (sources, compose profile, whisper
    /// size) to the defaults of `tier`, keeping everything else.
    pub fn apply_tier(&mut self, tier: Tier, client_host: &str) {
        let d = Choices::defaults_for(tier, client_host);
        self.tier = tier;
        self.profile = d.profile;
        self.stt_source = d.stt_source;
        self.llm_source = d.llm_source;
        self.tts_source = d.tts_source;
        self.whisper_model = d.whisper_model;
        self.whisper_override = None;
    }

    /// Moves every stack slot whose image is not published for `arch` to the
    /// agentbox service, and returns the slots it moved. The whisper.cpp
    /// images are `linux/amd64` only, so an aarch64 Linux host cannot take
    /// speech-to-text from the stack.
    pub fn apply_arch(&mut self, arch: &str) -> Vec<Slot> {
        if arch.is_empty() {
            return Vec::new();
        }
        let moved = crate::stack::unavailable(self.profile, &self.stack_slots(), arch);
        for slot in &moved {
            match slot {
                Slot::Stt => self.stt_source = Source::Remote,
                Slot::Llm => self.llm_source = Source::Remote,
                Slot::Tts => self.tts_source = Source::Remote,
            }
        }
        moved
    }

    /// Pre-fills the wizard from an existing configuration.
    pub fn from_config(cfg: &Config) -> Self {
        let mut c = Choices::defaults_for(Tier::Cpu, "client");
        c.ssh_destination = cfg.connection.ssh_destination.clone();
        c.models_dir = cfg.models_dir.clone();
        c.input_device = cfg.audio.input_device.clone();
        c.output_device = cfg.audio.output_device.clone();
        c.hotkey_enabled = cfg.hotkey.enabled;
        c.hotkey_accel = cfg.hotkey.mute.clone();
        c.max_utterance_ms = cfg.audio.max_utterance_ms;
        match &cfg.stt {
            SttConfig::OpenAi {
                base_url, model, ..
            } => {
                c.stt_source = if base_url == COMPOSE_STT {
                    Source::Compose
                } else {
                    Source::Local
                };
                c.stt_url = base_url.clone();
                c.stt_model = model.clone();
            }
            SttConfig::Nemotron { .. } => c.stt_source = Source::Remote,
        }
        let LlmConfig::OpenAi {
            base_url, model, ..
        } = &cfg.llm;
        c.llm_model = model.clone();
        c.llm_url = base_url.clone();
        c.llm_source = if *base_url == loom_url() {
            Source::Remote
        } else if base_url == COMPOSE_LLM {
            Source::Compose
        } else {
            Source::Local
        };
        let TtsConfig::OpenAi {
            base_url,
            model,
            voice,
            ..
        } = &cfg.tts;
        c.tts_url = base_url.clone();
        c.tts_model = model.clone();
        if *base_url != pocket_url() {
            c.tts_voice = voice.clone();
        }
        c.tts_source = if *base_url == pocket_url() {
            c.pocket_voice = voice.clone();
            Source::Remote
        } else if base_url == COMPOSE_TTS {
            Source::Compose
        } else {
            Source::Local
        };
        if let Some(stack) = &cfg.stack {
            c.profile = stack.profile;
            c.tier = Tier::from_profile(stack.profile);
            c.llm_file = stack.llm_file.clone();
            match catalogue::find_by_file(&stack.whisper_file) {
                Some(e) => c.whisper_model = e.id.into(),
                None => c.whisper_override = Some(stack.whisper_file.clone()),
            }
        } else if [c.stt_source, c.llm_source, c.tts_source]
            .iter()
            .all(|s| *s == Source::Remote)
        {
            c.tier = Tier::Thin;
        } else if cfg!(target_os = "macos") {
            c.tier = Tier::Metal;
        }
        c.base = cfg.clone();
        c
    }

    /// The whisper catalogue entry, when not overridden.
    pub fn whisper_entry(&self) -> Option<&'static Entry> {
        catalogue::find(&self.whisper_model)
    }

    /// The whisper file the stack serves.
    pub fn whisper_file(&self) -> String {
        self.whisper_override
            .clone()
            .or_else(|| self.whisper_entry().map(|e| e.file.to_string()))
            .unwrap_or_else(|| DEFAULT_WHISPER_FILE.into())
    }

    /// The source of `slot`.
    pub fn source(&self, slot: Slot) -> Source {
        match slot {
            Slot::Stt => self.stt_source,
            Slot::Llm => self.llm_source,
            Slot::Tts => self.tts_source,
        }
    }

    /// The slots served by the compose stack.
    pub fn stack_slots(&self) -> Vec<Slot> {
        [Slot::Stt, Slot::Llm, Slot::Tts]
            .into_iter()
            .filter(|s| self.source(*s) == Source::Compose)
            .collect()
    }

    /// The `[stack]` table, when any slot uses the stack.
    pub fn stack(&self) -> Option<Stack> {
        (!self.stack_slots().is_empty()).then(|| Stack {
            profile: self.profile,
            whisper_file: self.whisper_file(),
            llm_file: self.llm_file.clone(),
        })
    }

    /// The forwards the choices need (named `loom`, `pocket`, `nemotron`),
    /// followed by any custom forwards carried over from the previous config.
    pub fn forwards(&self) -> Vec<Forward> {
        let mut out = Vec::new();
        if self.stt_source == Source::Remote {
            out.push(Forward {
                name: "nemotron".into(),
                local_port: NEMOTRON_PORT,
                remote: "127.0.0.1:8897".into(),
            });
        }
        if self.llm_source == Source::Remote {
            out.push(Forward {
                name: "loom".into(),
                local_port: LOOM_PORT,
                remote: "127.0.0.1:8084".into(),
            });
        }
        if self.tts_source == Source::Remote {
            out.push(Forward {
                name: "pocket".into(),
                local_port: POCKET_PORT,
                remote: "127.0.0.1:8898".into(),
            });
        }
        out.extend(
            self.base
                .connection
                .forwards
                .iter()
                .filter(|f| !matches!(f.name.as_str(), "loom" | "pocket" | "nemotron"))
                .cloned(),
        );
        out
    }

    /// Models the current choices need on disk.
    pub fn required_models(&self) -> Vec<&'static Entry> {
        let mut v = Vec::new();
        if self.stt_source == Source::Compose
            && self.whisper_override.is_none()
            && let Some(e) = self.whisper_entry()
        {
            v.push(e);
        }
        v
    }

    /// Builds the configuration.
    pub fn to_config(&self) -> Config {
        let mut cfg = self.base.clone();
        cfg.connection.ssh_destination = self.ssh_destination.trim().to_string();
        cfg.connection.forwards = self.forwards();
        cfg.audio.input_device = self.input_device.clone();
        cfg.audio.output_device = self.output_device.clone();
        cfg.hotkey.enabled = self.hotkey_enabled;
        cfg.hotkey.mute = self.hotkey_accel.clone();
        cfg.audio.max_utterance_ms = self.max_utterance_ms;
        cfg.models_dir = self.models_dir.clone();
        cfg.stack = self.stack();
        let base_stt_key = match &self.base.stt {
            SttConfig::OpenAi { api_key_env, .. } => api_key_env.clone(),
            SttConfig::Nemotron { .. } => None,
        };
        cfg.stt = match self.stt_source {
            Source::Compose => SttConfig::OpenAi {
                base_url: COMPOSE_STT.into(),
                model: self.stt_model.clone(),
                api_key_env: None,
            },
            Source::Local => SttConfig::OpenAi {
                base_url: self.stt_url.clone(),
                model: self.stt_model.clone(),
                api_key_env: base_stt_key,
            },
            Source::Remote => SttConfig::Nemotron {
                url: nemotron_url(),
            },
        };
        let (base_url, extra) = match self.llm_source {
            Source::Local => (self.llm_url.clone(), Map::new()),
            Source::Compose => (COMPOSE_LLM.to_string(), Map::new()),
            Source::Remote => (loom_url(), loom_extra()),
        };
        let LlmConfig::OpenAi { api_key_env, .. } = &self.base.llm;
        cfg.llm = LlmConfig::OpenAi {
            base_url,
            model: self.llm_model.clone(),
            api_key_env: api_key_env.clone(),
            extra,
        };
        let TtsConfig::OpenAi {
            api_key_env: base_tts_key,
            ..
        } = &self.base.tts;
        cfg.tts = match self.tts_source {
            Source::Compose => TtsConfig::OpenAi {
                base_url: COMPOSE_TTS.into(),
                model: self.tts_model.clone(),
                voice: self.tts_voice.clone(),
                api_key_env: None,
            },
            Source::Local => TtsConfig::OpenAi {
                base_url: self.tts_url.clone(),
                model: self.tts_model.clone(),
                voice: self.tts_voice.clone(),
                api_key_env: base_tts_key.clone(),
            },
            Source::Remote => TtsConfig::OpenAi {
                base_url: pocket_url(),
                model: "pocket-tts".into(),
                voice: self.pocket_voice.clone(),
                api_key_env: base_tts_key.clone(),
            },
        };
        cfg
    }

    /// Problems that would make the configuration unusable; empty when fine.
    pub fn validate(&self) -> Vec<String> {
        let mut p = Vec::new();
        let d = self.ssh_destination.trim();
        if d.is_empty() {
            p.push("ssh destination is empty".into());
        } else if d.starts_with('-') || d.chars().any(char::is_whitespace) {
            p.push("ssh destination must not start with '-' or contain whitespace".into());
        }
        let http = |u: &str| u.starts_with("http://") || u.starts_with("https://");
        if self.stt_source == Source::Local && !http(&self.stt_url) {
            p.push("STT URL must start with http:// or https://".into());
        }
        if self.llm_source == Source::Local && !http(&self.llm_url) {
            p.push("LLM URL must start with http:// or https://".into());
        }
        if self.tts_source == Source::Local && !http(&self.tts_url) {
            p.push("TTS URL must start with http:// or https://".into());
        }
        if self.llm_model.trim().is_empty() {
            p.push("LLM model name is empty".into());
        }
        if self.llm_source == Source::Compose
            && (self.llm_file.trim().is_empty() || self.llm_file.contains('/'))
        {
            p.push(
                "the GGUF file name must be a plain file name inside the models directory".into(),
            );
        }
        if !(1000..=MAX_UTTERANCE_CAP_MS).contains(&self.max_utterance_ms) {
            p.push(format!(
                "max utterance must be between 1000 and {MAX_UTTERANCE_CAP_MS} ms"
            ));
        }
        if self.hotkey_enabled && self.hotkey_accel.trim().is_empty() {
            p.push("hotkey accelerator is empty".into());
        }
        let c = &self.base.connection;
        for (what, v) in [("session", &c.session), ("view session", &c.view_session)] {
            if v.is_empty() || v.contains([':', '.']) || v.starts_with('-') {
                p.push(format!("tmux {what} name {v:?} is not usable"));
            }
        }
        p
    }

    /// Mutable access to the passed-through connection settings (headless flags).
    pub fn connection_mut(&mut self) -> &mut Connection {
        &mut self.base.connection
    }
}

/// Default `(stt, llm, tts)` sources for `tier`.
fn tier_sources(tier: Tier) -> (Source, Source, Source) {
    match tier {
        Tier::RocmLarge | Tier::Nvidia | Tier::SmallGpu => {
            (Source::Compose, Source::Compose, Source::Compose)
        }
        // CPU inference of the concierge model is slow: use the agentbox Loom.
        Tier::Cpu => (Source::Compose, Source::Remote, Source::Compose),
        // Docker on macOS has no GPU: native servers (whisper.cpp, Ollama, LM Studio).
        Tier::Metal => (Source::Local, Source::Local, Source::Remote),
        Tier::Thin => (Source::Remote, Source::Remote, Source::Remote),
    }
}

/// `ws://` URL of the Nemotron forward.
pub fn nemotron_url() -> String {
    format!("ws://127.0.0.1:{NEMOTRON_PORT}/api/asr-streaming")
}

/// Base URL of the Loom forward.
pub fn loom_url() -> String {
    format!("http://127.0.0.1:{LOOM_PORT}/v1")
}

/// Base URL of the Pocket TTS forward.
pub fn pocket_url() -> String {
    format!("http://127.0.0.1:{POCKET_PORT}/v1")
}

/// Request fields that stop the Loom wrapping a non-ontology subject in its scaffold.
pub fn loom_extra() -> Map<String, Value> {
    match json!({"loom_options": {"scaffold": false}}) {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults(t: Tier) -> Choices {
        Choices::defaults_for(t, "laptop")
    }

    #[test]
    fn tier_defaults_follow_the_brief() {
        assert_eq!(
            Tier::RocmLarge.default_whisper(),
            Some("whisper-large-v3-turbo")
        );
        assert_eq!(Tier::Cpu.default_whisper(), Some("whisper-base-en-q8"));
        assert_eq!(Tier::Thin.default_whisper(), None);
        for t in Tier::ALL {
            if let Some(id) = t.default_whisper() {
                assert!(catalogue::find(id).is_some(), "{id}");
            }
            assert_eq!(Tier::from_profile(t.profile()).profile(), t.profile());
        }
    }

    #[test]
    fn tiers_map_to_profiles_and_sources() {
        let gpu = defaults(Tier::Nvidia);
        assert_eq!(gpu.profile, ComposeProfile::Cuda);
        assert_eq!(gpu.stack_slots(), [Slot::Stt, Slot::Llm, Slot::Tts]);
        let cpu = defaults(Tier::Cpu);
        assert_eq!(cpu.profile, ComposeProfile::Cpu);
        assert_eq!(cpu.llm_source, Source::Remote);
        assert_eq!(cpu.stack_slots(), [Slot::Stt, Slot::Tts]);
        let mac = defaults(Tier::Metal);
        assert_eq!(
            (mac.stt_source, mac.llm_source, mac.tts_source),
            (Source::Local, Source::Local, Source::Remote)
        );
        assert!(mac.stack().is_none() && mac.to_config().stack.is_none());
    }

    #[test]
    fn loom_forward_and_url_derivation() {
        let mut c = defaults(Tier::Cpu);
        assert_eq!(c.llm_source, Source::Remote);
        let f = c.forwards();
        assert_eq!(
            f,
            vec![Forward {
                name: "loom".into(),
                local_port: 18084,
                remote: "127.0.0.1:8084".into()
            }]
        );
        let cfg = c.to_config();
        let LlmConfig::OpenAi {
            base_url, extra, ..
        } = &cfg.llm;
        assert_eq!(base_url, "http://127.0.0.1:18084/v1");
        assert_eq!(
            serde_json::Value::Object(extra.clone()),
            json!({"loom_options": {"scaffold": false}})
        );
        c.llm_source = Source::Local;
        assert!(c.forwards().is_empty());
    }

    #[test]
    fn thin_client_uses_three_forwards_and_no_stack() {
        let c = defaults(Tier::Thin);
        let names: Vec<_> = c
            .forwards()
            .into_iter()
            .map(|f| (f.name, f.local_port, f.remote))
            .collect();
        assert_eq!(
            names,
            vec![
                ("nemotron".into(), 18897, "127.0.0.1:8897".into()),
                ("loom".into(), 18084, "127.0.0.1:8084".into()),
                ("pocket".into(), 18898, "127.0.0.1:8898".into()),
            ]
        );
        let cfg = c.to_config();
        assert_eq!(
            cfg.stt,
            SttConfig::Nemotron {
                url: "ws://127.0.0.1:18897/api/asr-streaming".into()
            }
        );
        assert!(c.required_models().is_empty() && cfg.stack.is_none());
        for f in &cfg.connection.forwards {
            assert!(f.local_port > 1024);
        }
    }

    #[test]
    fn compose_slots_need_the_whisper_model_and_write_the_stack() {
        let c = defaults(Tier::Nvidia);
        let ids: Vec<_> = c.required_models().iter().map(|e| e.id).collect();
        assert_eq!(ids, vec!["whisper-medium-q8"]);
        let cfg = c.to_config();
        assert_eq!(
            cfg.stack,
            Some(Stack {
                profile: ComposeProfile::Cuda,
                whisper_file: "ggml-medium-q8_0.bin".into(),
                llm_file: "llm.gguf".into()
            })
        );
        assert!(matches!(&cfg.stt, SttConfig::OpenAi { base_url, .. } if base_url == COMPOSE_STT));
        let TtsConfig::OpenAi {
            base_url, voice, ..
        } = &cfg.tts;
        assert_eq!(
            (base_url.as_str(), voice.as_str()),
            (COMPOSE_TTS, "af_heart")
        );
    }

    #[test]
    fn from_config_round_trips_through_choices() {
        for tier in Tier::ALL {
            let mut c = defaults(tier);
            c.hotkey_enabled = true;
            c.input_device = Some("USB Mic".into());
            let cfg = c.to_config();
            let back = Choices::from_config(&cfg);
            assert_eq!(back.to_config(), cfg, "{tier:?}");
            assert_eq!(back.stt_source, c.stt_source);
            assert_eq!(back.llm_source, c.llm_source);
            assert_eq!(back.tts_source, c.tts_source);
        }
    }

    #[test]
    fn hand_edits_survive_a_rerun() {
        let mut cfg = defaults(Tier::Cpu).to_config();
        cfg.stack.as_mut().unwrap().whisper_file = "my-finetune.bin".into();
        cfg.tts = TtsConfig::OpenAi {
            base_url: COMPOSE_TTS.into(),
            model: "kokoro".into(),
            voice: "bf_emma".into(),
            api_key_env: None,
        };
        cfg.audio.max_utterance_ms = 12_000;
        let c = Choices::from_config(&cfg);
        assert_eq!(c.whisper_override.as_deref(), Some("my-finetune.bin"));
        assert!(
            c.required_models().is_empty(),
            "a custom file is not downloaded"
        );
        assert_eq!(c.to_config(), cfg);
        let mut http = cfg.clone();
        http.stt = SttConfig::OpenAi {
            base_url: "http://127.0.0.1:8080/v1".into(),
            model: "whisper-1".into(),
            api_key_env: Some("STT_KEY".into()),
        };
        let back = Choices::from_config(&http);
        assert_eq!(back.stt_source, Source::Local);
        assert_eq!(back.to_config().stt, http.stt);
    }

    #[test]
    fn utterance_cap_is_enforced() {
        let mut c = defaults(Tier::Cpu);
        assert_eq!(c.to_config().audio.max_utterance_ms, 30_000);
        c.max_utterance_ms = 30_001;
        assert!(!c.validate().is_empty());
        c.max_utterance_ms = 20_000;
        assert!(c.validate().is_empty());
    }

    #[test]
    fn custom_forwards_survive_and_validation_catches_bad_input() {
        let mut cfg = defaults(Tier::Cpu).to_config();
        cfg.connection.forwards.push(Forward {
            name: "extra".into(),
            local_port: 19000,
            remote: "10.0.0.1:1".into(),
        });
        let mut c = Choices::from_config(&cfg);
        assert!(c.forwards().iter().any(|f| f.name == "extra"));
        assert!(c.validate().is_empty());
        c.ssh_destination = "-oProxyCommand=x".into();
        assert!(!c.validate().is_empty());
        c.ssh_destination = "a@b".into();
        c.llm_source = Source::Local;
        c.llm_url = "ftp://x".into();
        assert!(!c.validate().is_empty());
        c.llm_url = LOCAL_LLM.into();
        c.llm_source = Source::Compose;
        c.llm_file = "../etc/passwd".into();
        assert!(!c.validate().is_empty());
    }

    #[test]
    fn remote_tts_uses_a_pocket_voice_not_a_kokoro_one() {
        // Pocket TTS rejects Kokoro voice names (`af_heart` is a 400 there).
        let mut c = defaults(Tier::Thin);
        let voice = |c: &Choices| match c.to_config().tts {
            TtsConfig::OpenAi { voice, .. } => voice,
        };
        assert_eq!(voice(&c), "alba");
        c.tts_source = Source::Compose;
        assert_eq!(voice(&c), "af_heart");
        c.pocket_voice = "jane".into();
        c.tts_source = Source::Remote;
        let back = Choices::from_config(&c.to_config());
        assert_eq!(
            (back.pocket_voice.as_str(), back.tts_voice.as_str()),
            ("jane", "af_heart")
        );
    }

    #[test]
    fn aarch64_cannot_take_speech_to_text_from_the_stack() {
        let mut c = defaults(Tier::SmallGpu);
        assert_eq!(c.apply_arch("aarch64"), [Slot::Stt]);
        assert_eq!(
            (c.stt_source, c.llm_source, c.tts_source),
            (Source::Remote, Source::Compose, Source::Compose)
        );
        let mut x = defaults(Tier::SmallGpu);
        assert!(x.apply_arch("x86_64").is_empty());
        assert_eq!(x.stt_source, Source::Compose);
        let mut r = defaults(Tier::RocmLarge);
        assert_eq!(r.apply_arch("aarch64"), [Slot::Stt, Slot::Llm]);
    }

    #[test]
    fn cycle_wraps() {
        assert_eq!(cycle(Source::ALL, Source::Compose, -1), Source::Remote);
        assert_eq!(cycle(Source::ALL, Source::Remote, 1), Source::Compose);
    }
}
