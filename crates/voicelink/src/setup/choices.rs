//! The wizard's editable model of a configuration, and its projection to and
//! from [`Config`].

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};
use voicelink_backends::{LlmConfig, SttConfig, TtsConfig};

use super::catalogue::{self, Entry};
use crate::config::{Config, Connection, Forward};

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
/// Base URLs of the optional local compose stack (`compose/compose.yml`).
pub const COMPOSE_LLM: &str = "http://127.0.0.1:8180/v1";
/// Compose whisper server.
pub const COMPOSE_STT: &str = "http://127.0.0.1:8181/v1";
/// Compose Kokoro-FastAPI server.
pub const COMPOSE_TTS: &str = "http://127.0.0.1:8182/v1";
/// Where a locally run llama-server listens by default.
pub const LOCAL_LLM: &str = "http://127.0.0.1:8080/v1";

/// Hardware class, deciding the default model sizes.
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
    /// Apple silicon (Metal).
    #[value(name = "metal")]
    Metal,
    /// No usable GPU.
    #[value(name = "cpu")]
    Cpu,
    /// Thin client: speech is remote.
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
            Tier::Metal => "metal (Apple silicon)",
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

    /// Whether the embedded engines should use the GPU build (Vulkan or Metal).
    pub fn gpu(self) -> bool {
        matches!(
            self,
            Tier::RocmLarge | Tier::Nvidia | Tier::SmallGpu | Tier::Metal
        )
    }
}

/// How speech-to-text is provided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SttKind {
    /// whisper.cpp in-process.
    #[value(name = "embedded")]
    Embedded,
    /// An OpenAI-compatible HTTP server.
    #[value(name = "http")]
    Http,
    /// Remote Nemotron over an ssh forward.
    #[value(name = "nemotron")]
    Nemotron,
}

/// Where the concierge LLM runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum LlmKind {
    /// A server already running on this machine (llama-server, Ollama, LM Studio).
    #[value(name = "local")]
    Local,
    /// The optional compose stack.
    #[value(name = "compose")]
    Compose,
    /// The agentbox Ontology Loom over an ssh forward.
    #[value(name = "loom")]
    Loom,
}

/// How text-to-speech is provided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TtsKind {
    /// Kokoro in-process.
    #[value(name = "embedded")]
    Embedded,
    /// An OpenAI-compatible HTTP server.
    #[value(name = "http")]
    Http,
    /// agentbox Pocket TTS over an ssh forward.
    #[value(name = "pocket")]
    Pocket,
}

macro_rules! kind_labels {
    ($t:ty, $($v:path => $l:expr),+) => {
        impl $t {
            /// Every variant in menu order.
            pub const ALL: &'static [$t] = &[$($v),+];
            /// Menu label.
            pub fn label(self) -> &'static str { match self { $($v => $l),+ } }
        }
    };
}
kind_labels!(SttKind, SttKind::Embedded => "embedded whisper.cpp", SttKind::Http => "HTTP (OpenAI-compatible)", SttKind::Nemotron => "remote Nemotron (ssh forward)");
kind_labels!(LlmKind, LlmKind::Local => "local server", LlmKind::Compose => "compose stack", LlmKind::Loom => "agentbox Ontology Loom (ssh forward)");
kind_labels!(TtsKind, TtsKind::Embedded => "embedded Kokoro", TtsKind::Http => "HTTP (OpenAI-compatible)", TtsKind::Pocket => "agentbox Pocket TTS (ssh forward)");

/// Steps through `all` from `current` by `dir` (+1 or -1), wrapping.
pub fn cycle<T: Copy + PartialEq>(all: &[T], current: T, dir: i32) -> T {
    let i = all.iter().position(|x| *x == current).unwrap_or(0) as i32;
    all[(i + dir).rem_euclid(all.len() as i32) as usize]
}

/// Everything the wizard edits.
#[derive(Debug, Clone, PartialEq)]
pub struct Choices {
    /// Embedded engines the defaults were derived for.
    pub engines: Engines,
    /// Hardware tier.
    pub tier: Tier,
    /// ssh destination.
    pub ssh_destination: String,
    /// STT provider.
    pub stt_kind: SttKind,
    /// Catalogue id of the embedded whisper model.
    pub whisper_model: String,
    /// Explicit whisper model path overriding the catalogue (from a hand-edited config).
    pub whisper_path_override: Option<String>,
    /// HTTP STT base URL.
    pub stt_url: String,
    /// HTTP STT model name.
    pub stt_model: String,
    /// LLM provider.
    pub llm_kind: LlmKind,
    /// Local LLM base URL.
    pub llm_url: String,
    /// LLM model name.
    pub llm_model: String,
    /// TTS provider.
    pub tts_kind: TtsKind,
    /// HTTP TTS base URL.
    pub tts_url: String,
    /// HTTP TTS model name.
    pub tts_model: String,
    /// TTS voice (HTTP and Pocket).
    pub tts_voice: String,
    /// Kokoro voice (embedded).
    pub kokoro_voice: String,
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
    /// Where models live.
    pub models_dir: PathBuf,
    /// The configuration being edited; fields the wizard does not own pass through.
    base: Config,
}

/// Embedded engines available in this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Engines {
    /// Embedded whisper.cpp speech-to-text.
    pub whisper: bool,
    /// Embedded Kokoro text-to-speech.
    pub kokoro: bool,
}

impl Engines {
    /// The engines compiled into this binary.
    pub fn compiled() -> Self {
        Engines {
            whisper: voicelink_backends::EMBEDDED_WHISPER,
            kokoro: voicelink_backends::EMBEDDED_KOKORO,
        }
    }
}

impl Choices {
    /// Defaults for `tier` on a machine called `client_host`.
    pub fn defaults_for(tier: Tier, client_host: &str) -> Self {
        Self::defaults_with(Engines::compiled(), tier, client_host)
    }

    /// As [`Choices::defaults_for`], for an explicit set of embedded engines.
    pub fn defaults_with(engines: Engines, tier: Tier, client_host: &str) -> Self {
        let models_dir = crate::paths::models_dir().unwrap_or_else(|_| PathBuf::from("models"));
        let thin = tier == Tier::Thin;
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
        };
        Choices {
            engines,
            tier,
            ssh_destination: DEFAULT_SSH.into(),
            // Without an embedded engine compiled in, fall back to the agentbox
            // speech plane, which every agentbox host has.
            stt_kind: if thin || !engines.whisper {
                SttKind::Nemotron
            } else {
                SttKind::Embedded
            },
            whisper_model: tier
                .default_whisper()
                .unwrap_or("whisper-base-en-q8")
                .into(),
            whisper_path_override: None,
            stt_url: COMPOSE_STT.into(),
            stt_model: "whisper-1".into(),
            llm_kind: if matches!(tier, Tier::Cpu | Tier::Thin) {
                LlmKind::Loom
            } else {
                LlmKind::Local
            },
            llm_url: LOCAL_LLM.into(),
            llm_model: "default".into(),
            tts_kind: if thin || !engines.kokoro {
                TtsKind::Pocket
            } else {
                TtsKind::Embedded
            },
            tts_url: COMPOSE_TTS.into(),
            tts_model: "kokoro".into(),
            tts_voice: "af_heart".into(),
            kokoro_voice: "af_heart".into(),
            input_device: None,
            output_device: None,
            hotkey_enabled: false,
            hotkey_accel: "Ctrl+Alt+M".into(),
            max_utterance_ms: MAX_UTTERANCE_CAP_MS,
            models_dir,
            base,
        }
    }

    /// Resets the tier-dependent choices (speech providers, whisper size, LLM
    /// provider) to the defaults of `tier`, keeping everything else.
    pub fn apply_tier(&mut self, tier: Tier, client_host: &str) {
        let d = Choices::defaults_with(self.engines, tier, client_host);
        self.tier = tier;
        self.stt_kind = d.stt_kind;
        self.whisper_model = d.whisper_model;
        self.whisper_path_override = None;
        self.tts_kind = d.tts_kind;
        self.llm_kind = d.llm_kind;
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
            SttConfig::Whisper {
                model_path, gpu, ..
            } => {
                c.stt_kind = SttKind::Embedded;
                let file = Path::new(model_path)
                    .file_name()
                    .and_then(|f| f.to_str())
                    .unwrap_or_default();
                match catalogue::find_by_file(file) {
                    Some(e) => c.whisper_model = e.id.into(),
                    None => c.whisper_path_override = Some(model_path.clone()),
                }
                c.tier = if *gpu { Tier::SmallGpu } else { Tier::Cpu };
            }
            SttConfig::OpenAi {
                base_url, model, ..
            } => {
                c.stt_kind = SttKind::Http;
                c.stt_url = base_url.clone();
                c.stt_model = model.clone();
            }
            SttConfig::Nemotron { .. } => {
                c.stt_kind = SttKind::Nemotron;
                c.tier = Tier::Thin;
            }
        }
        #[allow(irrefutable_let_patterns)]
        if let LlmConfig::OpenAi {
            base_url, model, ..
        } = &cfg.llm
        {
            c.llm_model = model.clone();
            c.llm_url = base_url.clone();
            c.llm_kind = if *base_url == loom_url() {
                LlmKind::Loom
            } else if base_url == COMPOSE_LLM {
                LlmKind::Compose
            } else {
                LlmKind::Local
            };
        }
        match &cfg.tts {
            TtsConfig::Kokoro { voice, .. } => {
                c.tts_kind = TtsKind::Embedded;
                c.kokoro_voice = voice.clone();
            }
            TtsConfig::OpenAi {
                base_url,
                model,
                voice,
                ..
            } => {
                c.tts_url = base_url.clone();
                c.tts_model = model.clone();
                c.tts_voice = voice.clone();
                c.tts_kind = if *base_url == pocket_url() {
                    TtsKind::Pocket
                } else {
                    TtsKind::Http
                };
            }
        }
        c.base = cfg.clone();
        c
    }

    /// The whisper catalogue entry, when not overridden.
    pub fn whisper_entry(&self) -> Option<&'static Entry> {
        catalogue::find(&self.whisper_model)
    }

    /// The forwards the choices need (named `loom`, `pocket`, `nemotron`),
    /// followed by any custom forwards carried over from the previous config.
    pub fn forwards(&self) -> Vec<Forward> {
        let mut out = Vec::new();
        if self.stt_kind == SttKind::Nemotron {
            out.push(Forward {
                name: "nemotron".into(),
                local_port: NEMOTRON_PORT,
                remote: "127.0.0.1:8897".into(),
            });
        }
        if self.llm_kind == LlmKind::Loom {
            out.push(Forward {
                name: "loom".into(),
                local_port: LOOM_PORT,
                remote: "127.0.0.1:8084".into(),
            });
        }
        if self.tts_kind == TtsKind::Pocket {
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
        if self.stt_kind == SttKind::Embedded
            && self.whisper_path_override.is_none()
            && let Some(e) = self.whisper_entry()
        {
            v.push(e);
        }
        if self.tts_kind == TtsKind::Embedded {
            v.extend(
                ["kokoro-model", "kokoro-voices"]
                    .iter()
                    .filter_map(|id| catalogue::find(id)),
            );
        }
        v
    }

    fn threads(&self) -> u16 {
        if let SttConfig::Whisper { threads, .. } = &self.base.stt {
            return *threads;
        }
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        (n / 2).clamp(2, 8) as u16
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
        cfg.stt = match self.stt_kind {
            SttKind::Embedded => {
                let entry = self.whisper_entry();
                let path = self.whisper_path_override.clone().unwrap_or_else(|| {
                    self.models_dir
                        .join(entry.map_or("ggml-base.en-q8_0.bin", |e| e.file))
                        .to_string_lossy()
                        .into_owned()
                });
                let english_only = Path::new(&path)
                    .file_name()
                    .and_then(|f| f.to_str())
                    .is_some_and(|f| f.contains(".en"));
                // Keep a hand-edited language while the model file is unchanged.
                let language = match &self.base.stt {
                    SttConfig::Whisper {
                        model_path,
                        language,
                        ..
                    } if *model_path == path => language.clone(),
                    _ => english_only.then(|| "en".to_string()),
                };
                SttConfig::Whisper {
                    model_path: path,
                    language,
                    threads: self.threads(),
                    gpu: self.tier.gpu(),
                }
            }
            SttKind::Http => SttConfig::OpenAi {
                base_url: self.stt_url.clone(),
                model: self.stt_model.clone(),
                api_key_env: match &self.base.stt {
                    SttConfig::OpenAi { api_key_env, .. } => api_key_env.clone(),
                    _ => None,
                },
            },
            SttKind::Nemotron => SttConfig::Nemotron {
                url: nemotron_url(),
            },
        };
        let (base_url, extra) = match self.llm_kind {
            LlmKind::Local => (self.llm_url.clone(), Map::new()),
            LlmKind::Compose => (COMPOSE_LLM.to_string(), Map::new()),
            LlmKind::Loom => (loom_url(), loom_extra()),
        };
        let api_key_env = match &self.base.llm {
            LlmConfig::OpenAi { api_key_env, .. } => api_key_env.clone(),
        };
        cfg.llm = LlmConfig::OpenAi {
            base_url,
            model: self.llm_model.clone(),
            api_key_env,
            extra,
        };
        let base_tts_key = match &self.base.tts {
            TtsConfig::OpenAi { api_key_env, .. } => api_key_env.clone(),
            _ => None,
        };
        cfg.tts = match self.tts_kind {
            TtsKind::Embedded => TtsConfig::Kokoro {
                model_path: self
                    .models_dir
                    .join(catalogue::find("kokoro-model").map_or("", |e| e.file))
                    .to_string_lossy()
                    .into_owned(),
                voices_path: self
                    .models_dir
                    .join(catalogue::find("kokoro-voices").map_or("", |e| e.file))
                    .to_string_lossy()
                    .into_owned(),
                voice: self.kokoro_voice.clone(),
            },
            TtsKind::Http => TtsConfig::OpenAi {
                base_url: self.tts_url.clone(),
                model: self.tts_model.clone(),
                voice: self.tts_voice.clone(),
                api_key_env: base_tts_key.clone(),
            },
            TtsKind::Pocket => TtsConfig::OpenAi {
                base_url: pocket_url(),
                model: "pocket-tts".into(),
                voice: self.tts_voice.clone(),
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
        if self.stt_kind == SttKind::Http && !http(&self.stt_url) {
            p.push("STT URL must start with http:// or https://".into());
        }
        if self.llm_kind == LlmKind::Local && !http(&self.llm_url) {
            p.push("LLM URL must start with http:// or https://".into());
        }
        if self.tts_kind == TtsKind::Http && !http(&self.tts_url) {
            p.push("TTS URL must start with http:// or https://".into());
        }
        if self.llm_model.trim().is_empty() {
            p.push("LLM model name is empty".into());
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
        }
    }

    #[test]
    fn loom_forward_and_url_derivation() {
        let mut c = Choices::defaults_with(
            crate::setup::choices::Engines {
                whisper: true,
                kokoro: true,
            },
            Tier::Cpu,
            "laptop",
        );
        assert_eq!(c.llm_kind, LlmKind::Loom);
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
        c.llm_kind = LlmKind::Local;
        assert!(c.forwards().is_empty());
    }

    #[test]
    fn thin_client_uses_three_forwards() {
        let c = Choices::defaults_with(
            crate::setup::choices::Engines {
                whisper: true,
                kokoro: true,
            },
            Tier::Thin,
            "laptop",
        );
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
        assert!(c.required_models().is_empty());
        for f in &cfg.connection.forwards {
            assert!(f.local_port > 1024);
        }
    }

    #[test]
    fn embedded_tiers_need_models() {
        let c = Choices::defaults_with(
            crate::setup::choices::Engines {
                whisper: true,
                kokoro: true,
            },
            Tier::Nvidia,
            "x",
        );
        let ids: Vec<_> = c.required_models().iter().map(|e| e.id).collect();
        assert_eq!(
            ids,
            vec!["whisper-medium-q8", "kokoro-model", "kokoro-voices"]
        );
        let SttConfig::Whisper {
            model_path,
            gpu,
            language,
            ..
        } = c.to_config().stt
        else {
            panic!()
        };
        assert!(model_path.ends_with("ggml-medium-q8_0.bin"));
        assert!(gpu);
        assert_eq!(language, None);
        let SttConfig::Whisper { language, gpu, .. } = Choices::defaults_with(
            crate::setup::choices::Engines {
                whisper: true,
                kokoro: true,
            },
            Tier::Cpu,
            "x",
        )
        .to_config()
        .stt
        else {
            panic!()
        };
        assert_eq!(language.as_deref(), Some("en"));
        assert!(!gpu);
    }

    #[test]
    fn from_config_round_trips_through_choices() {
        for tier in Tier::ALL {
            let mut c = Choices::defaults_with(
                crate::setup::choices::Engines {
                    whisper: true,
                    kokoro: true,
                },
                tier,
                "host",
            );
            c.hotkey_enabled = true;
            c.input_device = Some("USB Mic".into());
            let cfg = c.to_config();
            let back = Choices::from_config(&cfg);
            assert_eq!(back.to_config(), cfg, "{tier:?}");
            assert_eq!(back.stt_kind, c.stt_kind);
            assert_eq!(back.llm_kind, c.llm_kind);
            assert_eq!(back.tts_kind, c.tts_kind);
        }
    }

    #[test]
    fn hand_edits_survive_a_rerun() {
        let mut cfg = Choices::defaults_with(
            crate::setup::choices::Engines {
                whisper: true,
                kokoro: true,
            },
            Tier::Cpu,
            "h",
        )
        .to_config();
        let model = catalogue::find("whisper-small-q8").unwrap().file;
        let path = cfg.models_dir.join(model).to_string_lossy().into_owned();
        cfg.stt = SttConfig::Whisper {
            model_path: path,
            language: Some("de".into()),
            threads: 3,
            gpu: false,
        };
        cfg.tts = TtsConfig::OpenAi {
            base_url: COMPOSE_TTS.into(),
            model: "kokoro".into(),
            voice: "bf_emma".into(),
            api_key_env: Some("TTS_KEY".into()),
        };
        cfg.audio.max_utterance_ms = 12_000;
        let c = Choices::from_config(&cfg);
        assert_eq!(c.to_config(), cfg);
        let mut http = cfg.clone();
        http.stt = SttConfig::OpenAi {
            base_url: COMPOSE_STT.into(),
            model: "whisper-1".into(),
            api_key_env: Some("STT_KEY".into()),
        };
        assert_eq!(Choices::from_config(&http).to_config(), http);
    }

    #[test]
    fn utterance_cap_is_enforced() {
        let mut c = Choices::defaults_with(
            crate::setup::choices::Engines {
                whisper: true,
                kokoro: true,
            },
            Tier::Cpu,
            "h",
        );
        assert_eq!(c.to_config().audio.max_utterance_ms, 30_000);
        c.max_utterance_ms = 30_001;
        assert!(!c.validate().is_empty());
        c.max_utterance_ms = 20_000;
        assert!(c.validate().is_empty());
    }

    #[test]
    fn unknown_whisper_path_is_preserved() {
        let mut cfg = Choices::defaults_with(
            crate::setup::choices::Engines {
                whisper: true,
                kokoro: true,
            },
            Tier::Cpu,
            "h",
        )
        .to_config();
        cfg.stt = SttConfig::Whisper {
            model_path: "/opt/custom/model.bin".into(),
            language: None,
            threads: 3,
            gpu: false,
        };
        let c = Choices::from_config(&cfg);
        assert!(
            c.required_models()
                .iter()
                .all(|e| e.kind != catalogue::Kind::Whisper)
        );
        let SttConfig::Whisper { model_path, .. } = c.to_config().stt else {
            panic!()
        };
        assert_eq!(model_path, "/opt/custom/model.bin");
    }

    #[test]
    fn custom_forwards_survive_and_validation_catches_bad_input() {
        let mut cfg = Choices::defaults_with(
            crate::setup::choices::Engines {
                whisper: true,
                kokoro: true,
            },
            Tier::Cpu,
            "h",
        )
        .to_config();
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
        c.llm_kind = LlmKind::Local;
        c.llm_url = "ftp://x".into();
        assert!(!c.validate().is_empty());
    }

    #[test]
    fn cycle_wraps() {
        assert_eq!(
            cycle(SttKind::ALL, SttKind::Embedded, -1),
            SttKind::Nemotron
        );
        assert_eq!(cycle(SttKind::ALL, SttKind::Nemotron, 1), SttKind::Embedded);
    }

    #[test]
    fn builds_without_embedded_engines_default_to_the_agentbox_speech_plane() {
        let none = Engines {
            whisper: false,
            kokoro: false,
        };
        let c = Choices::defaults_with(none, Tier::RocmLarge, "laptop");
        assert_eq!(c.stt_kind, SttKind::Nemotron);
        assert_eq!(c.tts_kind, TtsKind::Pocket);
        assert!(c.required_models().is_empty());
    }
}
