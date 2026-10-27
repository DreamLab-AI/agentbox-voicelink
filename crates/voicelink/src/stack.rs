//! The bundled Docker compose stack (`compose/compose.yml`): profiles, the
//! exact `docker compose` command to start it, and where to find the file.
//!
//! Nothing here starts a process. [`ComposeCommand`] only describes the
//! command; the setup effect runner is the single place that executes it, and
//! only after the operator confirmed (or passed `--start-stack`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::config::Slot;

/// Default whisper model file of `compose.yml` (`WHISPER_MODEL_FILE`).
pub const DEFAULT_WHISPER_FILE: &str = "ggml-base.en-q8_0.bin";
/// Default GGUF file name of `compose.yml` (`LLM_MODEL_FILE`).
pub const DEFAULT_LLM_FILE: &str = "llm.gguf";
/// Default `video` and `render` group ids baked into `compose.yml`.
const DEFAULT_VIDEO_GID: u32 = 44;
const DEFAULT_RENDER_GID: u32 = 992;

/// A compose profile: one set of images per GPU runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ComposeProfile {
    /// llama.cpp on ROCm; speech on Vulkan (there is no ROCm whisper.cpp image).
    #[value(name = "rocm")]
    Rocm,
    /// NVIDIA CUDA (needs the NVIDIA Container Toolkit).
    #[value(name = "cuda")]
    Cuda,
    /// Vulkan through Mesa (AMD, Intel, and NVIDIA via the open driver).
    #[value(name = "vulkan")]
    Vulkan,
    /// CPU only.
    #[value(name = "cpu")]
    Cpu,
}

impl ComposeProfile {
    /// Every profile, in menu order.
    pub const ALL: [ComposeProfile; 4] = [
        ComposeProfile::Rocm,
        ComposeProfile::Cuda,
        ComposeProfile::Vulkan,
        ComposeProfile::Cpu,
    ];

    /// The profile name as `compose.yml` spells it.
    pub fn name(self) -> &'static str {
        match self {
            ComposeProfile::Rocm => "rocm",
            ComposeProfile::Cuda => "cuda",
            ComposeProfile::Vulkan => "vulkan",
            ComposeProfile::Cpu => "cpu",
        }
    }

    /// The compose service that serves `slot` under this profile.
    pub fn service(self, slot: Slot) -> &'static str {
        match (slot, self) {
            (Slot::Llm, ComposeProfile::Rocm) => "llm-rocm",
            (Slot::Llm, ComposeProfile::Cuda) => "llm-cuda",
            (Slot::Llm, ComposeProfile::Vulkan) => "llm-vulkan",
            (Slot::Llm, ComposeProfile::Cpu) => "llm-cpu",
            (Slot::Stt, ComposeProfile::Cuda) => "stt-cuda",
            // ROCm hosts transcribe through the Vulkan build of whisper.cpp.
            (Slot::Stt, ComposeProfile::Rocm | ComposeProfile::Vulkan) => "stt-vulkan",
            (Slot::Stt, ComposeProfile::Cpu) => "stt-cpu",
            (Slot::Tts, ComposeProfile::Cuda) => "tts-cuda",
            (Slot::Tts, _) => "tts-cpu",
        }
    }
}

/// The `[stack]` table of the config: which compose profile the operator chose
/// and which model files it serves. Present only when a slot uses the stack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stack {
    /// Compose profile.
    pub profile: ComposeProfile,
    /// whisper.cpp model file inside the models directory.
    pub whisper_file: String,
    /// GGUF file inside the models directory that the LLM service loads.
    pub llm_file: String,
}

/// Whether the pinned image behind `service` for `slot` is published for the
/// CPU architecture `arch` (`std::env::consts::ARCH`: `x86_64`, `aarch64`, ...).
///
/// From the image indexes pinned in `compose.yml`: the whisper.cpp images
/// (`main`, `main-vulkan`, `main-cuda`), `server-rocm` and the Kokoro GPU image
/// are `linux/amd64` only; the other llama.cpp images and the Kokoro CPU image
/// also have `linux/arm64`.
pub fn image_available(profile: ComposeProfile, slot: Slot, arch: &str) -> bool {
    let amd64 = arch == "x86_64";
    let arm64 = arch == "aarch64";
    match (slot, profile) {
        (Slot::Stt, _) | (Slot::Llm, ComposeProfile::Rocm) | (Slot::Tts, ComposeProfile::Cuda) => {
            amd64
        }
        _ => amd64 || arm64,
    }
}

/// The slots of `slots` whose stack image does not exist for `arch`.
pub fn unavailable(profile: ComposeProfile, slots: &[Slot], arch: &str) -> Vec<Slot> {
    slots
        .iter()
        .copied()
        .filter(|s| !image_available(profile, *s, arch))
        .collect()
}

/// A `docker compose ... up -d` invocation, described but not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposeCommand {
    /// Environment variables set for the command (only those that differ from the compose defaults).
    pub env: Vec<(String, String)>,
    /// Path of `compose.yml`.
    pub file: PathBuf,
    /// Profile to enable.
    pub profile: ComposeProfile,
    /// Services to start (only the slots that use the stack).
    pub services: Vec<&'static str>,
}

fn shell_word(s: &str) -> String {
    let plain = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:=@%+,".contains(c));
    if plain {
        s.to_string()
    } else {
        crate::connect::sh_quote(s)
    }
}

impl ComposeCommand {
    /// Builds the command for `stack` with the models in `models_dir`.
    ///
    /// `slots` are the slots served by the stack; their services are named
    /// explicitly so an unused service (for example an LLM with no GGUF) is
    /// not started and left crash-looping. `home` and `groups` (the text of
    /// `/etc/group`) only decide which variables differ from the compose
    /// defaults and need to be exported.
    pub fn new(
        stack: &Stack,
        models_dir: &Path,
        slots: &[Slot],
        file: &Path,
        home: Option<&Path>,
        groups: Option<&str>,
    ) -> Self {
        let mut env = Vec::new();
        let default_models = home.map(|h| h.join(".local/share/voicelink/models"));
        if default_models.as_deref() != Some(models_dir) {
            env.push((
                "VOICELINK_MODELS_DIR".to_string(),
                models_dir.to_string_lossy().into_owned(),
            ));
        }
        if slots.contains(&Slot::Stt) && stack.whisper_file != DEFAULT_WHISPER_FILE {
            env.push(("WHISPER_MODEL_FILE".into(), stack.whisper_file.clone()));
        }
        if slots.contains(&Slot::Llm) && stack.llm_file != DEFAULT_LLM_FILE {
            env.push(("LLM_MODEL_FILE".into(), stack.llm_file.clone()));
        }
        if matches!(stack.profile, ComposeProfile::Rocm | ComposeProfile::Vulkan)
            && let Some(g) = groups
        {
            let (video, render) = parse_group_ids(g);
            if let Some(v) = video.filter(|v| *v != DEFAULT_VIDEO_GID) {
                env.push(("VIDEO_GID".into(), v.to_string()));
            }
            if let Some(r) = render.filter(|r| *r != DEFAULT_RENDER_GID) {
                env.push(("RENDER_GID".into(), r.to_string()));
            }
        }
        let services = [Slot::Llm, Slot::Stt, Slot::Tts]
            .into_iter()
            .filter(|s| slots.contains(s))
            .map(|s| stack.profile.service(s))
            .collect();
        ComposeCommand {
            env,
            file: file.to_path_buf(),
            profile: stack.profile,
            services,
        }
    }

    /// The command as the operator would type it.
    pub fn display(&self) -> String {
        let mut parts: Vec<String> = self
            .env
            .iter()
            .map(|(k, v)| format!("{k}={}", shell_word(v)))
            .collect();
        parts.push("docker compose -f".into());
        parts.push(shell_word(&self.file.to_string_lossy()));
        parts.push(format!("--profile {} up -d", self.profile.name()));
        parts.extend(self.services.iter().map(|s| s.to_string()));
        parts.join(" ")
    }

    /// The process to run. Only the setup effect runner calls this.
    pub(crate) fn to_command(&self) -> std::process::Command {
        let mut c = std::process::Command::new("docker");
        c.arg("compose")
            .arg("-f")
            .arg(&self.file)
            .arg("--profile")
            .arg(self.profile.name())
            .arg("up")
            .arg("-d")
            .args(&self.services);
        c.envs(self.env.iter().map(|(k, v)| (k, v)));
        c
    }
}

impl ComposeCommand {
    /// The command for the stack a saved config uses, or `None` when no slot
    /// uses it. `file` is the located `compose.yml`.
    pub fn from_config(cfg: &crate::config::Config, file: &Path) -> Option<Self> {
        let stack = cfg.stack.as_ref()?;
        let slots = stack_slots(cfg);
        (!slots.is_empty()).then(|| {
            ComposeCommand::new(
                stack,
                &cfg.models_dir,
                &slots,
                file,
                std::env::var_os("HOME").map(PathBuf::from).as_deref(),
                system_groups().as_deref(),
            )
        })
    }
}

/// The slots of `cfg` that point at the compose stack's loopback ports.
pub fn stack_slots(cfg: &crate::config::Config) -> Vec<Slot> {
    use crate::setup::choices::{COMPOSE_LLM, COMPOSE_STT, COMPOSE_TTS};
    use voicelink_backends::{LlmConfig, SttConfig, TtsConfig};
    let mut v = Vec::new();
    if matches!(&cfg.stt, SttConfig::OpenAi { base_url, .. } if base_url == COMPOSE_STT) {
        v.push(Slot::Stt);
    }
    let LlmConfig::OpenAi { base_url, .. } = &cfg.llm;
    if base_url == COMPOSE_LLM {
        v.push(Slot::Llm);
    }
    let TtsConfig::OpenAi { base_url, .. } = &cfg.tts;
    if base_url == COMPOSE_TTS {
        v.push(Slot::Tts);
    }
    v
}

impl ComposeCommand {
    /// Runs the command with its output captured and returns whether it
    /// succeeded plus the last lines of output (stdout then stderr), for the
    /// wizard's notes. A first image pull can take minutes.
    pub(crate) fn run_captured(&self) -> (bool, Vec<String>) {
        match self.to_command().output() {
            Ok(o) => {
                let mut lines: Vec<String> = String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .chain(String::from_utf8_lossy(&o.stderr).lines())
                    .map(str::to_owned)
                    .collect();
                let keep = lines.len().saturating_sub(12);
                lines.drain(..keep);
                (o.status.success(), lines)
            }
            Err(e) => (
                false,
                vec![format!("cannot run docker ({e}); is Docker installed?")],
            ),
        }
    }

    /// Runs the command with the terminal's stdio (headless `--start-stack`).
    pub(crate) fn run_inherit(&self) -> std::io::Result<std::process::ExitStatus> {
        self.to_command().status()
    }
}

/// `(video, render)` group ids from the text of `/etc/group`.
pub fn parse_group_ids(text: &str) -> (Option<u32>, Option<u32>) {
    let gid = |name: &str| {
        text.lines().find_map(|l| {
            let mut f = l.split(':');
            (f.next()? == name).then(|| f.nth(1)?.parse().ok())?
        })
    };
    (gid("video"), gid("render"))
}

/// Where `compose.yml` is looked for, best first: the `VOICELINK_COMPOSE`
/// override, a `compose/` directory next to the binary (the release tarball
/// layout), `../share/voicelink/compose/` (a packaged install), then `compose/`
/// under the current directory (a source checkout).
pub fn candidates(
    exe_dir: Option<&Path>,
    cwd: Option<&Path>,
    over: Option<OsString>,
) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(o) = over.filter(|o| !o.is_empty()) {
        v.push(PathBuf::from(o));
    }
    if let Some(d) = exe_dir {
        v.push(d.join("compose/compose.yml"));
        v.push(d.join("../share/voicelink/compose/compose.yml"));
    }
    if let Some(d) = cwd {
        v.push(d.join("compose/compose.yml"));
    }
    v
}

/// The first existing candidate from [`candidates`], for this process.
pub fn locate() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let cwd = std::env::current_dir().ok();
    candidates(
        exe.as_deref().and_then(Path::parent),
        cwd.as_deref(),
        std::env::var_os("VOICELINK_COMPOSE"),
    )
    .into_iter()
    .find(|p| p.is_file())
}

/// Facts about the host that decide the compose command. Kept apart from the
/// wizard state machine so its tests do not depend on the machine they run on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostFacts {
    /// The located `compose.yml`, if any.
    pub compose_file: Option<PathBuf>,
    /// The home directory (decides whether `VOICELINK_MODELS_DIR` is needed).
    pub home: Option<PathBuf>,
    /// The text of `/etc/group` (for the `video` and `render` gids).
    pub groups: Option<String>,
}

impl HostFacts {
    /// Reads the facts from this machine.
    pub fn detect() -> Self {
        HostFacts {
            compose_file: locate(),
            home: std::env::var_os("HOME").map(PathBuf::from),
            groups: system_groups(),
        }
    }
}

/// Reads `/etc/group` where it exists.
pub fn system_groups() -> Option<String> {
    std::fs::read_to_string("/etc/group").ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack(p: ComposeProfile) -> Stack {
        Stack {
            profile: p,
            whisper_file: DEFAULT_WHISPER_FILE.into(),
            llm_file: DEFAULT_LLM_FILE.into(),
        }
    }

    #[test]
    fn default_command_has_no_env_and_names_only_chosen_services() {
        let home = Path::new("/home/me");
        let c = ComposeCommand::new(
            &stack(ComposeProfile::Vulkan),
            Path::new("/home/me/.local/share/voicelink/models"),
            &[Slot::Stt, Slot::Tts],
            Path::new("/opt/vl/compose/compose.yml"),
            Some(home),
            None,
        );
        assert_eq!(
            c.display(),
            "docker compose -f /opt/vl/compose/compose.yml --profile vulkan up -d stt-vulkan tts-cpu"
        );
    }

    #[test]
    fn non_default_models_files_and_gids_are_exported() {
        let mut s = stack(ComposeProfile::Rocm);
        s.whisper_file = "ggml-large-v3-turbo.bin".into();
        s.llm_file = "my model.gguf".into();
        let groups = "root:x:0:\nvideo:x:26:me\nrender:x:303:me\n";
        let c = ComposeCommand::new(
            &s,
            Path::new("/data/models"),
            &[Slot::Llm, Slot::Stt, Slot::Tts],
            Path::new("/c/compose.yml"),
            Some(Path::new("/home/me")),
            Some(groups),
        );
        assert_eq!(
            c.display(),
            "VOICELINK_MODELS_DIR=/data/models WHISPER_MODEL_FILE=ggml-large-v3-turbo.bin \
             LLM_MODEL_FILE='my model.gguf' VIDEO_GID=26 RENDER_GID=303 \
             docker compose -f /c/compose.yml --profile rocm up -d llm-rocm stt-vulkan tts-cpu"
        );
        // CUDA does not mount /dev/dri, so the group ids are irrelevant there.
        let cuda = ComposeCommand::new(
            &stack(ComposeProfile::Cuda),
            Path::new("/data/models"),
            &[Slot::Stt, Slot::Tts],
            Path::new("/c/compose.yml"),
            None,
            Some(groups),
        );
        assert!(cuda.env.iter().all(|(k, _)| k == "VOICELINK_MODELS_DIR"));
        assert_eq!(cuda.services, ["stt-cuda", "tts-cuda"]);
    }

    #[test]
    fn arm64_hosts_have_no_whisper_image() {
        for p in ComposeProfile::ALL {
            assert!(!image_available(p, Slot::Stt, "aarch64"), "{p:?}");
            assert!(image_available(p, Slot::Stt, "x86_64"));
        }
        assert!(image_available(ComposeProfile::Cpu, Slot::Llm, "aarch64"));
        assert!(image_available(
            ComposeProfile::Vulkan,
            Slot::Tts,
            "aarch64"
        ));
        assert!(!image_available(ComposeProfile::Rocm, Slot::Llm, "aarch64"));
        assert!(!image_available(ComposeProfile::Cuda, Slot::Tts, "aarch64"));
        assert_eq!(
            unavailable(
                ComposeProfile::Cpu,
                &[Slot::Stt, Slot::Llm, Slot::Tts],
                "aarch64"
            ),
            [Slot::Stt]
        );
        assert!(unavailable(ComposeProfile::Cpu, &[Slot::Stt], "riscv64").len() == 1);
    }

    #[test]
    fn group_ids_parse() {
        assert_eq!(
            parse_group_ids("video:x:44:\nrender:x:992:a,b\nbroken\n"),
            (Some(44), Some(992))
        );
        assert_eq!(parse_group_ids("video:x:nan:\n"), (None, None));
    }

    #[test]
    fn every_profile_maps_every_slot_to_a_service_in_compose_yml() {
        let yml = include_str!("../../../compose/compose.yml");
        for p in ComposeProfile::ALL {
            for s in [Slot::Llm, Slot::Stt, Slot::Tts] {
                let svc = p.service(s);
                assert!(yml.contains(&format!("\n  {svc}:")), "{svc} missing");
            }
        }
    }

    #[test]
    fn compose_images_are_digest_pinned_and_ports_are_loopback() {
        let yml = include_str!("../../../compose/compose.yml");
        let images: Vec<&str> = yml
            .lines()
            .filter_map(|l| l.trim().strip_prefix("image: "))
            .collect();
        assert_eq!(images.len(), 9);
        for i in images {
            let (name, digest) = i
                .rsplit_once("@sha256:")
                .unwrap_or_else(|| panic!("{i} is not pinned"));
            assert!(name.contains(':'), "keep the tag for readability: {i}");
            assert!(
                digest.len() == 64
                    && digest
                        .chars()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{i}"
            );
        }
        for l in yml.lines().filter(|l| l.trim().starts_with("ports:")) {
            assert!(l.contains("[\"127.0.0.1:"), "{l}");
        }
        assert_eq!(yml.matches("healthcheck: *health-").count(), 9);
    }

    #[test]
    fn compose_file_lookup_order() {
        let c = candidates(
            Some(Path::new("/opt/vl")),
            Some(Path::new("/src")),
            Some("/x/compose.yml".into()),
        );
        assert_eq!(
            c,
            [
                "/x/compose.yml",
                "/opt/vl/compose/compose.yml",
                "/opt/vl/../share/voicelink/compose/compose.yml",
                "/src/compose/compose.yml"
            ]
            .map(PathBuf::from)
        );
        assert!(candidates(None, None, Some("".into())).is_empty());
    }
}
