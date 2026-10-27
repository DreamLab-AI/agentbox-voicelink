//! The wizard as a pure state machine: keys and effect results go in, a list
//! of effects to run comes out. No terminal, network or subprocess is touched
//! here, so every transition is unit-tested.

use std::path::PathBuf;

use serde_json::{Map, Value};

use super::catalogue::{self, Entry};
use super::choices::{Choices, Source, Tier, cycle};
use super::net::{DetectedServer, ToolProbe};
use super::probe::{HardwareReport, Proposal};
use crate::audio::DeviceList;
use crate::config::{self, Config, Slot};
use crate::stack::{ComposeCommand, HostFacts};

/// Wizard pages, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Step {
    /// Hardware probe and tier.
    Hardware,
    /// Speech-to-text.
    Speech,
    /// Concierge LLM.
    Llm,
    /// Text-to-speech.
    Voice,
    /// Model downloads and the compose stack.
    Models,
    /// Audio devices.
    Audio,
    /// Mute hotkey.
    Hotkey,
    /// ssh connection.
    Connection,
    /// Smoke test.
    Smoke,
    /// Review and write.
    Review,
}

impl Step {
    /// Every step in order.
    pub const ALL: [Step; 10] = [
        Step::Hardware,
        Step::Speech,
        Step::Llm,
        Step::Voice,
        Step::Models,
        Step::Audio,
        Step::Hotkey,
        Step::Connection,
        Step::Smoke,
        Step::Review,
    ];

    /// Page title.
    pub fn title(self) -> &'static str {
        match self {
            Step::Hardware => "Hardware",
            Step::Speech => "Speech to text",
            Step::Llm => "Concierge model",
            Step::Voice => "Text to speech",
            Step::Models => "Models and stack",
            Step::Audio => "Audio devices",
            Step::Hotkey => "Mute hotkey",
            Step::Connection => "Connection",
            Step::Smoke => "Smoke test",
            Step::Review => "Review",
        }
    }

    fn index(self) -> usize {
        Step::ALL.iter().position(|s| *s == self).unwrap_or(0)
    }
}

/// Keys the wizard understands (terminal-independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    /// Up arrow.
    Up,
    /// Down arrow.
    Down,
    /// Left arrow.
    Left,
    /// Right arrow.
    Right,
    /// Enter.
    Enter,
    /// Escape.
    Esc,
    /// Backspace.
    Backspace,
    /// Tab.
    Tab,
    /// Shift-Tab.
    BackTab,
    /// Ctrl-C.
    CtrlC,
    /// A printable character.
    Char(char),
}

/// Work the runner performs on the wizard's behalf.
#[derive(Debug, Clone)]
pub enum Effect {
    /// Probe the hardware.
    Probe,
    /// Look for local model servers (llama-server, Ollama, LM Studio, whisper-server, Kokoro-FastAPI).
    Detect,
    /// Tool-calling probe.
    ToolProbe {
        /// Server base URL.
        base_url: String,
        /// Model name.
        model: String,
        /// Extra request fields.
        extra: Map<String, Value>,
    },
    /// List audio devices.
    ListDevices,
    /// Start the live level meter on the configured input.
    StartMeter(config::Audio),
    /// Stop the level meter.
    StopMeter,
    /// Record three seconds and play them back.
    RecordPlayback(config::Audio),
    /// Download catalogue entries into a directory.
    Download(Vec<&'static Entry>, PathBuf),
    /// Run `docker compose ... up -d` (only ever after an explicit confirmation).
    StartStack(Box<ComposeCommand>),
    /// `voice-paned --version` over ssh.
    CheckSsh(config::Connection),
    /// Hello/Welcome over ssh.
    Handshake(config::Connection),
    /// STT and TTS round trip with the candidate configuration.
    Smoke(Box<Config>),
    /// Write the configuration.
    Save(Box<Config>),
}

/// Results fed back by the runner.
#[derive(Debug, Clone)]
pub enum Outcome {
    /// Probe finished.
    Hardware(HardwareReport),
    /// Local servers found.
    Detected(Vec<DetectedServer>),
    /// Tool probe finished.
    Tool(ToolProbe),
    /// Device list (or why there is none).
    Devices(Result<DeviceList, String>),
    /// Input level in dBFS.
    Level(f32),
    /// The meter ended (with an error, if it failed).
    MeterStopped(Option<String>),
    /// Download progress: file, bytes done, bytes total.
    Progress(String, u64, u64),
    /// A finished action with report lines.
    Lines {
        /// Which action (`"ssh"`, `"handshake"`, `"download"`, `"stack"`, `"record"`, `"smoke"`).
        what: &'static str,
        /// Whether it succeeded.
        ok: bool,
        /// Report.
        lines: Vec<String>,
    },
    /// The configuration was written (or why not).
    Saved(Result<PathBuf, String>),
}

/// How the wizard ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    /// The config was written to this path.
    Saved(PathBuf),
    /// The operator quit.
    Cancelled,
}

/// What a row does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// Left/Right (or Enter) cycles the value.
    Choice,
    /// Enter edits the text.
    Text,
    /// Space or Enter flips it.
    Toggle,
    /// Enter runs it.
    Action,
}

/// Identifies a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum FieldId {
    TierSel,
    Reprobe,
    SttSourceSel,
    WhisperModel,
    SttUrl,
    SttModel,
    LlmSourceSel,
    DetectedSel,
    Redetect,
    LlmUrl,
    LlmModel,
    LlmFile,
    ToolProbeRun,
    TtsSourceSel,
    TtsUrl,
    TtsModel,
    TtsVoice,
    PocketVoice,
    ProfileSel,
    Download,
    StartStack,
    InputDevice,
    OutputDevice,
    Meter,
    RecordPlay,
    MaxUtterance,
    HotkeyEnabled,
    HotkeyAccel,
    SshDest,
    CheckSsh,
    Handshake,
    Smoke,
    Save,
}

/// One row of the current page.
#[derive(Debug, Clone)]
pub struct Field {
    /// Row id.
    pub id: FieldId,
    /// Label.
    pub label: String,
    /// Current value text.
    pub value: String,
    /// Behaviour.
    pub kind: FieldKind,
}

fn slot_index(slot: Slot) -> usize {
    match slot {
        Slot::Stt => 0,
        Slot::Llm => 1,
        Slot::Tts => 2,
    }
}

fn field(id: FieldId, label: &str, value: impl Into<String>, kind: FieldKind) -> Field {
    Field {
        id,
        label: label.into(),
        value: value.into(),
        kind,
    }
}

/// The wizard state.
#[derive(Debug)]
pub struct Wizard {
    /// Current page.
    pub step: Step,
    /// The configuration being built.
    pub choices: Choices,
    /// Selected row.
    pub cursor: usize,
    /// Text being edited, if any.
    pub editing: Option<String>,
    /// Hardware report.
    pub hw: Option<HardwareReport>,
    /// Tier proposal and evidence.
    pub proposal: Option<Proposal>,
    /// Local servers found, for every slot.
    pub detected: Vec<DetectedServer>,
    /// Detection has completed.
    pub detect_done: bool,
    /// Selected server per slot (index into that slot's detected servers).
    pub detected_idx: [usize; 3],
    /// Facts about the host (compose file, home, groups).
    pub facts: HostFacts,
    /// The start-stack row has been pressed once and awaits confirmation.
    pub stack_confirm: bool,
    /// Audio devices.
    pub devices: Option<Result<DeviceList, String>>,
    /// Latest input level.
    pub level_db: Option<f32>,
    /// The meter is running.
    pub meter_on: bool,
    /// Download progress.
    pub progress: Option<(String, u64, u64)>,
    /// Report lines of the current page.
    pub notes: Vec<String>,
    /// Actions in flight.
    pub busy: Vec<&'static str>,
    /// Actions that finished successfully.
    pub passed: Vec<&'static str>,
    /// Set when the wizard is over.
    pub finished: Option<Finish>,
    /// Whether an earlier configuration pre-filled the choices.
    pub prefilled: bool,
    /// Tool-probe verdict for the current LLM settings.
    pub tool: Option<ToolProbe>,
    host: String,
}

impl Wizard {
    /// Starts a wizard. `existing` pre-fills the choices (re-run).
    pub fn new(existing: Option<&Config>, host: &str) -> Self {
        let (choices, prefilled) = match existing {
            Some(c) => (Choices::from_config(c), true),
            None => (Choices::defaults_for(Tier::Cpu, host), false),
        };
        Wizard {
            step: Step::Hardware,
            choices,
            cursor: 0,
            editing: None,
            hw: None,
            proposal: None,
            detected: Vec::new(),
            detect_done: false,
            detected_idx: [0; 3],
            facts: HostFacts::default(),
            stack_confirm: false,
            devices: None,
            level_db: None,
            meter_on: false,
            progress: None,
            notes: Vec::new(),
            busy: Vec::new(),
            passed: Vec::new(),
            finished: None,
            prefilled,
            tool: None,
            host: host.to_string(),
        }
    }

    /// Effects to run at start-up.
    pub fn start(&mut self) -> Vec<Effect> {
        self.enter_effects()
    }

    fn is_busy(&self, what: &'static str) -> bool {
        self.busy.contains(&what)
    }

    fn mark_busy(&mut self, what: &'static str) {
        if !self.busy.contains(&what) {
            self.busy.push(what);
        }
    }

    fn clear_busy(&mut self, what: &str) {
        self.busy.retain(|b| *b != what);
    }

    fn enter_effects(&mut self) -> Vec<Effect> {
        let mut e = Vec::new();
        match self.step {
            Step::Hardware if self.hw.is_none() && !self.is_busy("probe") => {
                self.mark_busy("probe");
                e.push(Effect::Probe);
            }
            Step::Speech | Step::Llm | Step::Voice
                if !self.detect_done && !self.is_busy("detect") =>
            {
                self.mark_busy("detect");
                e.push(Effect::Detect);
            }
            Step::Audio if self.devices.is_none() && !self.is_busy("devices") => {
                self.mark_busy("devices");
                e.push(Effect::ListDevices);
            }
            _ => {}
        }
        e
    }

    /// Switches page, stopping the level meter when leaving the audio page.
    pub fn goto(&mut self, step: Step) -> Vec<Effect> {
        let mut e = Vec::new();
        if self.meter_on {
            e.push(Effect::StopMeter);
        }
        self.step = step;
        self.cursor = 0;
        self.editing = None;
        self.stack_confirm = false;
        self.notes.clear();
        e.extend(self.enter_effects());
        e
    }

    fn apply_tier(&mut self, tier: Tier) {
        let host = self.host.clone();
        self.choices.apply_tier(tier, &host);
    }

    /// The rows of the current page.
    pub fn fields(&self) -> Vec<Field> {
        use FieldId::*;
        use FieldKind::*;
        let c = &self.choices;
        let mut f = Vec::new();
        let done = |k: &str| {
            if self.passed.contains(&k) {
                "passed"
            } else {
                "not run"
            }
        };
        match self.step {
            Step::Hardware => {
                f.push(field(TierSel, "Tier", c.tier.label(), Choice));
                f.push(field(
                    Reprobe,
                    "Probe again",
                    if self.is_busy("probe") {
                        "probing..."
                    } else {
                        ""
                    },
                    Action,
                ));
            }
            Step::Speech => {
                f.push(field(
                    SttSourceSel,
                    "Source",
                    c.stt_source.label(Slot::Stt),
                    Choice,
                ));
                match c.stt_source {
                    Source::Compose => {
                        let label = match (&c.whisper_override, c.whisper_entry()) {
                            (Some(p), _) => format!("custom file {p}"),
                            (None, Some(e)) => e.label.to_string(),
                            _ => c.whisper_model.clone(),
                        };
                        f.push(field(WhisperModel, "Whisper model", label, Choice));
                    }
                    Source::Local => {
                        self.push_detected(&mut f, Slot::Stt);
                        f.push(field(SttUrl, "Base URL", &c.stt_url, Text));
                        f.push(field(SttModel, "Model name", &c.stt_model, Text));
                    }
                    Source::Remote => {}
                }
            }
            Step::Llm => {
                f.push(field(
                    LlmSourceSel,
                    "Source",
                    c.llm_source.label(Slot::Llm),
                    Choice,
                ));
                match c.llm_source {
                    Source::Compose => f.push(field(
                        LlmFile,
                        "GGUF file (in the models dir)",
                        &c.llm_file,
                        Text,
                    )),
                    Source::Local => {
                        self.push_detected(&mut f, Slot::Llm);
                        f.push(field(LlmUrl, "Base URL", &c.llm_url, Text));
                    }
                    Source::Remote => {}
                }
                f.push(field(LlmModel, "Model name", &c.llm_model, Text));
                f.push(field(
                    ToolProbeRun,
                    "Tool-calling probe",
                    if self.is_busy("tool") {
                        "running..."
                    } else {
                        done("tool")
                    },
                    Action,
                ));
            }
            Step::Voice => {
                f.push(field(
                    TtsSourceSel,
                    "Source",
                    c.tts_source.label(Slot::Tts),
                    Choice,
                ));
                if c.tts_source == Source::Local {
                    self.push_detected(&mut f, Slot::Tts);
                    f.push(field(TtsUrl, "Base URL", &c.tts_url, Text));
                    f.push(field(TtsModel, "Model name", &c.tts_model, Text));
                }
                if c.tts_source == Source::Remote {
                    f.push(field(PocketVoice, "Pocket voice", &c.pocket_voice, Text));
                } else {
                    f.push(field(TtsVoice, "Voice", &c.tts_voice, Text));
                }
            }
            Step::Models => {
                if !c.stack_slots().is_empty() {
                    f.push(field(
                        ProfileSel,
                        "Compose profile",
                        c.profile.name(),
                        Choice,
                    ));
                }
                if !c.required_models().is_empty() {
                    f.push(field(
                        Download,
                        "Download missing models",
                        if self.is_busy("download") {
                            "downloading..."
                        } else {
                            done("download")
                        },
                        Action,
                    ));
                }
                if !c.stack_slots().is_empty() {
                    f.push(field(
                        StartStack,
                        "Start the compose stack",
                        if self.is_busy("stack") {
                            "starting..."
                        } else if self.facts.compose_file.is_none() {
                            "compose files not found"
                        } else if self.stack_confirm {
                            "press Enter again to run it"
                        } else {
                            done("stack")
                        },
                        Action,
                    ));
                }
            }
            Step::Audio => {
                f.push(field(
                    InputDevice,
                    "Input device",
                    c.input_device
                        .clone()
                        .unwrap_or_else(|| "system default".into()),
                    Choice,
                ));
                f.push(field(
                    OutputDevice,
                    "Output device",
                    c.output_device
                        .clone()
                        .unwrap_or_else(|| "system default".into()),
                    Choice,
                ));
                f.push(field(
                    Meter,
                    "Level meter",
                    if self.meter_on {
                        "on (Enter to stop)"
                    } else {
                        "off (Enter to start)"
                    },
                    Action,
                ));
                f.push(field(
                    MaxUtterance,
                    "Longest utterance (ms, max 30000)",
                    c.max_utterance_ms.to_string(),
                    Text,
                ));
                f.push(field(
                    RecordPlay,
                    "Record 3 s and play back",
                    if self.is_busy("record") {
                        "recording..."
                    } else {
                        done("record")
                    },
                    Action,
                ));
            }
            Step::Hotkey => {
                f.push(field(
                    HotkeyEnabled,
                    "Global mute hotkey",
                    if c.hotkey_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    },
                    Toggle,
                ));
                if c.hotkey_enabled {
                    f.push(field(HotkeyAccel, "Accelerator", &c.hotkey_accel, Text));
                }
            }
            Step::Connection => {
                f.push(field(SshDest, "ssh destination", &c.ssh_destination, Text));
                f.push(field(
                    CheckSsh,
                    "Check voice-paned",
                    if self.is_busy("ssh") {
                        "running..."
                    } else {
                        done("ssh")
                    },
                    Action,
                ));
                f.push(field(
                    Handshake,
                    "Protocol handshake",
                    if self.is_busy("handshake") {
                        "running..."
                    } else {
                        done("handshake")
                    },
                    Action,
                ));
            }
            Step::Smoke => f.push(field(
                Smoke,
                "Run smoke test",
                if self.is_busy("smoke") {
                    "speak now..."
                } else {
                    done("smoke")
                },
                Action,
            )),
            Step::Review => f.push(field(Save, "Write config", "", Action)),
        }
        f
    }

    fn options_for_whisper() -> Vec<&'static str> {
        catalogue::CATALOGUE.iter().map(|e| e.id).collect()
    }

    /// Servers detected for `slot`.
    pub fn detected_for(&self, slot: Slot) -> Vec<&DetectedServer> {
        self.detected.iter().filter(|d| d.slot == slot).collect()
    }

    fn push_detected(&self, f: &mut Vec<Field>, slot: Slot) {
        let found = self.detected_for(slot);
        let shown = match found.get(self.detected_idx[slot_index(slot)]) {
            Some(d) => format!("{} at {}", d.label, d.base_url),
            None if self.detect_done => "none found".to_string(),
            None => "looking...".to_string(),
        };
        f.push(field(
            FieldId::DetectedSel,
            "Detected server",
            shown,
            FieldKind::Choice,
        ));
        f.push(field(
            FieldId::Redetect,
            "Detect again",
            "",
            FieldKind::Action,
        ));
    }

    /// The compose command the current choices describe, if the stack is used
    /// and `compose.yml` was found.
    pub fn stack_command(&self) -> Option<ComposeCommand> {
        let stack = self.choices.stack()?;
        let file = self.facts.compose_file.as_ref()?;
        Some(ComposeCommand::new(
            &stack,
            &self.choices.models_dir,
            &self.choices.stack_slots(),
            file,
            self.facts.home.as_deref(),
            self.facts.groups.as_deref(),
        ))
    }

    /// Reasons the stack command is likely to misbehave, shown before the
    /// operator confirms it.
    pub fn stack_warnings(&self) -> Vec<String> {
        let c = &self.choices;
        let arch = self
            .hw
            .as_ref()
            .map_or(std::env::consts::ARCH, |h| h.arch.as_str());
        let mut v = Vec::new();
        for s in crate::stack::unavailable(c.profile, &c.stack_slots(), arch) {
            v.push(format!(
                "Warning: there is no {arch} image for {}; docker will fail to pull it. Pick another source for the {}.",
                c.profile.service(s),
                s.label()
            ));
        }
        if c.llm_source == Source::Compose && !c.models_dir.join(&c.llm_file).is_file() {
            v.push(format!(
                "Warning: {} does not exist, so {} would restart in a loop. Put a tool-calling GGUF there first, or leave the concierge model on another source.",
                c.models_dir.join(&c.llm_file).display(),
                c.profile.service(Slot::Llm)
            ));
        }
        v
    }

    fn page_slot(&self) -> Option<Slot> {
        match self.step {
            Step::Speech => Some(Slot::Stt),
            Step::Llm => Some(Slot::Llm),
            Step::Voice => Some(Slot::Tts),
            _ => None,
        }
    }

    fn cycle_field(&mut self, id: FieldId, dir: i32) {
        use FieldId::*;
        match id {
            TierSel => {
                let t = cycle(&super::choices::Tier::ALL, self.choices.tier, dir);
                self.apply_tier(t);
            }
            SttSourceSel => {
                self.choices.stt_source = cycle(Source::ALL, self.choices.stt_source, dir);
                self.passed.retain(|p| *p != "stack");
            }
            ProfileSel => {
                self.choices.profile = cycle(
                    &crate::stack::ComposeProfile::ALL,
                    self.choices.profile,
                    dir,
                );
                self.passed.retain(|p| *p != "stack");
            }
            WhisperModel => {
                let ids = Self::options_for_whisper();
                let cur = ids
                    .iter()
                    .position(|i| *i == self.choices.whisper_model)
                    .unwrap_or(0) as i32;
                self.choices.whisper_model =
                    ids[(cur + dir).rem_euclid(ids.len() as i32) as usize].to_string();
                self.choices.whisper_override = None;
                self.passed.retain(|p| *p != "download");
            }
            LlmSourceSel => {
                self.choices.llm_source = cycle(Source::ALL, self.choices.llm_source, dir);
                self.tool = None;
                self.passed.retain(|p| *p != "tool" && *p != "stack");
            }
            DetectedSel => {
                if let Some(slot) = self.page_slot() {
                    let n = self.detected_for(slot).len();
                    if n > 0 {
                        let i = &mut self.detected_idx[slot_index(slot)];
                        *i = (*i as i32 + dir).rem_euclid(n as i32) as usize;
                        self.apply_detected(slot);
                    }
                }
            }
            TtsSourceSel => {
                self.choices.tts_source = cycle(Source::ALL, self.choices.tts_source, dir);
                self.passed.retain(|p| *p != "stack");
            }
            InputDevice | OutputDevice => {
                let Some(Ok(list)) = &self.devices else {
                    return;
                };
                let (names, cur) = if id == InputDevice {
                    (&list.inputs, &mut self.choices.input_device)
                } else {
                    (&list.outputs, &mut self.choices.output_device)
                };
                let mut options: Vec<Option<String>> = vec![None];
                options.extend(names.iter().cloned().map(Some));
                let i = options.iter().position(|o| o == cur).unwrap_or(0) as i32;
                *cur = options[(i + dir).rem_euclid(options.len() as i32) as usize].clone();
            }
            HotkeyEnabled => self.choices.hotkey_enabled = !self.choices.hotkey_enabled,
            _ => {}
        }
    }

    fn apply_detected(&mut self, slot: Slot) {
        let Some(d) = self
            .detected_for(slot)
            .get(self.detected_idx[slot_index(slot)])
            .map(|d| (*d).clone())
        else {
            return;
        };
        match slot {
            Slot::Llm => {
                self.choices.llm_url = d.base_url;
                if let Some(m) = d.models.first() {
                    self.choices.llm_model = m.clone();
                }
                self.tool = None;
                self.passed.retain(|p| *p != "tool");
            }
            Slot::Stt => self.choices.stt_url = d.base_url,
            Slot::Tts => self.choices.tts_url = d.base_url,
        }
        if let Some(n) = d.note {
            self.notes = vec![n];
        }
    }

    fn set_text(&mut self, id: FieldId, s: String) {
        use FieldId::*;
        let c = &mut self.choices;
        match id {
            SttUrl => c.stt_url = s,
            SttModel => c.stt_model = s,
            LlmUrl => c.llm_url = s,
            LlmModel => c.llm_model = s,
            LlmFile => c.llm_file = s,
            TtsUrl => c.tts_url = s,
            TtsModel => c.tts_model = s,
            TtsVoice => c.tts_voice = s,
            PocketVoice => c.pocket_voice = s,
            HotkeyAccel => c.hotkey_accel = s,
            MaxUtterance => {
                if let Ok(v) = s.parse::<u32>() {
                    c.max_utterance_ms = v.min(super::choices::MAX_UTTERANCE_CAP_MS);
                }
            }
            SshDest => {
                c.ssh_destination = s;
                self.passed.retain(|p| *p != "ssh" && *p != "handshake");
            }
            _ => {}
        }
    }

    fn text_value(&self, id: FieldId) -> String {
        self.fields()
            .into_iter()
            .find(|f| f.id == id)
            .map(|f| f.value)
            .unwrap_or_default()
    }

    fn run_action(&mut self, id: FieldId) -> Vec<Effect> {
        use FieldId::*;
        match id {
            Reprobe => {
                self.hw = None;
                self.enter_effects()
            }
            Redetect => {
                self.detect_done = false;
                self.detected.clear();
                self.enter_effects()
            }
            ToolProbeRun if !self.is_busy("tool") => {
                self.mark_busy("tool");
                self.tool = None;
                self.notes.clear();
                let (base_url, extra) = match self.choices.llm_source {
                    Source::Local => (self.choices.llm_url.clone(), Map::new()),
                    Source::Compose => (super::choices::COMPOSE_LLM.to_string(), Map::new()),
                    Source::Remote => {
                        self.notes.push("The Loom is behind an ssh forward that only exists while connected; probing it needs the forward up, so this may fail until `voicelink connect`.".into());
                        (super::choices::loom_url(), super::choices::loom_extra())
                    }
                };
                vec![Effect::ToolProbe {
                    base_url,
                    model: self.choices.llm_model.clone(),
                    extra,
                }]
            }
            Download if !self.is_busy("download") => {
                let need: Vec<_> = self.choices.required_models();
                self.mark_busy("download");
                self.notes.clear();
                vec![Effect::Download(need, self.choices.models_dir.clone())]
            }
            StartStack if !self.is_busy("stack") => {
                self.notes.clear();
                let Some(cmd) = self.stack_command() else {
                    self.notes.push(
                        "compose/compose.yml was not found next to the binary or in the current directory. It ships in the release tarball (compose/); set VOICELINK_COMPOSE to its path."
                            .into(),
                    );
                    return Vec::new();
                };
                if !self.stack_confirm {
                    self.stack_confirm = true;
                    self.notes.extend(self.stack_warnings());
                    self.notes.push("This will run:".into());
                    self.notes.push(cmd.display());
                    self.notes
                        .push("Press Enter again to run it; any other key cancels.".into());
                    return Vec::new();
                }
                self.stack_confirm = false;
                self.mark_busy("stack");
                vec![Effect::StartStack(Box::new(cmd))]
            }
            Meter => {
                if self.meter_on {
                    vec![Effect::StopMeter]
                } else {
                    self.meter_on = true;
                    vec![Effect::StartMeter(self.choices.to_config().audio)]
                }
            }
            RecordPlay if !self.is_busy("record") => {
                self.mark_busy("record");
                self.notes.clear();
                vec![Effect::RecordPlayback(self.choices.to_config().audio)]
            }
            CheckSsh if !self.is_busy("ssh") => {
                self.notes.clear();
                if let Some(p) = self
                    .choices
                    .validate()
                    .into_iter()
                    .find(|p| p.contains("ssh"))
                {
                    self.notes.push(p);
                    return Vec::new();
                }
                self.mark_busy("ssh");
                vec![Effect::CheckSsh(self.choices.to_config().connection)]
            }
            Handshake if !self.is_busy("handshake") => {
                self.notes.clear();
                if let Some(p) = self
                    .choices
                    .validate()
                    .into_iter()
                    .find(|p| p.contains("ssh"))
                {
                    self.notes.push(p);
                    return Vec::new();
                }
                self.mark_busy("handshake");
                vec![Effect::Handshake(self.choices.to_config().connection)]
            }
            Smoke if !self.is_busy("smoke") => {
                self.mark_busy("smoke");
                self.notes.clear();
                vec![Effect::Smoke(Box::new(self.choices.to_config()))]
            }
            Save => {
                self.notes.clear();
                let problems = self.choices.validate();
                if problems.is_empty() {
                    vec![Effect::Save(Box::new(self.choices.to_config()))]
                } else {
                    self.notes.extend(
                        problems
                            .into_iter()
                            .map(|p| format!("Cannot write yet: {p}")),
                    );
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }

    /// Handles a key press.
    pub fn key(&mut self, key: Key) -> Vec<Effect> {
        if let Some(buf) = self.editing.as_mut() {
            match key {
                Key::Char(c) => buf.push(c),
                Key::Backspace => {
                    buf.pop();
                }
                Key::Esc => self.editing = None,
                Key::Enter => {
                    let text = self.editing.take().unwrap_or_default();
                    if let Some(f) = self.fields().get(self.cursor) {
                        self.set_text(f.id, text.trim().to_string());
                    }
                }
                Key::CtrlC => return self.quit(),
                _ => {}
            }
            return Vec::new();
        }
        let fields = self.fields();
        let n = fields.len().max(1);
        if !(key == Key::Enter
            && fields
                .get(self.cursor)
                .is_some_and(|f| f.id == FieldId::StartStack))
        {
            self.stack_confirm = false;
        }
        match key {
            Key::CtrlC | Key::Char('q') => return self.quit(),
            Key::Up => self.cursor = (self.cursor + n - 1) % n,
            Key::Down => self.cursor = (self.cursor + 1) % n,
            Key::Tab | Key::Char('n') => return self.next(),
            Key::BackTab | Key::Char('b') => return self.prev(),
            Key::Left | Key::Right | Key::Char(' ') | Key::Enter => {
                let Some(f) = fields.get(self.cursor) else {
                    return Vec::new();
                };
                let dir = if key == Key::Left { -1 } else { 1 };
                match (f.kind, key) {
                    (FieldKind::Choice, _) => self.cycle_field(f.id, dir),
                    (FieldKind::Toggle, Key::Left | Key::Right | Key::Char(' ') | Key::Enter) => {
                        self.cycle_field(f.id, 1)
                    }
                    (FieldKind::Text, Key::Enter) => self.editing = Some(self.text_value(f.id)),
                    (FieldKind::Action, Key::Enter) => return self.run_action(f.id),
                    _ => {}
                }
            }
            _ => {}
        }
        Vec::new()
    }

    fn quit(&mut self) -> Vec<Effect> {
        self.finished = Some(Finish::Cancelled);
        if self.meter_on {
            vec![Effect::StopMeter]
        } else {
            Vec::new()
        }
    }

    fn next(&mut self) -> Vec<Effect> {
        match Step::ALL.get(self.step.index() + 1) {
            Some(s) => self.goto(*s),
            None => Vec::new(),
        }
    }

    fn prev(&mut self) -> Vec<Effect> {
        match self.step.index().checked_sub(1) {
            Some(i) => self.goto(Step::ALL[i]),
            None => Vec::new(),
        }
    }

    /// Absorbs the result of an effect.
    pub fn apply(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Hardware(hw) => {
                self.clear_busy("probe");
                let p = hw.propose();
                if !self.prefilled {
                    self.apply_tier(p.tier);
                    self.choices.profile = p.profile;
                    self.choices.apply_arch(&hw.arch);
                }
                self.proposal = Some(p);
                self.hw = Some(hw);
            }
            Outcome::Detected(list) => {
                self.clear_busy("detect");
                self.detect_done = true;
                self.detected = list;
                self.detected_idx = [0; 3];
                let c = &self.choices;
                let defaults = [
                    (
                        Slot::Stt,
                        c.stt_source,
                        c.stt_url == super::choices::LOCAL_STT,
                    ),
                    (
                        Slot::Llm,
                        c.llm_source,
                        c.llm_url == super::choices::LOCAL_LLM,
                    ),
                    (
                        Slot::Tts,
                        c.tts_source,
                        c.tts_url == super::choices::LOCAL_TTS,
                    ),
                ];
                for (slot, source, default_url) in defaults {
                    if source == Source::Local && (default_url || !self.prefilled) {
                        self.apply_detected(slot);
                    }
                }
            }
            Outcome::Tool(t) => {
                self.clear_busy("tool");
                self.notes.push(t.explain());
                if matches!(t, ToolProbe::Structured { .. }) {
                    self.passed.push("tool");
                }
                self.tool = Some(t);
            }
            Outcome::Devices(d) => {
                self.clear_busy("devices");
                if let Err(e) = &d {
                    self.notes.push(format!("Audio devices unavailable: {e}"));
                }
                self.devices = Some(d);
            }
            Outcome::Level(db) => self.level_db = Some(db),
            Outcome::MeterStopped(err) => {
                self.meter_on = false;
                self.level_db = None;
                if let Some(e) = err {
                    self.notes.push(format!("Level meter stopped: {e}"));
                }
            }
            Outcome::Progress(file, done, total) => self.progress = Some((file, done, total)),
            Outcome::Lines { what, ok, lines } => {
                self.clear_busy(what);
                self.passed.retain(|p| *p != what);
                if ok {
                    self.passed.push(what);
                }
                if what == "download" {
                    self.progress = None;
                }
                self.notes = lines;
            }
            Outcome::Saved(Ok(p)) => self.finished = Some(Finish::Saved(p)),
            Outcome::Saved(Err(e)) => self.notes = vec![format!("Could not write the config: {e}")],
        }
    }

    /// Summary lines for the review page.
    pub fn summary(&self) -> Vec<String> {
        let c = &self.choices;
        let cfg = c.to_config();
        let mut v = vec![
            format!("tier: {}", c.tier.label()),
            format!(
                "ssh: {}   view session: {}",
                c.ssh_destination, cfg.connection.view_session
            ),
            format!("stt: {}", c.stt_source.label(Slot::Stt)),
            format!("llm: {} ({})", c.llm_source.label(Slot::Llm), c.llm_model),
            format!("tts: {}", c.tts_source.label(Slot::Tts)),
            format!(
                "hotkey: {}",
                if c.hotkey_enabled {
                    c.hotkey_accel.as_str()
                } else {
                    "disabled"
                }
            ),
            format!("models: {}", c.models_dir.display()),
        ];
        if let Some(stack) = &cfg.stack {
            v.push(format!(
                "compose profile: {} (whisper {}, GGUF {})",
                stack.profile.name(),
                stack.whisper_file,
                stack.llm_file
            ));
        }
        for f in &cfg.connection.forwards {
            v.push(format!(
                "forward {}: 127.0.0.1:{} -> {}",
                f.name, f.local_port, f.remote
            ));
        }
        let missing: Vec<_> = c
            .required_models()
            .into_iter()
            .filter(|e| !c.models_dir.join(e.file).exists())
            .map(|e| e.file)
            .collect();
        if !missing.is_empty() {
            v.push(format!("not downloaded yet: {}", missing.join(", ")));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stack::ComposeProfile;

    fn wiz() -> Wizard {
        Wizard::new(None, "laptop")
    }

    fn facts() -> HostFacts {
        HostFacts {
            compose_file: Some("/opt/vl/compose/compose.yml".into()),
            home: Some("/home/me".into()),
            groups: None,
        }
    }

    fn strix() -> HardwareReport {
        HardwareReport {
            os: "linux".into(),
            arch: "x86_64".into(),
            cpus: 32,
            ram_mib: 64 * 1024,
            rocm_gfx: vec!["gfx1151".into()],
            rocm_vram_mib: Some(512),
            rocm_gtt_mib: Some(61_440),
            ..Default::default()
        }
    }

    fn press(w: &mut Wizard, keys: &[Key]) -> Vec<Effect> {
        keys.iter().flat_map(|k| w.key(*k)).collect()
    }

    #[test]
    fn starts_by_probing_and_adopts_the_proposed_tier() {
        let mut w = wiz();
        let e = w.start();
        assert!(matches!(e.as_slice(), [Effect::Probe]));
        assert!(
            w.start().is_empty(),
            "no duplicate probe while one is running"
        );
        w.apply(Outcome::Hardware(strix()));
        assert_eq!(w.choices.tier, Tier::RocmLarge);
        assert_eq!(w.choices.whisper_model, "whisper-large-v3-turbo");
        // Strix Halo (gfx1151) proposes the Vulkan profile, not rocm.
        assert_eq!(w.choices.profile, ComposeProfile::Vulkan);
        assert!(
            w.proposal
                .as_ref()
                .unwrap()
                .evidence
                .iter()
                .any(|l| l.contains("gfx1151"))
        );
    }

    #[test]
    fn prefilled_wizard_keeps_existing_choices() {
        let mut cfg = Choices::defaults_for(Tier::Cpu, "x").to_config();
        cfg.connection.ssh_destination = "me@tail.net".into();
        let mut w = Wizard::new(Some(&cfg), "laptop");
        w.start();
        w.apply(Outcome::Hardware(strix()));
        assert_eq!(w.choices.ssh_destination, "me@tail.net");
        assert_ne!(
            w.choices.tier,
            Tier::RocmLarge,
            "an existing config is not overridden by the probe"
        );
    }

    #[test]
    fn walks_all_pages_and_saves() {
        let mut w = wiz();
        w.start();
        w.apply(Outcome::Hardware(strix()));
        let mut seen = vec![w.step];
        for _ in 0..9 {
            w.key(Key::Char('n'));
            seen.push(w.step);
        }
        assert_eq!(seen, Step::ALL.to_vec());
        assert!(w.key(Key::Char('n')).is_empty(), "no page after the last");
        let e = w.key(Key::Enter);
        match e.as_slice() {
            [Effect::Save(cfg)] => {
                assert_eq!(cfg.connection.ssh_destination, "machinelearn@192.168.2.132")
            }
            other => panic!("{other:?}"),
        }
        w.apply(Outcome::Saved(Ok("/tmp/c.toml".into())));
        assert_eq!(w.finished, Some(Finish::Saved("/tmp/c.toml".into())));
    }

    #[test]
    fn entering_pages_triggers_their_probes() {
        let mut w = wiz();
        w.start();
        w.apply(Outcome::Hardware(HardwareReport::default()));
        let e = w.goto(Step::Speech);
        assert!(matches!(e.as_slice(), [Effect::Detect]));
        assert!(
            w.goto(Step::Llm).is_empty(),
            "one detection serves all three pages"
        );
        let e = w.goto(Step::Audio);
        assert!(matches!(e.as_slice(), [Effect::ListDevices]));
        w.meter_on = true;
        let e = w.goto(Step::Hotkey);
        assert!(matches!(e.as_slice(), [Effect::StopMeter]));
    }

    #[test]
    fn text_editing() {
        let mut w = wiz();
        w.goto(Step::Connection);
        assert_eq!(w.fields()[0].id, FieldId::SshDest);
        press(&mut w, &[Key::Enter]);
        assert_eq!(w.editing.as_deref(), Some("machinelearn@192.168.2.132"));
        for _ in 0..40 {
            w.key(Key::Backspace);
        }
        for c in "me@tail".chars() {
            w.key(Key::Char(c));
        }
        // 'q' and 'n' are text while editing, not navigation.
        w.key(Key::Char('q'));
        w.key(Key::Char('n'));
        assert!(w.finished.is_none() && w.step == Step::Connection);
        w.key(Key::Enter);
        assert_eq!(w.choices.ssh_destination, "me@tailqn");
        press(&mut w, &[Key::Enter, Key::Esc]);
        assert_eq!(w.choices.ssh_destination, "me@tailqn", "escape cancels");
    }

    #[test]
    fn ssh_checks_emit_effects_and_refuse_bad_destinations() {
        let mut w = wiz();
        w.goto(Step::Connection);
        w.key(Key::Down);
        let e = w.key(Key::Enter);
        assert!(
            matches!(e.as_slice(), [Effect::CheckSsh(c)] if c.ssh_destination == "machinelearn@192.168.2.132")
        );
        assert!(w.key(Key::Enter).is_empty(), "no second concurrent check");
        w.apply(Outcome::Lines {
            what: "ssh",
            ok: true,
            lines: vec!["voice-paned 0.1.0".into()],
        });
        assert!(w.passed.contains(&"ssh"));
        w.choices.ssh_destination = "-oProxyCommand=evil".into();
        w.key(Key::Down);
        assert!(w.key(Key::Enter).is_empty());
        assert!(w.notes.iter().any(|n| n.contains("ssh destination")));
    }

    fn detected() -> Vec<DetectedServer> {
        let s =
            |slot, label: &str, url: &str, models: &[&str], note: Option<&str>| DetectedServer {
                slot,
                label: label.into(),
                base_url: url.into(),
                models: models.iter().map(|m| m.to_string()).collect(),
                note: note.map(String::from),
            };
        vec![
            s(
                Slot::Llm,
                "llama-server",
                "http://127.0.0.1:8080/v1",
                &["qwen"],
                None,
            ),
            s(
                Slot::Llm,
                "Ollama",
                "http://127.0.0.1:11434/v1",
                &["llama3"],
                None,
            ),
            s(
                Slot::Stt,
                "whisper-server",
                "http://127.0.0.1:8080/v1",
                &[],
                Some("needs --inference-path /v1/audio/transcriptions"),
            ),
            s(
                Slot::Tts,
                "Kokoro-FastAPI",
                "http://127.0.0.1:8880/v1",
                &["kokoro"],
                None,
            ),
        ]
    }

    #[test]
    fn llm_detection_prefills_and_tool_probe_warns() {
        let mut w = wiz();
        w.choices.llm_source = Source::Local;
        w.goto(Step::Llm);
        w.apply(Outcome::Detected(detected()));
        assert_eq!(w.choices.llm_model, "qwen");
        // Row 1 is the detected-server chooser.
        press(&mut w, &[Key::Down, Key::Right]);
        assert_eq!(w.choices.llm_url, "http://127.0.0.1:11434/v1");
        assert_eq!(w.choices.llm_model, "llama3");
        // Find the probe row and run it.
        let idx = w
            .fields()
            .iter()
            .position(|f| f.id == FieldId::ToolProbeRun)
            .unwrap();
        w.cursor = idx;
        let e = w.key(Key::Enter);
        assert!(
            matches!(e.as_slice(), [Effect::ToolProbe { base_url, model, .. }] if base_url.contains("11434") && model == "llama3")
        );
        w.apply(Outcome::Tool(ToolProbe::NoToolCalls {
            reply: "5".into(),
            looks_like_text_call: false,
        }));
        assert!(w.notes.iter().any(|n| n.contains("--jinja")));
        assert!(!w.passed.contains(&"tool"));
    }

    #[test]
    fn stt_and_tts_detection_is_per_slot_and_carries_notes() {
        let mut w = wiz();
        w.choices.stt_source = Source::Local;
        w.choices.tts_source = Source::Local;
        w.goto(Step::Speech);
        w.apply(Outcome::Detected(detected()));
        assert_eq!(w.choices.stt_url, "http://127.0.0.1:8080/v1");
        assert_eq!(w.detected_for(Slot::Stt).len(), 1);
        assert_eq!(w.detected_for(Slot::Llm).len(), 2);
        // Applying the whisper-server surfaces its --inference-path requirement.
        w.notes.clear();
        press(&mut w, &[Key::Down, Key::Right]);
        assert!(
            w.notes.iter().any(|n| n.contains("--inference-path")),
            "{:?}",
            w.notes
        );
        w.goto(Step::Voice);
        assert_eq!(w.choices.tts_url, "http://127.0.0.1:8880/v1");
        let shown = w
            .fields()
            .into_iter()
            .find(|f| f.id == FieldId::DetectedSel)
            .unwrap();
        assert!(shown.value.contains("Kokoro-FastAPI"));
    }

    #[test]
    fn loom_probe_uses_forward_url_and_extra() {
        let mut w = wiz();
        w.choices.llm_source = Source::Remote;
        w.goto(Step::Llm);
        let idx = w
            .fields()
            .iter()
            .position(|f| f.id == FieldId::ToolProbeRun)
            .unwrap();
        w.cursor = idx;
        match w.key(Key::Enter).as_slice() {
            [
                Effect::ToolProbe {
                    base_url, extra, ..
                },
            ] => {
                assert_eq!(base_url, "http://127.0.0.1:18084/v1");
                assert!(extra.contains_key("loom_options"));
            }
            other => panic!("{other:?}"),
        }
    }

    fn stack_row(w: &Wizard) -> usize {
        w.fields()
            .iter()
            .position(|f| f.id == FieldId::StartStack)
            .unwrap()
    }

    #[test]
    fn the_stack_only_starts_after_a_second_explicit_enter() {
        let mut w = wiz();
        w.facts = facts();
        w.apply(Outcome::Hardware(HardwareReport {
            os: "linux".into(),
            arch: "x86_64".into(),
            ram_mib: 64 * 1024,
            nvidia: vec![("RTX 4090".into(), 24_564)],
            ..Default::default()
        }));
        assert_eq!(w.choices.profile, ComposeProfile::Cuda);
        w.goto(Step::Models);
        w.cursor = stack_row(&w);
        // First Enter only shows the exact command.
        assert!(w.key(Key::Enter).is_empty());
        assert!(w.stack_confirm);
        assert!(
            w.notes.iter().any(|n| n
                == "docker compose -f /opt/vl/compose/compose.yml --profile cuda up -d llm-cuda stt-cuda tts-cuda"
                || n.ends_with("--profile cuda up -d llm-cuda stt-cuda tts-cuda")),
            "{:?}",
            w.notes
        );
        // Any other key cancels the pending confirmation.
        w.key(Key::Down);
        w.key(Key::Up);
        assert!(!w.stack_confirm);
        assert!(w.key(Key::Enter).is_empty(), "back to asking again");
        // Second Enter on the same row runs it, once.
        match w.key(Key::Enter).as_slice() {
            [Effect::StartStack(cmd)] => {
                assert_eq!(cmd.profile, ComposeProfile::Cuda);
                assert_eq!(cmd.services, ["llm-cuda", "stt-cuda", "tts-cuda"]);
            }
            other => panic!("{other:?}"),
        }
        assert!(w.key(Key::Enter).is_empty(), "no second concurrent start");
        w.apply(Outcome::Lines {
            what: "stack",
            ok: true,
            lines: vec!["started".into()],
        });
        assert!(w.passed.contains(&"stack"));
    }

    #[test]
    fn an_aarch64_host_does_not_get_a_whisper_stack_proposal() {
        let mut w = wiz();
        w.apply(Outcome::Hardware(HardwareReport {
            os: "linux".into(),
            arch: "aarch64".into(),
            ram_mib: 32 * 1024,
            vulkan: vec![("Mali".into(), "INTEGRATED_GPU".into())],
            ..Default::default()
        }));
        assert_eq!(w.choices.tier, Tier::SmallGpu);
        assert_eq!(
            w.choices.stt_source,
            Source::Remote,
            "no arm64 whisper image"
        );
        assert_eq!(w.choices.llm_source, Source::Compose);
        assert!(
            w.proposal
                .as_ref()
                .unwrap()
                .evidence
                .iter()
                .any(|l| l.contains("aarch64") && l.contains("stt-vulkan"))
        );
        // Choosing the stack for speech anyway is warned about before confirmation.
        w.choices.stt_source = Source::Compose;
        let warnings = w.stack_warnings();
        assert!(
            warnings
                .iter()
                .any(|m| m.contains("no aarch64 image for stt-vulkan")),
            "{warnings:?}"
        );
    }

    #[test]
    fn a_missing_gguf_is_warned_about_before_the_stack_starts() {
        let mut w = wiz();
        w.facts = facts();
        w.choices.apply_tier(Tier::Nvidia, "laptop");
        w.choices.models_dir = "/nonexistent/vl-models".into();
        w.goto(Step::Models);
        w.cursor = stack_row(&w);
        assert!(w.key(Key::Enter).is_empty());
        assert!(
            w.notes
                .iter()
                .any(|n| n.contains("llm.gguf does not exist") && n.contains("llm-cuda")),
            "{:?}",
            w.notes
        );
    }

    #[test]
    fn leaving_the_page_cancels_a_pending_start_and_missing_files_explain() {
        let mut w = wiz();
        w.facts = facts();
        w.choices.apply_tier(Tier::Nvidia, "laptop");
        w.goto(Step::Models);
        w.cursor = stack_row(&w);
        w.key(Key::Enter);
        assert!(w.stack_confirm);
        w.key(Key::Char('n'));
        w.goto(Step::Models);
        assert!(!w.stack_confirm);
        // Without compose.yml the wizard says where it belongs, and runs nothing.
        w.facts = HostFacts::default();
        w.cursor = stack_row(&w);
        assert_eq!(w.fields()[w.cursor].value, "compose files not found");
        assert!(w.key(Key::Enter).is_empty());
        assert!(w.notes.iter().any(|n| n.contains("VOICELINK_COMPOSE")));
    }

    #[test]
    fn device_choice_cycles_through_defaults_and_names() {
        let mut w = wiz();
        w.goto(Step::Audio);
        w.apply(Outcome::Devices(Ok(DeviceList {
            inputs: vec!["USB Mic".into(), "Built-in".into()],
            outputs: vec!["Speakers".into()],
            default_input: None,
            default_output: None,
        })));
        press(&mut w, &[Key::Right]);
        assert_eq!(w.choices.input_device.as_deref(), Some("USB Mic"));
        press(&mut w, &[Key::Right, Key::Right]);
        assert_eq!(
            w.choices.input_device, None,
            "wraps back to the system default"
        );
        press(&mut w, &[Key::Left]);
        assert_eq!(w.choices.input_device.as_deref(), Some("Built-in"));
        // meter toggles on, then off with a stop effect
        w.cursor = 2;
        assert!(matches!(
            w.key(Key::Enter).as_slice(),
            [Effect::StartMeter(_)]
        ));
        w.apply(Outcome::Level(-30.0));
        assert!(matches!(w.key(Key::Enter).as_slice(), [Effect::StopMeter]));
        w.apply(Outcome::MeterStopped(None));
        assert!(!w.meter_on && w.level_db.is_none());
    }

    #[test]
    fn missing_audio_does_not_block() {
        let mut w = wiz();
        w.goto(Step::Audio);
        w.apply(Outcome::Devices(Err("audio not built yet".into())));
        assert!(w.notes[0].contains("unavailable"));
        assert!(w.key(Key::Char('n')).is_empty());
        assert_eq!(w.step, Step::Hotkey);
    }

    #[test]
    fn hotkey_is_off_by_default_and_toggles() {
        let mut w = wiz();
        w.goto(Step::Hotkey);
        assert!(!w.choices.hotkey_enabled);
        assert_eq!(w.fields().len(), 1);
        w.key(Key::Char(' '));
        assert!(w.choices.hotkey_enabled);
        assert_eq!(w.fields().len(), 2);
    }

    #[test]
    fn model_download_lists_required_files() {
        let mut w = wiz();
        w.apply(Outcome::Hardware(strix()));
        w.goto(Step::Models);
        let ids: Vec<_> = w.fields().iter().map(|f| f.id).collect();
        assert_eq!(
            ids,
            [FieldId::ProfileSel, FieldId::Download, FieldId::StartStack]
        );
        w.cursor = 1;
        match w.key(Key::Enter).as_slice() {
            [Effect::Download(entries, _)] => {
                let ids: Vec<_> = entries.iter().map(|e| e.id).collect();
                assert_eq!(ids, vec!["whisper-large-v3-turbo"]);
            }
            other => panic!("{other:?}"),
        }
        w.apply(Outcome::Progress("x".into(), 1, 2));
        w.apply(Outcome::Lines {
            what: "download",
            ok: true,
            lines: vec!["done".into()],
        });
        assert!(w.progress.is_none() && w.passed.contains(&"download"));
    }

    #[test]
    fn thin_tier_needs_no_models() {
        let mut w = wiz();
        w.apply(Outcome::Hardware(HardwareReport {
            ram_mib: 2048,
            ..Default::default()
        }));
        assert_eq!(w.choices.tier, Tier::Thin);
        w.goto(Step::Models);
        assert!(w.fields().is_empty());
        assert_eq!(w.choices.to_config().connection.forwards.len(), 3);
    }

    #[test]
    fn invalid_config_is_not_written_and_quit_cancels() {
        let mut w = wiz();
        w.choices.ssh_destination = String::new();
        w.goto(Step::Review);
        assert!(w.key(Key::Enter).is_empty());
        assert!(w.notes[0].contains("Cannot write"));
        w.key(Key::Char('q'));
        assert_eq!(w.finished, Some(Finish::Cancelled));
    }
}
