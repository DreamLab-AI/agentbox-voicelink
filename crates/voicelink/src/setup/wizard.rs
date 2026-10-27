//! The wizard as a pure state machine: keys and effect results go in, a list
//! of effects to run comes out. No terminal, network or subprocess is touched
//! here, so every transition is unit-tested.

use std::path::PathBuf;

use serde_json::{Map, Value};

use super::catalogue::{self, Entry};
use super::choices::{Choices, LlmKind, Tier, cycle};
use super::net::{DetectedServer, ToolProbe};
use super::probe::{HardwareReport, Proposal};
use crate::audio::DeviceList;
use crate::config::{self, Config};

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
    /// Model downloads.
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
            Step::Models => "Model downloads",
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
    /// Look for local OpenAI-compatible servers.
    DetectLlm,
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
        /// Which action (`"ssh"`, `"handshake"`, `"download"`, `"record"`, `"smoke"`).
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
    SttKindSel,
    WhisperModel,
    SttUrl,
    SttModel,
    LlmKindSel,
    LlmDetected,
    Redetect,
    LlmUrl,
    LlmModel,
    ToolProbeRun,
    TtsKindSel,
    KokoroVoice,
    TtsUrl,
    TtsModel,
    TtsVoice,
    Download,
    InputDevice,
    OutputDevice,
    Meter,
    RecordPlay,
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
    /// Local LLM servers found.
    pub detected: Vec<DetectedServer>,
    /// Detection has completed.
    pub detect_done: bool,
    /// Index into `detected` last applied.
    pub detected_idx: usize,
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
            detected_idx: 0,
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
            Step::Llm if !self.detect_done && !self.is_busy("detect") => {
                self.mark_busy("detect");
                e.push(Effect::DetectLlm);
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
                f.push(field(SttKindSel, "Provider", c.stt_kind.label(), Choice));
                match c.stt_kind {
                    super::choices::SttKind::Embedded => {
                        let label = match (&c.whisper_path_override, c.whisper_entry()) {
                            (Some(p), _) => format!("custom file {p}"),
                            (None, Some(e)) => e.label.to_string(),
                            _ => c.whisper_model.clone(),
                        };
                        f.push(field(WhisperModel, "Model", label, Choice));
                    }
                    super::choices::SttKind::Http => {
                        f.push(field(SttUrl, "Base URL", &c.stt_url, Text));
                        f.push(field(SttModel, "Model name", &c.stt_model, Text));
                    }
                    super::choices::SttKind::Nemotron => {}
                }
            }
            Step::Llm => {
                f.push(field(LlmKindSel, "Provider", c.llm_kind.label(), Choice));
                if c.llm_kind == super::choices::LlmKind::Local {
                    let found = match self.detected.get(self.detected_idx) {
                        Some(d) => format!("{} at {}", d.label, d.base_url),
                        None if self.detect_done => "none found".to_string(),
                        None => "looking...".to_string(),
                    };
                    f.push(field(LlmDetected, "Detected server", found, Choice));
                    f.push(field(Redetect, "Detect again", "", Action));
                    f.push(field(LlmUrl, "Base URL", &c.llm_url, Text));
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
                f.push(field(TtsKindSel, "Provider", c.tts_kind.label(), Choice));
                match c.tts_kind {
                    super::choices::TtsKind::Embedded => {
                        f.push(field(KokoroVoice, "Voice", &c.kokoro_voice, Text))
                    }
                    super::choices::TtsKind::Http => {
                        f.push(field(TtsUrl, "Base URL", &c.tts_url, Text));
                        f.push(field(TtsModel, "Model name", &c.tts_model, Text));
                        f.push(field(TtsVoice, "Voice", &c.tts_voice, Text));
                    }
                    super::choices::TtsKind::Pocket => {
                        f.push(field(TtsVoice, "Voice", &c.tts_voice, Text))
                    }
                }
            }
            Step::Models => {
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
        catalogue::whisper_entries().map(|e| e.id).collect()
    }

    fn cycle_field(&mut self, id: FieldId, dir: i32) {
        use FieldId::*;
        match id {
            TierSel => {
                let t = cycle(&super::choices::Tier::ALL, self.choices.tier, dir);
                self.apply_tier(t);
            }
            SttKindSel => {
                self.choices.stt_kind =
                    cycle(super::choices::SttKind::ALL, self.choices.stt_kind, dir)
            }
            WhisperModel => {
                let ids = Self::options_for_whisper();
                let cur = ids
                    .iter()
                    .position(|i| *i == self.choices.whisper_model)
                    .unwrap_or(0) as i32;
                self.choices.whisper_model =
                    ids[(cur + dir).rem_euclid(ids.len() as i32) as usize].to_string();
                self.choices.whisper_path_override = None;
            }
            LlmKindSel => {
                self.choices.llm_kind =
                    cycle(super::choices::LlmKind::ALL, self.choices.llm_kind, dir);
                self.tool = None;
                self.passed.retain(|p| *p != "tool");
            }
            LlmDetected if !self.detected.is_empty() => {
                self.detected_idx = (self.detected_idx as i32 + dir)
                    .rem_euclid(self.detected.len() as i32)
                    as usize;
                self.apply_detected();
            }
            TtsKindSel => {
                self.choices.tts_kind =
                    cycle(super::choices::TtsKind::ALL, self.choices.tts_kind, dir)
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

    fn apply_detected(&mut self) {
        if let Some(d) = self.detected.get(self.detected_idx) {
            self.choices.llm_url = d.base_url.clone();
            if let Some(m) = d.models.first() {
                self.choices.llm_model = m.clone();
            }
            self.tool = None;
            self.passed.retain(|p| *p != "tool");
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
            KokoroVoice => c.kokoro_voice = s,
            TtsUrl => c.tts_url = s,
            TtsModel => c.tts_model = s,
            TtsVoice => c.tts_voice = s,
            HotkeyAccel => c.hotkey_accel = s,
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
                self.enter_effects()
            }
            ToolProbeRun if !self.is_busy("tool") => {
                self.mark_busy("tool");
                self.tool = None;
                self.notes.clear();
                let (base_url, extra) = match self.choices.llm_kind {
                    LlmKind::Local => (self.choices.llm_url.clone(), Map::new()),
                    LlmKind::Compose => (super::choices::COMPOSE_LLM.to_string(), Map::new()),
                    LlmKind::Loom => {
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
                }
                self.proposal = Some(p);
                self.hw = Some(hw);
            }
            Outcome::Detected(list) => {
                self.clear_busy("detect");
                self.detect_done = true;
                self.detected = list;
                self.detected_idx = 0;
                let default_url = self.choices.llm_url == super::choices::LOCAL_LLM;
                if self.choices.llm_kind == LlmKind::Local && (default_url || !self.prefilled) {
                    self.apply_detected();
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
            format!("stt: {}", c.stt_kind.label()),
            format!("llm: {} ({})", c.llm_kind.label(), c.llm_model),
            format!("tts: {}", c.tts_kind.label()),
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

    fn wiz() -> Wizard {
        Wizard::new(None, "laptop")
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
        let e = w.goto(Step::Llm);
        assert!(matches!(e.as_slice(), [Effect::DetectLlm]));
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

    #[test]
    fn llm_detection_prefills_and_tool_probe_warns() {
        let mut w = wiz();
        w.choices.llm_kind = LlmKind::Local;
        w.goto(Step::Llm);
        w.apply(Outcome::Detected(vec![
            DetectedServer {
                label: "llama-server".into(),
                base_url: "http://127.0.0.1:8080/v1".into(),
                models: vec!["qwen".into()],
            },
            DetectedServer {
                label: "Ollama".into(),
                base_url: "http://127.0.0.1:11434/v1".into(),
                models: vec!["llama3".into()],
            },
        ]));
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
    fn loom_probe_uses_forward_url_and_extra() {
        let mut w = wiz();
        w.choices.llm_kind = LlmKind::Loom;
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
        match w.key(Key::Enter).as_slice() {
            [Effect::Download(entries, _)] => {
                let ids: Vec<_> = entries.iter().map(|e| e.id).collect();
                assert_eq!(
                    ids,
                    vec!["whisper-large-v3-turbo", "kokoro-model", "kokoro-voices"]
                );
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
