//! The concierge: a local LLM with tools that sits between the operator's
//! spoken words and the agent CLIs in tmux.
//!
//! It rewrites speech into clean prompts, confirms what it did, narrates
//! agent events, and keeps the most recent permission prompt as context so a
//! bare "approve" or "option two" resolves against it. Writes go only to the
//! focused pane (I1); acting on a background prompt means `focus_pane` first,
//! which the operator can see.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{Value, json};
use tokio::sync::mpsc;
use voicelink_backends::{ChatMessage, ChatModel, ChatRequest, Role, ToolCall, ToolSpec};
use voicelink_proto::{AgentEvent, AgentKind, ErrorCode, EventKind, Focus, Key, PaneInfo};

use crate::config;
use crate::link::{LinkError, LinkHandle, LinkNotice};

/// Most LLM calls in one turn (tool rounds included).
const MAX_ROUNDS: usize = 4;
/// Pane lines read to find a prompt's options.
const PROMPT_LINES: u32 = 40;
/// Longest pane text handed back to the model, in characters (from the tail).
const MAX_TOOL_TEXT: usize = 4000;

/// Work for the concierge loop.
#[derive(Debug)]
pub(crate) enum Input {
    /// What the operator said.
    Transcript(String),
    /// Something the controller link reported.
    Link(LinkNotice),
}

/// A permission prompt waiting for the operator's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingPrompt {
    pane: String,
    window: Option<u32>,
    agent: AgentKind,
    question: String,
    options: Vec<(u32, String)>,
    screen: String,
}

/// A send that the controller refused because the input box held text.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingForce {
    text: String,
    submit: bool,
}

#[derive(Default)]
struct State {
    focus: Option<Focus>,
    panes: Vec<PaneInfo>,
    /// Whole turns, oldest first; a turn is never split (tool calls stay with their replies).
    history: VecDeque<Vec<ChatMessage>>,
    prompt: Option<PendingPrompt>,
    force: Option<PendingForce>,
    link_was_down: bool,
}

/// The concierge. Cheap to clone via `Arc`.
pub(crate) struct Concierge {
    llm: Arc<dyn ChatModel>,
    link: LinkHandle,
    say: mpsc::UnboundedSender<String>,
    cfg: config::Concierge,
    state: Mutex<State>,
}

/// What a tool call tells the turn loop to do next.
enum Outcome {
    Continue(String),
    /// Stop the turn without asking the model again (the operator was asked something).
    Halt(String),
}

fn agent_name(kind: AgentKind) -> &'static str {
    match kind {
        AgentKind::Claude => "Claude",
        AgentKind::Codex => "Codex",
        AgentKind::Other => "The agent",
    }
}

/// Normalises a transcript for matching: lower-case, letters and spaces only.
fn normalise(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// True for an utterance that is just "stop", "interrupt" or "cancel".
pub(crate) fn is_stop_word(text: &str) -> bool {
    matches!(
        normalise(text).as_str(),
        "stop" | "interrupt" | "cancel" | "please stop" | "stop please"
    )
}

fn is_affirmative(text: &str) -> bool {
    matches!(
        normalise(text).as_str(),
        "yes"
            | "yeah"
            | "yep"
            | "sure"
            | "send"
            | "send it"
            | "send anyway"
            | "go ahead"
            | "do it"
            | "yes please"
            | "yes send it"
            | "ok"
            | "okay"
    )
}

fn is_negative(text: &str) -> bool {
    matches!(
        normalise(text).as_str(),
        "no" | "nope"
            | "don't"
            | "do not"
            | "never mind"
            | "nevermind"
            | "cancel"
            | "no thanks"
            | "leave it"
            | "don't send"
            | "no don't"
    )
}

/// Maps a spoken or model-written key name to a wire key.
pub(crate) fn parse_key(name: &str) -> Option<Key> {
    let n = name.trim().to_ascii_lowercase();
    Some(match n.as_str() {
        "enter" | "return" => Key::Enter,
        "esc" | "escape" => Key::Escape,
        "tab" => Key::Tab,
        "shift+tab" | "shift-tab" | "backtab" | "btab" => Key::BackTab,
        "space" => Key::Space,
        "up" => Key::Up,
        "down" => Key::Down,
        "left" => Key::Left,
        "right" => Key::Right,
        "zero" => Key::Digit(0),
        "one" => Key::Digit(1),
        "two" => Key::Digit(2),
        "three" => Key::Digit(3),
        "four" => Key::Digit(4),
        "five" => Key::Digit(5),
        "six" => Key::Digit(6),
        "seven" => Key::Digit(7),
        "eight" => Key::Digit(8),
        "nine" => Key::Digit(9),
        _ => {
            let mut chars = name.trim().chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_digit() => Key::Digit(c as u8 - b'0'),
                (Some(c), None) if c.is_ascii_alphabetic() => Key::Letter(c),
                _ => return None,
            }
        }
    })
}

fn digit_word(d: u8) -> &'static str {
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ][usize::from(d.min(9))]
}

/// Plain-English description of a key press, for the spoken announcement.
fn describe_keys(keys: &[Key]) -> String {
    match keys {
        [Key::Digit(d)] | [Key::Digit(d), Key::Enter] => {
            format!("Choosing option {}", digit_word(*d))
        }
        _ => {
            let parts: Vec<String> = keys
                .iter()
                .map(|k| match k {
                    Key::Digit(d) => digit_word(*d).to_string(),
                    Key::Letter(c) => c.to_string(),
                    other => format!("{other:?}").to_lowercase(),
                })
                .collect();
            format!("Pressing {}", parts.join(", "))
        }
    }
}

/// Finds numbered menu options (`1. Yes`, `❯ 2) No`) in pane text.
pub(crate) fn parse_options(screen: &str) -> Vec<(u32, String)> {
    let mut out: Vec<(u32, String)> = Vec::new();
    for line in screen.lines() {
        let t = line.trim_start_matches(|c: char| c.is_whitespace() || "❯>›»*│|".contains(c));
        let digits: String = t.chars().take_while(char::is_ascii_digit).collect();
        if digits.is_empty() || digits.len() > 2 {
            continue;
        }
        let rest = &t[digits.len()..];
        let Some(rest) = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')')) else {
            continue;
        };
        let label = rest
            .trim()
            .trim_end_matches(|c: char| c.is_whitespace() || "│|".contains(c))
            .to_string();
        if label.is_empty() {
            continue;
        }
        let n: u32 = digits.parse().unwrap_or(0);
        // Options count up from 1; a restart means an older menu earlier on screen.
        if n == 1 {
            out.clear();
        }
        if n as usize == out.len() + 1 {
            out.push((n, label));
        }
    }
    out
}

/// The last meaningful line before the options, used when the event has no text.
fn guess_question(screen: &str) -> String {
    let mut best = String::new();
    for line in screen.lines() {
        let t = line.trim_matches(|c: char| c.is_whitespace() || "│|╭╮╰╯─".contains(c));
        if t.is_empty() {
            continue;
        }
        let is_option = t
            .trim_start_matches(|c: char| "❯>›»* ".contains(c))
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit());
        if is_option {
            break;
        }
        best = t.to_string();
    }
    best
}

fn tail_chars(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        s.to_string()
    } else {
        s.chars().skip(count - max).collect()
    }
}

fn tool_specs() -> Vec<ToolSpec> {
    let spec = |name: &str, description: &str, parameters: Value| ToolSpec {
        name: name.into(),
        description: description.into(),
        parameters,
    };
    vec![
        spec(
            "list_panes",
            "List every pane in every tab: pane id, tab number, tab name, agent, and which one is focused.",
            json!({"type": "object", "properties": {}}),
        ),
        spec(
            "read_pane",
            "Read the last lines of any pane's screen, focused or not.",
            json!({"type": "object", "properties": {
                "pane": {"type": "string", "description": "Pane id such as %3"},
                "lines": {"type": "integer", "description": "How many trailing lines (default 40)"}
            }, "required": ["pane"]}),
        ),
        spec(
            "send_to_focused",
            "Type a prompt into the FOCUSED pane. Never takes a pane: to write elsewhere, focus_pane first.",
            json!({"type": "object", "properties": {
                "text": {"type": "string", "description": "The clean prompt to type"},
                "submit": {"type": "boolean", "description": "Press enter afterwards (default true)"}
            }, "required": ["text"]}),
        ),
        spec(
            "press_keys",
            "Press keys in the FOCUSED pane, for menus and permission prompts. Names: enter, esc, tab, shift+tab, space, up, down, left, right, a digit 0-9, or a single letter.",
            json!({"type": "object", "properties": {
                "keys": {"type": "array", "items": {"type": "string"}, "description": "For example [\"2\", \"enter\"]"}
            }, "required": ["keys"]}),
        ),
        spec(
            "interrupt",
            "Interrupt the focused agent (Ctrl-C). Only when the operator explicitly says stop or interrupt.",
            json!({"type": "object", "properties": {}}),
        ),
        spec(
            "focus_pane",
            "Switch the operator's view to a pane (visible to them). Required before writing to a background tab.",
            json!({"type": "object", "properties": {
                "pane": {"type": "string", "description": "Pane id such as %3"}
            }, "required": ["pane"]}),
        ),
    ]
}

const SYSTEM_PROMPT: &str = "You are a terse spoken assistant between the operator and AI agent CLIs (Claude Code, Codex) \
running in tmux tabs. Everything you say is read aloud: answer in one or two short sentences, with no markdown, lists or \
code. When the operator asks an agent to do something, rewrite their spoken words into a clean, specific prompt (fix \
speech-recognition slips, drop filler, invent nothing) and send it with send_to_focused. Afterwards confirm in a few words \
what you typed or approved. Writing tools only ever act on the focused pane; to answer a prompt in a background tab, call \
focus_pane for that pane first, then press_keys. Use list_panes and read_pane when you need to see what is on screen. \
Resolve 'approve', 'yes', 'option two' and similar against the pending prompt described below: press the option's digit, \
and add enter only if the menu needs confirmation. Never interrupt unless the operator explicitly says stop or interrupt. \
If a request is unclear, ask one short question instead of guessing. If the operator is only chatting, answer briefly and \
touch no pane.";

impl Concierge {
    /// Creates a concierge that speaks through `say`.
    pub(crate) fn new(
        llm: Arc<dyn ChatModel>,
        link: LinkHandle,
        say: mpsc::UnboundedSender<String>,
        cfg: config::Concierge,
    ) -> Arc<Self> {
        Arc::new(Concierge {
            llm,
            link,
            say,
            cfg,
            state: Mutex::new(State::default()),
        })
    }

    fn st(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn speak(&self, text: impl Into<String>) {
        let text = text.into();
        if !text.trim().is_empty() {
            let _ = self.say.send(text);
        }
    }

    /// Runs the concierge until `inbox` closes. Transcripts and events are
    /// handled one at a time; a bare "stop" pre-empts whatever is in flight.
    pub(crate) async fn run(self: Arc<Self>, mut inbox: mpsc::UnboundedReceiver<Input>) {
        enum Work {
            Transcript(String),
            Event(AgentEvent),
        }
        let mut queue: VecDeque<Work> = VecDeque::new();
        let mut current: Option<tokio::task::JoinHandle<()>> = None;
        loop {
            tokio::select! {
                input = inbox.recv() => {
                    let Some(input) = input else { break };
                    match input {
                        Input::Transcript(t) if is_stop_word(&t) && !(self.st().force.is_some() && is_negative(&t)) => {
                            if let Some(c) = current.take() { c.abort(); }
                            queue.clear();
                            self.interrupt_now().await;
                        }
                        Input::Transcript(t) => queue.push_back(Work::Transcript(t)),
                        Input::Link(LinkNotice::Event(e)) => queue.push_back(Work::Event(e)),
                        Input::Link(n) => self.on_notice(n),
                    }
                }
                _ = async { if let Some(c) = current.as_mut() { let _ = c.await; } }, if current.is_some() => {
                    current = None;
                }
            }
            if current.is_none() {
                if let Some(work) = queue.pop_front() {
                    let me = Arc::clone(&self);
                    current = Some(tokio::spawn(async move {
                        match work {
                            Work::Transcript(t) => me.on_transcript(&t).await,
                            Work::Event(e) => me.on_event(e).await,
                        }
                    }));
                }
            }
        }
        if let Some(c) = current {
            let _ = c.await;
        }
    }

    fn on_notice(&self, n: LinkNotice) {
        match n {
            LinkNotice::Connected(f) => {
                let was_down = {
                    let mut st = self.st();
                    st.focus = Some(f);
                    std::mem::take(&mut st.link_was_down)
                };
                if was_down {
                    self.speak("Reconnected to agentbox.");
                }
            }
            LinkNotice::Focus(f) => {
                let mut st = self.st();
                st.focus = Some(f);
                st.force = None;
            }
            LinkNotice::Disconnected(_) => {
                let first = !std::mem::replace(&mut self.st().link_was_down, true);
                if first {
                    self.speak("Lost the connection to agentbox. Reconnecting.");
                }
            }
            LinkNotice::Fatal(reason) => self.speak(format!("Cannot continue: {reason}")),
            LinkNotice::Event(_) => {}
        }
    }

    /// The "stop" fast path: no LLM, straight to an interrupt.
    pub(crate) async fn interrupt_now(&self) {
        self.st().force = None;
        match self.link.interrupt().await {
            Ok(()) => self.speak("Interrupted."),
            Err(e) => self.speak(format!("I could not interrupt: {e}.")),
        }
    }

    /// Handles one transcript from the operator.
    pub(crate) async fn on_transcript(&self, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        // A refused send is waiting on a yes/no.
        let pending_force = self.st().force.take();
        if let Some(pf) = pending_force {
            if is_affirmative(text) {
                match self.link.send(&pf.text, pf.submit, true).await {
                    Ok(()) => self.speak("Sent."),
                    Err(e) => self.speak(format!("That failed: {e}.")),
                }
                return;
            }
            if is_negative(text) {
                self.speak("Okay, not sending.");
                return;
            }
        }
        self.run_turn(text).await;
    }

    fn context_note(&self) -> String {
        let st = self.st();
        let mut note = String::new();
        match &st.focus {
            Some(f) => note.push_str(&format!(
                "Focused pane: {} ({}, tab {} '{}').\n",
                f.pane,
                agent_name(f.agent),
                f.window,
                f.window_name
            )),
            None => note.push_str("Focused pane: unknown.\n"),
        }
        if let Some(p) = &st.prompt {
            let opts: Vec<String> = p.options.iter().map(|(n, l)| format!("{n}) {l}")).collect();
            let background = st.focus.as_ref().is_none_or(|f| f.pane != p.pane);
            note.push_str(&format!(
                "Pending prompt on pane {} ({}{}{}): \"{}\". Options: {}.\nScreen tail:\n{}\n",
                p.pane,
                agent_name(p.agent),
                p.window.map(|w| format!(", tab {w}")).unwrap_or_default(),
                if background {
                    ", NOT focused: call focus_pane first"
                } else {
                    ", focused"
                },
                p.question,
                if opts.is_empty() {
                    "none detected".to_string()
                } else {
                    opts.join("  ")
                },
                tail_chars(&p.screen, 1200),
            ));
        } else {
            note.push_str("No pending prompt.\n");
        }
        note
    }

    async fn run_turn(&self, text: &str) {
        let mut turn: Vec<ChatMessage> = vec![user_msg(text)];
        let tools = tool_specs();
        let mut finished = false;
        for _ in 0..MAX_ROUNDS {
            let mut messages = vec![ChatMessage {
                role: Role::System,
                content: Some(format!("{SYSTEM_PROMPT}\n\n{}", self.context_note())),
                tool_calls: vec![],
                tool_call_id: None,
            }];
            messages.extend(self.st().history.iter().flatten().cloned());
            messages.extend(turn.iter().cloned());
            let req = ChatRequest {
                messages,
                tools: tools.clone(),
                temperature: 0.2,
                max_tokens: 1536,
            };
            let reply = match self.llm.chat(&req).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("concierge llm error: {e}");
                    self.speak("I could not reach the language model.");
                    return;
                }
            };
            let calls = reply.tool_calls.clone();
            let content = reply.content.clone().unwrap_or_default();
            turn.push(ChatMessage {
                role: Role::Assistant,
                ..reply
            });
            if calls.is_empty() {
                self.speak(content);
                finished = true;
                break;
            }
            let mut halted = false;
            for call in &calls {
                let outcome = self.exec_tool(call).await;
                let (result, halt) = match outcome {
                    Outcome::Continue(r) => (r, false),
                    Outcome::Halt(r) => (r, true),
                };
                turn.push(ChatMessage {
                    role: Role::Tool,
                    content: Some(result),
                    tool_calls: vec![],
                    tool_call_id: Some(call.id.clone()),
                });
                if halt {
                    halted = true;
                    break;
                }
            }
            if halted {
                // The tool already asked the operator a question; drop the unanswered calls.
                finished = true;
                // Keep the transcript well-formed: every call needs a reply.
                let answered: Vec<String> =
                    turn.iter().filter_map(|m| m.tool_call_id.clone()).collect();
                for call in calls.iter().filter(|c| !answered.contains(&c.id)) {
                    turn.push(ChatMessage {
                        role: Role::Tool,
                        content: Some("not run".into()),
                        tool_calls: vec![],
                        tool_call_id: Some(call.id.clone()),
                    });
                }
                break;
            }
        }
        if !finished {
            // Out of rounds with the model still calling tools: say so rather than go quiet.
            self.speak("Done.");
        }
        self.commit_turn(turn);
    }

    fn commit_turn(&self, turn: Vec<ChatMessage>) {
        let mut st = self.st();
        st.history.push_back(turn);
        let keep = usize::from(self.cfg.history_turns).max(1);
        while st.history.len() > keep {
            st.history.pop_front();
        }
    }

    async fn exec_tool(&self, call: &ToolCall) -> Outcome {
        let args = &call.arguments;
        let str_arg = |k: &str| {
            args.get(k)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
        };
        match call.name.as_str() {
            "list_panes" => match self.link.list_panes().await {
                Ok(panes) => {
                    let lines: Vec<String> = panes
                        .iter()
                        .map(|p| {
                            format!(
                                "{} | tab {} '{}' | {} | {}",
                                p.pane,
                                p.window,
                                p.window_name,
                                p.command,
                                if p.focused { "focused" } else { "background" }
                            )
                        })
                        .collect();
                    self.st().panes = panes;
                    Outcome::Continue(lines.join("\n"))
                }
                Err(e) => Outcome::Continue(format!("error: {e}")),
            },
            "read_pane" => {
                let Some(pane) = str_arg("pane") else {
                    return Outcome::Continue("error: pane is required".into());
                };
                let lines = args
                    .get("lines")
                    .and_then(Value::as_u64)
                    .unwrap_or(40)
                    .clamp(1, 200) as u32;
                match self.link.read_pane(pane, lines).await {
                    Ok(t) => Outcome::Continue(tail_chars(&t, MAX_TOOL_TEXT)),
                    Err(e) => Outcome::Continue(format!("error: {e}")),
                }
            }
            "send_to_focused" => {
                let Some(text) = str_arg("text") else {
                    return Outcome::Continue("error: text is required".into());
                };
                let submit = args.get("submit").and_then(Value::as_bool).unwrap_or(true);
                let focus = self.st().focus.clone();
                if self.cfg.announce_actions {
                    let who = focus.as_ref().map_or("the agent", |f| agent_name(f.agent));
                    self.speak(format!("Sending to {who}: {text}"));
                }
                match self.link.send(text, submit, false).await {
                    Ok(()) => {
                        self.st().prompt = None;
                        Outcome::Continue("sent".into())
                    }
                    Err(LinkError::Controller {
                        code: ErrorCode::InputNotEmpty,
                        ..
                    }) => {
                        self.st().force = Some(PendingForce {
                            text: text.to_string(),
                            submit,
                        });
                        self.speak("The prompt box already has text. Send anyway?");
                        Outcome::Halt("not sent: the input box already holds text; the operator was asked whether to send anyway".into())
                    }
                    Err(e) => Outcome::Continue(format!("error: {e}")),
                }
            }
            "press_keys" => {
                let Some(names) = args.get("keys").and_then(Value::as_array) else {
                    return Outcome::Continue("error: keys must be an array of strings".into());
                };
                let mut keys = Vec::new();
                for n in names {
                    match n.as_str().and_then(parse_key) {
                        Some(k) => keys.push(k),
                        None => {
                            return Outcome::Continue(format!(
                                "error: unknown key {n}; use enter, esc, tab, shift+tab, space, up, down, left, right, a digit or a letter"
                            ));
                        }
                    }
                }
                if keys.is_empty() {
                    return Outcome::Continue("error: no keys given".into());
                }
                if self.cfg.announce_actions {
                    self.speak(describe_keys(&keys));
                }
                match self.link.keys(keys).await {
                    Ok(()) => {
                        self.st().prompt = None;
                        Outcome::Continue("pressed".into())
                    }
                    Err(e) => Outcome::Continue(format!("error: {e}")),
                }
            }
            "interrupt" => {
                if self.cfg.announce_actions {
                    self.speak("Interrupting.");
                }
                match self.link.interrupt().await {
                    Ok(()) => Outcome::Continue("interrupted".into()),
                    Err(e) => Outcome::Continue(format!("error: {e}")),
                }
            }
            "focus_pane" => {
                let Some(pane) = str_arg("pane") else {
                    return Outcome::Continue("error: pane is required".into());
                };
                if self.cfg.announce_actions {
                    self.speak(self.focus_announcement(pane));
                }
                match self.link.select_pane(pane).await {
                    Ok(()) => Outcome::Continue(format!("focus is now {pane}")),
                    Err(e) => Outcome::Continue(format!("error: {e}")),
                }
            }
            other => Outcome::Continue(format!("error: unknown tool {other}")),
        }
    }

    fn focus_announcement(&self, pane: &str) -> String {
        let st = self.st();
        if let Some(p) = st.prompt.as_ref().filter(|p| p.pane == pane) {
            return match p.window {
                Some(w) => format!("Switching to {} in tab {w}", agent_name(p.agent)),
                None => format!("Switching to {}", agent_name(p.agent)),
            };
        }
        match st.panes.iter().find(|p| p.pane == pane) {
            Some(p) => format!("Switching to {} in tab {}", agent_name(p.agent), p.window),
            None => "Switching focus".to_string(),
        }
    }

    /// Handles an agent event from the spool.
    pub(crate) async fn on_event(&self, ev: AgentEvent) {
        let focused = self.st().focus.as_ref().is_some_and(|f| f.pane == ev.pane);
        match ev.kind {
            EventKind::TurnEnd => {
                {
                    let mut st = self.st();
                    if st.prompt.as_ref().is_some_and(|p| p.pane == ev.pane) {
                        st.prompt = None;
                    }
                }
                if focused {
                    if self.cfg.speak_turn_end {
                        self.summarise_turn(&ev).await;
                    }
                } else if self.cfg.announce_background {
                    let place = self.place(&ev).await;
                    self.speak(format!("{} {place} has finished.", agent_name(ev.agent)));
                }
            }
            EventKind::Prompt => self.on_prompt(ev, focused).await,
            EventKind::Notice => tracing::info!("agent notice from {}: {:?}", ev.pane, ev.text),
        }
    }

    /// "in tab 2, Services," style location for an announcement.
    async fn place(&self, ev: &AgentEvent) -> String {
        let known = |st: &State| {
            st.panes
                .iter()
                .find(|p| p.pane == ev.pane)
                .map(|p| (p.window, p.window_name.clone()))
        };
        let mut found = known(&self.st());
        if found.is_none() {
            if let Ok(panes) = self.link.list_panes().await {
                self.st().panes = panes;
                found = known(&self.st());
            }
        }
        match (found, ev.window) {
            (Some((w, name)), _) if !name.is_empty() => format!("in tab {w}, {name},"),
            (Some((w, _)), _) | (None, Some(w)) => format!("in tab {w}"),
            (None, None) => "in another tab".to_string(),
        }
    }

    async fn summarise_turn(&self, ev: &AgentEvent) {
        let who = agent_name(ev.agent);
        let mut body = ev.text.clone().unwrap_or_default();
        if body.trim().is_empty() {
            body = self.link.read_pane(&ev.pane, 40).await.unwrap_or_default();
        }
        if body.trim().is_empty() {
            self.speak(format!("{who} has finished."));
            return;
        }
        let words = self.cfg.summary_words.max(5);
        let req = ChatRequest {
            messages: vec![
                ChatMessage {
                    role: Role::System,
                    content: Some(format!(
                        "You turn an AI coding agent's final message into one spoken update of at most {words} words. \
                         Plain sentences only: no markdown, no code, no lists. Lead with the outcome, then anything the \
                         operator must decide. Do not mention that you are summarising."
                    )),
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                user_msg(&tail_chars(&body, 6000)),
            ],
            tools: vec![],
            temperature: 0.2,
            max_tokens: 1536,
        };
        match self.llm.chat(&req).await {
            Ok(r) if r.content.as_deref().is_some_and(|c| !c.trim().is_empty()) => {
                self.speak(format!("{who}: {}", r.content.unwrap_or_default().trim()));
            }
            Ok(_) => self.speak(format!("{who} has finished.")),
            Err(e) => {
                tracing::warn!("summary failed: {e}");
                self.speak(format!("{who} has finished."));
            }
        }
    }

    async fn on_prompt(&self, ev: AgentEvent, focused: bool) {
        let screen = self
            .link
            .read_pane(&ev.pane, PROMPT_LINES)
            .await
            .unwrap_or_default();
        let options = parse_options(&screen);
        let question = ev
            .text
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| guess_question(&screen));
        let pending = PendingPrompt {
            pane: ev.pane.clone(),
            window: ev.window,
            agent: ev.agent,
            question: question.clone(),
            options: options.clone(),
            screen,
        };
        self.st().prompt = Some(pending);
        if !focused && !self.cfg.announce_background {
            return;
        }
        let who = agent_name(ev.agent);
        let lead = if focused {
            format!("{who} is asking")
        } else {
            format!("{who} {} is asking", self.place(&ev).await)
        };
        let mut line = if question.is_empty() {
            format!("{lead} for input.")
        } else {
            format!("{lead}: {question}")
        };
        if !line.ends_with(['.', '?', '!']) {
            line.push('.');
        }
        if !options.is_empty() {
            let spoken: Vec<String> = options
                .iter()
                .take(6)
                .map(|(n, l)| format!("option {n}: {}", tail_chars(l, 80)))
                .collect();
            line.push(' ');
            line.push_str(&spoken.join(". "));
            line.push('.');
        }
        self.speak(line);
    }
}

fn user_msg(text: &str) -> ChatMessage {
    ChatMessage {
        role: Role::User,
        content: Some(text.to_string()),
        tool_calls: vec![],
        tool_call_id: None,
    }
}

#[cfg(test)]
pub(crate) mod fake {
    //! A scripted chat model for tests.

    use async_trait::async_trait;
    use voicelink_backends::{BackendError, Result};

    use super::*;

    /// Replays queued replies and records every request.
    #[derive(Default)]
    pub(crate) struct ScriptedLlm {
        pub replies: Mutex<VecDeque<ChatMessage>>,
        pub requests: Mutex<Vec<ChatRequest>>,
    }

    impl ScriptedLlm {
        pub(crate) fn push(&self, m: ChatMessage) {
            self.replies.lock().unwrap().push_back(m);
        }
        pub(crate) fn request_count(&self) -> usize {
            self.requests.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl ChatModel for ScriptedLlm {
        async fn chat(&self, request: &ChatRequest) -> Result<ChatMessage> {
            self.requests.lock().unwrap().push(request.clone());
            self.replies
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| BackendError::Backend("script exhausted".into()))
        }
        fn name(&self) -> String {
            "scripted".into()
        }
    }

    pub(crate) fn say(text: &str) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            content: Some(text.into()),
            tool_calls: vec![],
            tool_call_id: None,
        }
    }

    pub(crate) fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    pub(crate) fn calls(list: Vec<ToolCall>) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            content: None,
            tool_calls: list,
            tool_call_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use voicelink_proto::ClientMessage;

    use super::fake::*;
    use super::*;
    use crate::link::fake::{FakeController, fast_config, focus};
    use crate::link::spawn;

    struct Rig {
        fc: FakeController,
        llm: Arc<ScriptedLlm>,
        concierge: Arc<Concierge>,
        said: mpsc::UnboundedReceiver<String>,
        _notices: mpsc::UnboundedReceiver<LinkNotice>,
        _task: tokio::task::JoinHandle<()>,
    }

    async fn rig(cfg: config::Concierge) -> Rig {
        let fc = FakeController::new(focus(1, "main", "%1", "claude"));
        let (link, mut notices, task) = spawn(fc.clone(), fast_config());
        let llm = Arc::new(ScriptedLlm::default());
        let (say_tx, said) = mpsc::unbounded_channel();
        let concierge = Concierge::new(llm.clone(), link, say_tx, cfg);
        // Establish the session and feed the initial focus in, as the daemon does.
        let first = tokio::time::timeout(Duration::from_secs(2), notices.recv())
            .await
            .unwrap()
            .unwrap();
        concierge.on_notice(first);
        Rig {
            fc,
            llm,
            concierge,
            said,
            _notices: notices,
            _task: task,
        }
    }

    fn drain(said: &mut mpsc::UnboundedReceiver<String>) -> Vec<String> {
        let mut out = Vec::new();
        while let Ok(s) = said.try_recv() {
            out.push(s);
        }
        out
    }

    fn event(
        kind: EventKind,
        pane: &str,
        window: u32,
        agent: AgentKind,
        text: Option<&str>,
    ) -> AgentEvent {
        AgentEvent {
            ts_ms: 1,
            pane: pane.into(),
            window: Some(window),
            agent,
            kind,
            text: text.map(String::from),
        }
    }

    #[tokio::test]
    async fn spoken_request_becomes_a_send_and_a_spoken_confirmation() {
        let mut r = rig(config::Concierge::default()).await;
        r.llm.push(calls(vec![call(
            "c1",
            "send_to_focused",
            json!({"text": "Fix the failing build in the api crate"}),
        )]));
        r.llm.push(say("Sent the build fix request to Claude."));
        r.concierge
            .on_transcript("um fix the the failing build in the API crate")
            .await;
        let log = r.fc.log();
        assert!(log.iter().any(|m| matches!(m,
            ClientMessage::Send { text, submit: true, force: false, .. } if text == "Fix the failing build in the api crate")));
        let said = drain(&mut r.said);
        assert_eq!(
            said[0],
            "Sending to Claude: Fix the failing build in the api crate"
        );
        assert_eq!(
            said.last().unwrap(),
            "Sent the build fix request to Claude."
        );
        // The tool result went back to the model on the second round.
        let reqs = r.llm.requests.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        assert!(
            reqs[1]
                .messages
                .iter()
                .any(|m| m.role == Role::Tool && m.content.as_deref() == Some("sent"))
        );
        // The user's words were passed through to the model.
        assert!(
            reqs[0].messages.iter().any(
                |m| m.role == Role::User && m.content.as_deref().unwrap().contains("API crate")
            )
        );
    }

    #[tokio::test]
    async fn announcements_can_be_switched_off() {
        let mut r = rig(config::Concierge {
            announce_actions: false,
            ..config::Concierge::default()
        })
        .await;
        r.llm.push(calls(vec![call(
            "c1",
            "send_to_focused",
            json!({"text": "hello"}),
        )]));
        r.llm.push(say("Done."));
        r.concierge.on_transcript("say hello").await;
        assert_eq!(drain(&mut r.said), vec!["Done.".to_string()]);
    }

    #[tokio::test]
    async fn background_prompt_is_announced_and_approve_selects_the_pane_before_pressing_keys() {
        let mut r = rig(config::Concierge::default()).await;
        r.fc.script.lock().unwrap().pane_text.insert(
            "%5".into(),
            "Allow command: cargo test?\n\n❯ 1. Yes\n  2. Yes, and don't ask again for cargo\n  3. No\n".into(),
        );
        r.fc.script.lock().unwrap().panes = vec![
            PaneInfo {
                pane: "%1".into(),
                window: 1,
                window_name: "main".into(),
                command: "claude".into(),
                agent: AgentKind::Claude,
                active: true,
                focused: true,
            },
            PaneInfo {
                pane: "%5".into(),
                window: 2,
                window_name: "Services".into(),
                command: "codex".into(),
                agent: AgentKind::Codex,
                active: true,
                focused: false,
            },
        ];
        r.concierge
            .on_event(event(
                EventKind::Prompt,
                "%5",
                2,
                AgentKind::Codex,
                Some("Allow command: cargo test?"),
            ))
            .await;
        let said = drain(&mut r.said);
        assert_eq!(said.len(), 1, "{said:?}");
        let line = &said[0];
        assert!(
            line.starts_with("Codex in tab 2, Services, is asking: Allow command: cargo test?"),
            "{line}"
        );
        assert!(
            line.contains("option 1: Yes") && line.contains("option 3: No"),
            "{line}"
        );

        // "approve": the model (scripted) focuses the pane, then presses 1.
        r.llm.push(calls(vec![
            call("a", "focus_pane", json!({"pane": "%5"})),
            call("b", "press_keys", json!({"keys": ["1", "enter"]})),
        ]));
        r.llm.push(say("Approved the cargo test in tab 2."));
        r.concierge.on_transcript("approve").await;

        let log = r.fc.log();
        let select = log
            .iter()
            .position(|m| matches!(m, ClientMessage::SelectPane { pane, .. } if pane == "%5"));
        let keys = log.iter().position(|m| matches!(m, ClientMessage::Keys { keys, .. } if keys == &[Key::Digit(1), Key::Enter]));
        assert!(select.is_some() && keys.is_some(), "{log:?}");
        assert!(
            select < keys,
            "SelectPane must reach the controller before Keys: {log:?}"
        );

        // The pending prompt reached the model as context, flagged as not focused.
        let reqs = r.llm.requests.lock().unwrap();
        let sys = reqs[0].messages[0].content.clone().unwrap();
        assert!(
            sys.contains("Pending prompt on pane %5") && sys.contains("NOT focused"),
            "{sys}"
        );
        assert!(sys.contains("1) Yes") && sys.contains("3) No"), "{sys}");
        drop(reqs);

        let said = drain(&mut r.said);
        assert_eq!(said[0], "Switching to Codex in tab 2");
        assert_eq!(said[1], "Choosing option one");
        assert_eq!(said.last().unwrap(), "Approved the cargo test in tab 2.");
        // Answered: the prompt is no longer pending.
        assert!(r.concierge.context_note().contains("No pending prompt"));
    }

    #[tokio::test]
    async fn background_turn_end_is_announced_with_agent_and_window() {
        let mut r = rig(config::Concierge::default()).await;
        r.concierge
            .on_event(event(
                EventKind::TurnEnd,
                "%5",
                2,
                AgentKind::Codex,
                Some("done"),
            ))
            .await;
        let said = drain(&mut r.said);
        assert_eq!(said.len(), 1);
        assert!(
            said[0].starts_with("Codex in tab 2") && said[0].ends_with("has finished."),
            "{said:?}"
        );
        assert_eq!(
            r.llm.request_count(),
            0,
            "no LLM needed for a background announcement"
        );
    }

    #[tokio::test]
    async fn background_announcements_respect_the_setting() {
        let mut r = rig(config::Concierge {
            announce_background: false,
            ..config::Concierge::default()
        })
        .await;
        r.concierge
            .on_event(event(
                EventKind::TurnEnd,
                "%5",
                2,
                AgentKind::Codex,
                Some("done"),
            ))
            .await;
        assert!(drain(&mut r.said).is_empty());
    }

    #[tokio::test]
    async fn focused_turn_end_is_summarised_within_the_word_budget() {
        let mut r = rig(config::Concierge {
            summary_words: 25,
            ..config::Concierge::default()
        })
        .await;
        r.llm.push(say("Build fixed and all tests pass."));
        let long = "I investigated the failure. ".repeat(40);
        r.concierge
            .on_event(event(
                EventKind::TurnEnd,
                "%1",
                1,
                AgentKind::Claude,
                Some(&long),
            ))
            .await;
        assert_eq!(
            drain(&mut r.said),
            vec!["Claude: Build fixed and all tests pass.".to_string()]
        );
        let reqs = r.llm.requests.lock().unwrap();
        assert!(reqs[0].tools.is_empty());
        assert!(
            reqs[0].messages[0]
                .content
                .as_deref()
                .unwrap()
                .contains("at most 25 words")
        );
        assert!(
            reqs[0].messages[1]
                .content
                .as_deref()
                .unwrap()
                .contains("I investigated")
        );
    }

    #[tokio::test]
    async fn turn_end_summary_can_be_switched_off() {
        let mut r = rig(config::Concierge {
            speak_turn_end: false,
            ..config::Concierge::default()
        })
        .await;
        r.concierge
            .on_event(event(
                EventKind::TurnEnd,
                "%1",
                1,
                AgentKind::Claude,
                Some("done"),
            ))
            .await;
        assert!(drain(&mut r.said).is_empty());
        assert_eq!(r.llm.request_count(), 0);
    }

    #[tokio::test]
    async fn bare_stop_through_the_run_loop_interrupts_without_the_llm() {
        let mut r = rig(config::Concierge::default()).await;
        let (tx, rx) = mpsc::unbounded_channel();
        let run = tokio::spawn(Arc::clone(&r.concierge).run(rx));
        // Whisper-style punctuation and case.
        tx.send(Input::Transcript("Stop.".into())).unwrap();
        for _ in 0..50 {
            if r.fc
                .log()
                .iter()
                .any(|m| matches!(m, ClientMessage::Interrupt { .. }))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            r.fc.log()
                .iter()
                .any(|m| matches!(m, ClientMessage::Interrupt { .. }))
        );
        assert_eq!(r.llm.request_count(), 0);
        let mut said = Vec::new();
        for _ in 0..50 {
            said = drain(&mut r.said);
            if !said.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(said, vec!["Interrupted.".to_string()]);
        drop(tx);
        run.await.unwrap();
    }

    #[test]
    fn stop_words_are_exact_not_substrings() {
        for yes in ["stop", "Stop.", "  INTERRUPT! ", "cancel", "Please stop"] {
            assert!(is_stop_word(yes), "{yes}");
        }
        for no in [
            "stop the build and rerun it",
            "don't stop",
            "stopwatch",
            "",
            "go on",
        ] {
            assert!(!is_stop_word(no), "{no}");
        }
    }

    #[tokio::test]
    async fn input_not_empty_asks_then_yes_forces_the_send() {
        let mut r = rig(config::Concierge::default()).await;
        r.fc.script.lock().unwrap().input_not_empty = true;
        r.llm.push(calls(vec![call(
            "c1",
            "send_to_focused",
            json!({"text": "run the tests"}),
        )]));
        r.concierge.on_transcript("run the tests").await;
        let said = drain(&mut r.said);
        assert_eq!(
            said.last().unwrap(),
            "The prompt box already has text. Send anyway?"
        );
        assert_eq!(
            r.llm.request_count(),
            1,
            "the question is asked without another model round"
        );

        r.concierge.on_transcript("Yes.").await;
        let log = r.fc.log();
        assert!(log.iter().any(|m| matches!(m,
            ClientMessage::Send { text, force: true, submit: true, .. } if text == "run the tests")), "{log:?}");
        assert_eq!(
            r.llm.request_count(),
            1,
            "the answer is resolved deterministically"
        );
        assert_eq!(drain(&mut r.said), vec!["Sent.".to_string()]);
    }

    #[tokio::test]
    async fn cancel_answers_the_send_anyway_question_instead_of_interrupting() {
        let mut r = rig(config::Concierge::default()).await;
        r.fc.script.lock().unwrap().input_not_empty = true;
        r.llm.push(calls(vec![call(
            "c1",
            "send_to_focused",
            json!({"text": "run the tests"}),
        )]));
        let (tx, rx) = mpsc::unbounded_channel();
        let run = tokio::spawn(Arc::clone(&r.concierge).run(rx));
        tx.send(Input::Transcript("run the tests".into())).unwrap();
        for _ in 0..100 {
            if r.concierge.st().force.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(r.concierge.st().force.is_some());
        tx.send(Input::Transcript("Cancel.".into())).unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        tx.send(Input::Transcript("yes".into())).unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        let log = r.fc.log();
        assert!(
            !log.iter()
                .any(|m| matches!(m, ClientMessage::Interrupt { .. })),
            "{log:?}"
        );
        assert!(
            !log.iter()
                .any(|m| matches!(m, ClientMessage::Send { force: true, .. })),
            "{log:?}"
        );
        assert!(drain(&mut r.said).contains(&"Okay, not sending.".to_string()));
        drop(tx);
        run.await.unwrap();
    }

    #[tokio::test]
    async fn input_not_empty_then_no_drops_the_send() {
        let mut r = rig(config::Concierge::default()).await;
        r.fc.script.lock().unwrap().input_not_empty = true;
        r.llm.push(calls(vec![call(
            "c1",
            "send_to_focused",
            json!({"text": "run the tests"}),
        )]));
        r.concierge.on_transcript("run the tests").await;
        drain(&mut r.said);
        r.concierge.on_transcript("no").await;
        assert_eq!(drain(&mut r.said), vec!["Okay, not sending.".to_string()]);
        assert!(
            !r.fc
                .log()
                .iter()
                .any(|m| matches!(m, ClientMessage::Send { force: true, .. }))
        );
    }

    #[tokio::test]
    async fn unknown_key_is_reported_to_the_model_not_sent() {
        let mut r = rig(config::Concierge::default()).await;
        r.llm.push(calls(vec![call(
            "c1",
            "press_keys",
            json!({"keys": ["hyper+x"]}),
        )]));
        r.llm.push(say("Sorry, I could not press that."));
        r.concierge.on_transcript("press hyper x").await;
        assert!(
            !r.fc
                .log()
                .iter()
                .any(|m| matches!(m, ClientMessage::Keys { .. }))
        );
        let reqs = r.llm.requests.lock().unwrap();
        assert!(
            reqs[1]
                .messages
                .iter()
                .any(|m| m.role == Role::Tool
                    && m.content.as_deref().unwrap().contains("unknown key"))
        );
        drop(reqs);
        assert_eq!(
            drain(&mut r.said).last().unwrap(),
            "Sorry, I could not press that."
        );
    }

    #[tokio::test]
    async fn tool_loop_is_bounded_to_four_rounds() {
        let mut r = rig(config::Concierge::default()).await;
        for i in 0..10 {
            r.llm
                .push(calls(vec![call(&format!("c{i}"), "list_panes", json!({}))]));
        }
        r.concierge.on_transcript("look around forever").await;
        assert_eq!(r.llm.request_count(), MAX_ROUNDS);
        assert_eq!(drain(&mut r.said).last().unwrap(), "Done.");
    }

    #[tokio::test]
    async fn history_is_bounded_by_whole_turns_and_keeps_tool_pairs_together() {
        let r = rig(config::Concierge {
            history_turns: 2,
            ..config::Concierge::default()
        })
        .await;
        for i in 0..4 {
            r.llm
                .push(calls(vec![call(&format!("t{i}"), "list_panes", json!({}))]));
            r.llm.push(say(&format!("answer {i}")));
            r.concierge.on_transcript(&format!("question {i}")).await;
        }
        let st = r.concierge.st();
        assert_eq!(st.history.len(), 2);
        let flat: Vec<&ChatMessage> = st.history.iter().flatten().collect();
        assert_eq!(flat[0].content.as_deref(), Some("question 2"));
        // Every tool reply has its assistant tool_calls message in the kept window.
        for (i, m) in flat.iter().enumerate() {
            if m.role == Role::Tool {
                let id = m.tool_call_id.as_deref().unwrap();
                assert!(
                    flat[..i]
                        .iter()
                        .any(|a| a.tool_calls.iter().any(|c| c.id == id))
                );
            }
        }
    }

    #[tokio::test]
    async fn llm_failure_is_spoken_not_silent() {
        let mut r = rig(config::Concierge::default()).await;
        r.concierge.on_transcript("hello there").await;
        assert_eq!(
            drain(&mut r.said),
            vec!["I could not reach the language model.".to_string()]
        );
    }

    #[tokio::test]
    async fn focus_pushes_update_context_and_cancel_a_stale_force_question() {
        let r = rig(config::Concierge::default()).await;
        r.concierge.st().force = Some(PendingForce {
            text: "x".into(),
            submit: true,
        });
        r.concierge
            .on_notice(LinkNotice::Focus(focus(3, "logs", "%9", "codex")));
        assert!(r.concierge.st().force.is_none());
        assert!(r.concierge.context_note().contains("%9"));
    }

    #[tokio::test]
    async fn connection_loss_and_recovery_are_announced_once() {
        let mut r = rig(config::Concierge::default()).await;
        r.concierge.on_notice(LinkNotice::Disconnected("x".into()));
        r.concierge.on_notice(LinkNotice::Disconnected("y".into()));
        r.concierge
            .on_notice(LinkNotice::Connected(focus(1, "main", "%1", "claude")));
        assert_eq!(
            drain(&mut r.said),
            vec![
                "Lost the connection to agentbox. Reconnecting.".to_string(),
                "Reconnected to agentbox.".to_string()
            ]
        );
    }

    #[test]
    fn key_names_map_to_wire_keys() {
        assert_eq!(parse_key("Enter"), Some(Key::Enter));
        assert_eq!(parse_key("esc"), Some(Key::Escape));
        assert_eq!(parse_key("shift+tab"), Some(Key::BackTab));
        assert_eq!(parse_key("2"), Some(Key::Digit(2)));
        assert_eq!(parse_key("two"), Some(Key::Digit(2)));
        assert_eq!(parse_key("y"), Some(Key::Letter('y')));
        assert_eq!(parse_key("ctrl+c"), None);
        assert_eq!(parse_key("12"), None);
        assert_eq!(parse_key(""), None);
    }

    #[test]
    fn option_parser_handles_claude_and_codex_menus_and_stale_menus() {
        let claude = "Do you want to proceed?\n❯ 1. Yes\n  2. Yes, and don't ask again\n  3. No, and tell Claude what to do\n";
        let o = parse_options(claude);
        assert_eq!(o.len(), 3);
        assert_eq!(o[1], (2, "Yes, and don't ask again".to_string()));
        let boxed = "│ ❯ 1. Yes   │\n│   2. No    │\n";
        assert_eq!(
            parse_options(boxed),
            vec![(1, "Yes".to_string()), (2, "No".to_string())]
        );
        // An older menu above a newer one: only the newest counts.
        let stale = "1. old a\n2. old b\n3. old c\nnoise\n1. new a\n2. new b\n";
        assert_eq!(parse_options(stale).len(), 2);
        assert!(parse_options("version 2.1 released\nno menu").is_empty());
        assert_eq!(guess_question(claude), "Do you want to proceed?");
    }

    #[test]
    fn key_announcements_read_naturally() {
        assert_eq!(
            describe_keys(&[Key::Digit(2), Key::Enter]),
            "Choosing option two"
        );
        assert_eq!(describe_keys(&[Key::Enter]), "Pressing enter");
        assert_eq!(describe_keys(&[Key::Letter('y')]), "Pressing y");
    }
}
