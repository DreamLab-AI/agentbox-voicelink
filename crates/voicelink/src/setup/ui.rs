//! Full-screen rendering and the event loop around [`Wizard`].

use std::sync::mpsc;
use std::time::Duration;

use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, ListState, Paragraph, Wrap};

use super::effects::{self, Controls};
use super::wizard::{Finish, Key, Step, Wizard};

/// Restores the terminal when dropped, including on early return or panic
/// unwinding (`ratatui::init` also installs a panic hook that restores it).
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        ratatui::restore();
    }
}

fn map_key(code: KeyCode, mods: KeyModifiers) -> Option<Key> {
    Some(match code {
        KeyCode::Char('c') if mods.contains(KeyModifiers::CONTROL) => Key::CtrlC,
        KeyCode::Char(c) => Key::Char(c),
        KeyCode::Up => Key::Up,
        KeyCode::Down => Key::Down,
        KeyCode::Left => Key::Left,
        KeyCode::Right => Key::Right,
        KeyCode::Enter => Key::Enter,
        KeyCode::Esc => Key::Esc,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Tab => Key::Tab,
        KeyCode::BackTab => Key::BackTab,
        _ => return None,
    })
}

/// Runs the wizard on the terminal until it finishes.
pub fn run(mut wizard: Wizard) -> anyhow::Result<Finish> {
    let mut terminal = ratatui::init();
    let _guard = TerminalGuard;
    let (tx, rx) = mpsc::channel();
    let ctl = Controls::default();
    for e in wizard.start() {
        effects::spawn(e, tx.clone(), &ctl);
    }
    while wizard.finished.is_none() {
        terminal.draw(|f| draw(f, &wizard))?;
        if event::poll(Duration::from_millis(50))?
            && let Event::Key(k) = event::read()?
            && k.kind == KeyEventKind::Press
            && let Some(key) = map_key(k.code, k.modifiers)
        {
            for e in wizard.key(key) {
                effects::spawn(e, tx.clone(), &ctl);
            }
        }
        while let Ok(o) = rx.try_recv() {
            wizard.apply(o);
        }
    }
    ctl.meter_stop
        .store(true, std::sync::atomic::Ordering::Relaxed);
    ctl.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(wizard.finished.take().unwrap_or(Finish::Cancelled))
}

fn level_bar(db: f32) -> String {
    let filled = (((db + 60.0) / 60.0).clamp(0.0, 1.0) * 30.0) as usize;
    format!(
        "[{}{}] {db:.0} dBFS",
        "#".repeat(filled),
        " ".repeat(30 - filled)
    )
}

impl Wizard {
    /// Explanatory text for the current page.
    pub fn info(&self) -> Vec<String> {
        let macos = self
            .hw
            .as_ref()
            .map_or(cfg!(target_os = "macos"), |h| h.os == "macos");
        let mut v: Vec<String> = Vec::new();
        match self.step {
            Step::Hardware => match &self.proposal {
                Some(p) => v.extend(p.evidence.clone()),
                None => v.push("Probing hardware...".into()),
            },
            Step::Speech => {
                v.push("Embedded builds are CPU, Vulkan (Linux) or Metal (macOS). ROCm and CUDA are reached through an HTTP model server instead.".into());
                v.push("Remote Nemotron runs on the agentbox host and is reached through an ssh forward (127.0.0.1:18897).".into());
            }
            Step::Llm => {
                v.push("The concierge model is always an HTTP server (it outlives this client). It must return structured tool calls.".into());
                for d in &self.detected {
                    v.push(format!("found {} at {} ({} models)", d.label, d.base_url, d.models.len()));
                }
                if self.detect_done && self.detected.is_empty() {
                    v.push("No server found on :8080, :11434 or :1234. Start llama-server with --jinja, use the compose stack, or pick the Loom.".into());
                }
                if self.choices.llm_kind == super::choices::LlmKind::Loom {
                    v.push("The Loom is reached through an ssh forward: 127.0.0.1:18084 -> 127.0.0.1:8084 on the host (scaffold off).".into());
                }
            }
            Step::Voice => v.push("Pocket TTS on the agentbox host is reached through an ssh forward (127.0.0.1:18898 -> 127.0.0.1:8898, /v1/audio/speech).".into()),
            Step::Models => {
                let need = self.choices.required_models();
                if need.is_empty() {
                    v.push("The chosen backends need no local model files.".into());
                }
                for e in need {
                    let have = self.choices.models_dir.join(e.file).exists();
                    v.push(format!("{} {} [{}]", if have { "present" } else { "missing" }, e.label, e.licence));
                }
                v.push(format!("into {} (resumable, SHA-256 verified)", self.choices.models_dir.display()));
            }
            Step::Audio => {
                if let Some(db) = self.level_db {
                    v.push(level_bar(db));
                }
                if macos {
                    v.push("macOS: the microphone permission is granted to the terminal app you run voicelink in (System Settings > Privacy & Security > Microphone).".into());
                }
                v.push("Echo cancellation is built in; no PipeWire setup is needed.".into());
            }
            Step::Hotkey => {
                v.push("Disabled by default: a hardware mic mute is the primary control.".into());
                if macos && self.choices.hotkey_enabled {
                    v.push("macOS: a global hotkey needs Accessibility permission for the terminal app (System Settings > Privacy & Security > Accessibility).".into());
                }
            }
            Step::Connection => {
                v.push("Suggested: the LAN address above, or your tailnet name (user@host.tailnet.ts.net) when remote.".into());
                v.push("Checks run `ssh -o BatchMode=yes` (no prompts): load your key into ssh-agent first.".into());
            }
            Step::Smoke => v.push("Press Enter, then say a short phrase within 3 s. The transcript, a spoken line and latencies are shown.".into()),
            Step::Review => v.extend(self.summary()),
        }
        v
    }
}

fn draw(f: &mut Frame, w: &Wizard) {
    let area = f.area();
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(6),
        Constraint::Length(3),
    ])
    .split(area);
    let crumbs: Vec<Span> = Step::ALL
        .iter()
        .enumerate()
        .flat_map(|(i, s)| {
            if *s == w.step {
                [
                    Span::styled(
                        format!("[{}. {}]", i + 1, s.title()),
                        Style::new().add_modifier(Modifier::BOLD).fg(Color::Cyan),
                    ),
                    Span::raw(" "),
                ]
            } else {
                [
                    Span::styled((i + 1).to_string(), Style::new().fg(Color::DarkGray)),
                    Span::raw(" "),
                ]
            }
        })
        .collect();
    f.render_widget(
        Paragraph::new(Line::from(crumbs)).block(
            Block::new()
                .borders(Borders::ALL)
                .title(" voicelink setup "),
        ),
        rows[0],
    );

    let body = Layout::vertical([
        Constraint::Length(w.fields().len().max(1) as u16 + 2),
        Constraint::Min(3),
    ])
    .split(rows[1]);
    let fields = w.fields();
    let items: Vec<ListItem> = fields
        .iter()
        .enumerate()
        .map(|(i, fl)| {
            let value = match (&w.editing, i == w.cursor) {
                (Some(buf), true) => format!("{buf}_"),
                _ => fl.value.clone(),
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<28}", fl.label),
                    Style::new().add_modifier(Modifier::BOLD),
                ),
                Span::raw(value),
            ]))
        })
        .collect();
    let mut state = ListState::default();
    state.select((!fields.is_empty()).then_some(w.cursor));
    f.render_stateful_widget(
        List::new(items)
            .block(
                Block::new()
                    .borders(Borders::ALL)
                    .title(format!(" {} ", w.step.title())),
            )
            .highlight_style(Style::new().bg(Color::Blue))
            .highlight_symbol("> "),
        body[0],
        &mut state,
    );

    let mut text: Vec<Line> = w.info().into_iter().map(Line::from).collect();
    if !w.notes.is_empty() {
        text.push(Line::default());
        text.extend(w.notes.iter().map(|n| {
            Line::styled(
                n.clone(),
                Style::new().fg(
                    if n.starts_with("WARNING") || n.contains("FAIL") || n.contains("failed") {
                        Color::Yellow
                    } else {
                        Color::Green
                    },
                ),
            )
        }));
    }
    let info_area = if let Some((file, done, total)) = &w.progress {
        let parts = Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).split(body[1]);
        let ratio = if *total > 0 {
            (*done as f64 / *total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        f.render_widget(
            Gauge::default().ratio(ratio).label(format!(
                "{file} {:.0} / {:.0} MB",
                *done as f64 / 1e6,
                *total as f64 / 1e6
            )),
            parts[0],
        );
        parts[1]
    } else {
        body[1]
    };
    f.render_widget(
        Paragraph::new(text)
            .wrap(Wrap { trim: false })
            .block(Block::new().borders(Borders::ALL)),
        info_area,
    );

    let hint = if w.editing.is_some() {
        "type, Enter accepts, Esc cancels"
    } else {
        "Up/Down select   Left/Right change   Enter edit or run   n next   b back   q quit"
    };
    f.render_widget(
        Paragraph::new(hint).block(Block::new().borders(Borders::ALL)),
        rows[2],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn every_page_renders_on_a_small_terminal() {
        let mut w = Wizard::new(None, "laptop");
        for step in Step::ALL {
            w.goto(step);
            let mut t = Terminal::new(TestBackend::new(80, 24)).unwrap();
            t.draw(|f| draw(f, &w)).unwrap();
            let text: String = t
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            assert!(text.contains(step.title()), "{step:?}");
        }
    }

    #[test]
    fn level_bar_scales() {
        assert!(level_bar(-80.0).starts_with("[                              ]"));
        assert!(level_bar(0.0).contains(&"#".repeat(30)));
    }

    #[test]
    fn key_mapping() {
        assert_eq!(
            map_key(KeyCode::Char('c'), KeyModifiers::CONTROL),
            Some(Key::CtrlC)
        );
        assert_eq!(
            map_key(KeyCode::Char('x'), KeyModifiers::NONE),
            Some(Key::Char('x'))
        );
        assert_eq!(map_key(KeyCode::F(1), KeyModifiers::NONE), None);
    }
}
