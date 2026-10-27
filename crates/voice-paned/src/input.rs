//! Is the focused pane's input line empty?
//!
//! `Send` without `force` must not append to something the operator is half
//! way through typing. Agent TUIs render a *placeholder* hint in the empty
//! composer (`Try "fix typecheck errors"`), so a plain `capture-pane` would
//! mistake it for typed text. The placeholder is drawn with the SGR *dim*
//! attribute, which `capture-pane -e` preserves; dim runs are ignored.
//!
//! Heuristics and their limits:
//!
//! * **Claude Code**: the composer is the last line that starts with `❯` (or
//!   `>`) followed by a space (Claude uses a no-break space) and is not a numbered menu
//!   option. Lines down to the next horizontal rule continue a multi-line
//!   input. Verified against Claude Code 2.1.306 captures.
//! * **Codex**: the same, with the `›` marker. Verified against Codex 0.163.0.
//! * **Other** (shells, editors): the text left of the cursor on the cursor's
//!   row, minus a trailing prompt glyph (`$ # % > ❯ › λ →`). A prompt that does
//!   not end in one of those glyphs, or a TUI that is not line-oriented, reads
//!   as "has text": the safe direction, since `force` overrides it.
//! * If a Claude/Codex composer cannot be found (layout changed, screen
//!   mid-redraw) the check fails *open* and reports empty.
//!
//! Known blind spots: a reverse-video cursor sitting on typed text inside the
//! line, and agents that render their placeholder with a colour instead of the
//! dim attribute.

use voicelink_proto::AgentKind;

/// State of the input line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputState {
    /// Nothing typed (or only a placeholder).
    Empty,
    /// Text is present.
    HasText,
}

/// One screen cell's character with the attributes this module cares about.
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    ch: char,
    dim: bool,
    reverse: bool,
}

/// Parses a line of `capture-pane -e` output into cells, tracking SGR dim and
/// reverse and discarding every escape sequence.
pub fn parse_ansi(line: &str) -> Vec<Cell> {
    let mut cells = Vec::new();
    let (mut dim, mut reverse) = (false, false);
    let mut it = line.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\u{1b}' {
            cells.push(Cell {
                ch: c,
                dim,
                reverse,
            });
            continue;
        }
        match it.next() {
            Some('[') => {
                let mut params = String::new();
                let mut fin = '\0';
                for p in it.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&p) {
                        fin = p;
                        break;
                    }
                    params.push(p);
                }
                if fin == 'm' {
                    apply_sgr(&params, &mut dim, &mut reverse);
                }
            }
            Some(']') => {
                // OSC: up to BEL or ST.
                while let Some(p) = it.next() {
                    if p == '\u{7}' || (p == '\u{1b}' && it.peek() == Some(&'\\')) {
                        if p != '\u{7}' {
                            it.next();
                        }
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    cells
}

fn apply_sgr(params: &str, dim: &mut bool, reverse: &mut bool) {
    let nums: Vec<u32> = params
        .split([';', ':'])
        .map(|p| p.parse().unwrap_or(0))
        .collect();
    let mut i = 0;
    while i < nums.len() {
        match nums[i] {
            0 => {
                *dim = false;
                *reverse = false;
            }
            2 => *dim = true,
            22 => *dim = false,
            7 => *reverse = true,
            27 => *reverse = false,
            // Extended colours carry sub-parameters that must not be read as attributes.
            38 | 48 | 58 => match nums.get(i + 1) {
                Some(5) => i += 2,
                Some(2) => i += 4,
                _ => {}
            },
            _ => {}
        }
        i += 1;
    }
    if params.is_empty() {
        *dim = false;
        *reverse = false;
    }
}

/// Removes every escape sequence from `capture-pane -e` text.
pub fn strip_ansi(text: &str) -> String {
    text.lines()
        .map(|l| plain(&parse_ansi(l)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn plain(cells: &[Cell]) -> String {
    cells.iter().map(|c| c.ch).collect()
}

fn is_rule(line: &str) -> bool {
    let t = line.trim();
    t.chars().count() >= 10
        && t.chars()
            .all(|c| "\u{2500}\u{2501}\u{254c}\u{2574}-=".contains(c))
}

fn marker_chars(agent: AgentKind) -> &'static [char] {
    match agent {
        AgentKind::Claude => &['\u{276f}', '>'],
        AgentKind::Codex => &['\u{203a}'],
        AgentKind::Other => &[],
    }
}

/// Index of the marker cell if `cells` is a composer line.
fn composer_marker(cells: &[Cell], markers: &[char]) -> Option<usize> {
    let start = cells
        .iter()
        .position(|c| !c.ch.is_whitespace() && !"\u{2502}\u{2503}".contains(c.ch))?;
    if !markers.contains(&cells[start].ch) {
        return None;
    }
    // Marker must be followed by whitespace (Claude: NBSP) or end of line.
    match cells.get(start + 1) {
        Some(c) if !c.ch.is_whitespace() => return None,
        _ => {}
    }
    // Numbered menu options are not the composer.
    let rest: String = cells[start + 1..].iter().map(|c| c.ch).collect();
    let rest = rest.trim_start();
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    if digits > 0 && rest[digits..].starts_with(['.', ')']) {
        return None;
    }
    Some(start)
}

fn cells_have_text(cells: &[Cell]) -> bool {
    let n = cells.len();
    cells.iter().enumerate().any(|(i, c)| {
        if c.ch.is_whitespace() || "\u{2502}\u{2503}".contains(c.ch) || c.dim {
            return false;
        }
        // A reverse-video glyph directly before dim text is the cursor drawn
        // over the first character of the placeholder.
        if c.reverse
            && cells[i + 1..n]
                .iter()
                .find(|x| !x.ch.is_whitespace())
                .is_some_and(|x| x.dim)
        {
            return false;
        }
        true
    })
}

/// Input state of a Claude/Codex composer from `capture-pane -e -p` text, or
/// `None` when no composer is on screen.
pub fn composer_state(agent: AgentKind, ansi_screen: &str) -> Option<InputState> {
    let markers = marker_chars(agent);
    let lines: Vec<Vec<Cell>> = ansi_screen.lines().map(parse_ansi).collect();
    let at = lines
        .iter()
        .rposition(|l| composer_marker(l, markers).is_some())?;
    let marker = composer_marker(&lines[at], markers)?;

    let mut has_text = cells_have_text(&lines[at][marker + 1..]);
    // Continuation lines, only when a rule below delimits the input area.
    let rule_below = (at + 1..lines.len().min(at + 12)).find(|&i| is_rule(&plain(&lines[i])));
    if let Some(end) = rule_below {
        has_text |= lines[at + 1..end].iter().any(|l| cells_have_text(l));
    }
    Some(if has_text {
        InputState::HasText
    } else {
        InputState::Empty
    })
}

/// Glyphs a shell prompt commonly ends with.
const PROMPT_ENDINGS: &[char] = &[
    '$', '#', '%', '>', '\u{276f}', '\u{203a}', '\u{3bb}', '\u{2192}',
];

/// Input state of a plain pane from the cursor row's text left of the cursor.
pub fn plain_state(row_left_of_cursor: &str) -> InputState {
    let t = row_left_of_cursor.trim_end();
    match t.chars().last() {
        None => InputState::Empty,
        Some(c) if PROMPT_ENDINGS.contains(&c) => InputState::Empty,
        Some(_) => InputState::HasText,
    }
}

/// Top-level check. `ansi_screen` is `capture-pane -e -p`; `cursor_row_text`
/// is the plain text of the cursor's row up to the cursor column.
pub fn input_state(agent: AgentKind, ansi_screen: &str, cursor_row_text: &str) -> InputState {
    match agent {
        AgentKind::Claude | AgentKind::Codex => {
            composer_state(agent, ansi_screen).unwrap_or(InputState::Empty)
        }
        AgentKind::Other => plain_state(cursor_row_text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> String {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    }

    #[test]
    fn real_claude_placeholder_is_empty() {
        assert_eq!(
            composer_state(AgentKind::Claude, &fixture("real_claude_idle.ansi")),
            Some(InputState::Empty)
        );
    }

    #[test]
    fn real_claude_typed_text_is_detected() {
        assert_eq!(
            composer_state(AgentKind::Claude, &fixture("real_claude_typed.ansi")),
            Some(InputState::HasText)
        );
    }

    #[test]
    fn real_codex_placeholder_is_empty() {
        assert_eq!(
            composer_state(AgentKind::Codex, &fixture("real_codex_idle.ansi")),
            Some(InputState::Empty)
        );
    }

    #[test]
    fn real_codex_typed_text_is_detected() {
        assert_eq!(
            composer_state(AgentKind::Codex, &fixture("real_codex_typed.ansi")),
            Some(InputState::HasText)
        );
    }

    #[test]
    fn plain_capture_would_have_been_fooled() {
        // The point of -e: without attributes the placeholder reads as text.
        let plain_text = fixture("real_claude_idle.txt");
        let as_ansi_less = composer_state(AgentKind::Claude, &plain_text);
        assert_eq!(as_ansi_less, Some(InputState::HasText));
    }

    #[test]
    fn menus_are_not_composers() {
        let menu = "Question?\n\u{276f} 1. Yes\n  2. No\n";
        assert_eq!(composer_state(AgentKind::Claude, menu), None);
    }

    #[test]
    fn multi_line_claude_input() {
        let rule = "\u{2500}".repeat(20);
        let screen =
            format!("{rule}\n\u{276f}\u{a0}first line\n  second line\n{rule}\n  ? for shortcuts\n");
        assert_eq!(
            composer_state(AgentKind::Claude, &screen),
            Some(InputState::HasText)
        );
        let empty_continuation =
            format!("{rule}\n\u{276f}\u{a0}\u{1b}[2mTry it\u{1b}[0m\n  \n{rule}\n");
        assert_eq!(
            composer_state(AgentKind::Claude, &empty_continuation),
            Some(InputState::Empty)
        );
    }

    #[test]
    fn reverse_cursor_over_placeholder_is_ignored() {
        let screen = "\u{276f}\u{a0}\u{1b}[7mT\u{1b}[27m\u{1b}[2mry this\u{1b}[0m\n";
        assert_eq!(
            composer_state(AgentKind::Claude, screen),
            Some(InputState::Empty)
        );
        let typed_then_cursor = "\u{276f}\u{a0}hi\u{1b}[7m \u{1b}[27m\n";
        assert_eq!(
            composer_state(AgentKind::Claude, typed_then_cursor),
            Some(InputState::HasText)
        );
    }

    #[test]
    fn sgr_parsing_handles_extended_colours() {
        // 38;5;2 must not be read as "dim" (2).
        let cells = parse_ansi("\u{1b}[38;5;2mX\u{1b}[0m");
        assert!(!cells[0].dim);
        let cells = parse_ansi("\u{1b}[2;38;2;1;2;3mY");
        assert!(cells[0].dim);
        let cells = parse_ansi("\u{1b}[2mA\u{1b}[22mB\u{1b}]0;title\u{7}C");
        assert!(cells[0].dim && !cells[1].dim);
        assert_eq!(plain(&cells), "ABC");
    }

    #[test]
    fn plain_heuristic() {
        assert_eq!(plain_state(""), InputState::Empty);
        assert_eq!(plain_state("user@host:~$ "), InputState::Empty);
        assert_eq!(plain_state("\u{276f} "), InputState::Empty);
        assert_eq!(plain_state("user@host:~$ ls -l"), InputState::HasText);
        assert_eq!(plain_state("hello"), InputState::HasText);
    }
}
