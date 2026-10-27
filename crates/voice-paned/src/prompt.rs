//! Blocking-prompt detection from pane content.
//!
//! Codex raises no hook for approval prompts, and Claude Code's menus (for
//! example the `AskUserQuestion` tool) may not either, so the controller looks
//! at the screen. All patterns live in the tables below; the fixture tests in
//! `tests/fixtures/` pin them.
//!
//! **The fixtures are hand-built from the CLIs' known layouts. Refresh them
//! against real captures (`tmux capture-pane -p`) whenever Claude Code or
//! Codex is upgraded**, and extend the tables if a layout changed.
//!
//! Two shapes are recognised:
//!
//! 1. A *cursor menu*: two or more consecutively numbered options of which
//!    one carries the agent's selection cursor (`❯ 1. Yes`, `› 1. Yes, proceed`).
//!    The question is the nearest meaningful line above the first option.
//! 2. A *phrase prompt*: a known question line ([`PHRASES`]) or a `(y/n)`
//!    suffix, followed by answer lines, for prompts without a numbered list.

use voicelink_proto::AgentKind;

/// How many trailing lines of a pane are examined.
pub const SCAN_LINES: usize = 40;

/// Selection-cursor glyphs per agent.
const CURSORS: &[(AgentKind, &[char])] = &[
    (AgentKind::Claude, &['\u{276f}', '>']), // ❯ and a plain > fallback
    (AgentKind::Codex, &['\u{203a}', '\u{276f}', '>']), // › ❯ >
];

/// Question phrases that mark a prompt even without a cursor menu.
/// Matched case-insensitively against the whole line.
const PHRASES: &[(AgentKind, &str)] = &[
    (AgentKind::Claude, "do you want to proceed?"),
    (AgentKind::Claude, "would you like to proceed?"),
    (AgentKind::Claude, "do you want to make this edit"),
    (AgentKind::Codex, "allow command?"),
    (
        AgentKind::Codex,
        "would you like to run the following command",
    ),
    (
        AgentKind::Codex,
        "would you like to make the following edits",
    ),
];

/// Suffixes of inline yes/no questions, for either agent.
const YES_NO_SUFFIXES: &[&str] = &[
    "(y/n)", "[y/n]", "(y/n)?", "[y/n]?", "[y/N]", "[Y/n]", "(Y/n)", "(y/N)",
];

/// Lines that end a menu and are never part of the question.
const FOOTER_HINTS: &[&str] = &[
    "esc to cancel",
    "enter to select",
    "tab/arrow keys",
    "press enter to continue",
];

/// Detects a blocking prompt in the tail of a pane. Returns the text to
/// speak: the question followed by the option lines.
pub fn detect(agent: AgentKind, screen: &str) -> Option<String> {
    if agent == AgentKind::Other {
        return None;
    }
    let all: Vec<&str> = screen.lines().collect();
    let tail = &all[all.len().saturating_sub(SCAN_LINES)..];
    let lines: Vec<String> = tail.iter().map(|l| normalise(l)).collect();
    cursor_menu(agent, &lines).or_else(|| phrase_prompt(agent, &lines))
}

/// Strips box-drawing borders and surrounding whitespace.
fn normalise(line: &str) -> String {
    line.trim_matches(|c: char| c.is_whitespace() || "\u{2502}\u{2503}\u{2551}".contains(c))
        .to_owned()
}

fn is_rule(line: &str) -> bool {
    let n = line
        .chars()
        .filter(|c| "\u{2500}\u{2501}\u{254c}\u{256d}\u{256e}\u{2570}\u{256f}-=_".contains(*c))
        .count();
    n >= 8 && n * 10 >= line.chars().count() * 8
}

fn is_footer(line: &str) -> bool {
    let l = line.to_lowercase();
    FOOTER_HINTS.iter().any(|h| l.contains(h))
}

/// A parsed `N. label` option line.
struct Opt {
    cursor: bool,
    number: u32,
    label: String,
}

fn parse_option(line: &str, cursors: &[char]) -> Option<Opt> {
    let mut rest = line;
    let mut cursor = false;
    if let Some(c) = rest.chars().next() {
        if cursors.contains(&c) {
            cursor = true;
            rest = rest[c.len_utf8()..].trim_start();
        }
    }
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() || digits.len() > 2 {
        return None;
    }
    let after = &rest[digits.len()..];
    let after = after
        .strip_prefix('.')
        .or_else(|| after.strip_prefix(')'))?;
    let label = after.strip_prefix(' ')?.trim();
    if label.is_empty() {
        return None;
    }
    Some(Opt {
        cursor,
        number: digits.parse().ok()?,
        label: label.to_owned(),
    })
}

/// Most description lines tolerated between two options (`AskUserQuestion`
/// shows one under each label).
const MAX_DESCRIPTION_LINES: usize = 3;

fn cursor_menu(agent: AgentKind, lines: &[String]) -> Option<String> {
    let cursors = CURSORS.iter().find(|(a, _)| *a == agent)?.1;
    let parsed: Vec<Option<Opt>> = lines.iter().map(|l| parse_option(l, cursors)).collect();
    // The last line carrying the cursor on a numbered option.
    let at = parsed
        .iter()
        .rposition(|p| p.as_ref().is_some_and(|o| o.cursor))?;

    // Walk up to option 1.
    let mut first = at;
    let mut gap = 0;
    for i in (0..at).rev() {
        match &parsed[i] {
            Some(o) if o.number < parsed[first].as_ref()?.number => {
                first = i;
                gap = 0;
            }
            Some(_) => break,
            None if lines[i].is_empty() || is_rule(&lines[i]) || gap >= MAX_DESCRIPTION_LINES => {
                break;
            }
            None => gap += 1,
        }
    }
    if parsed[first].as_ref()?.number != 1 {
        return None;
    }

    // Walk down over consecutively numbered options.
    let mut options = vec![first];
    let mut next = 2;
    gap = 0;
    for (i, p) in parsed.iter().enumerate().skip(first + 1) {
        match p {
            Some(o) if o.number == next => {
                options.push(i);
                next += 1;
                gap = 0;
            }
            Some(_) => break,
            None if lines[i].is_empty()
                || is_rule(&lines[i])
                || is_footer(&lines[i])
                || gap >= MAX_DESCRIPTION_LINES =>
            {
                break;
            }
            None => gap += 1,
        }
    }
    if options.len() < 2 || !options.contains(&at) {
        return None;
    }

    // Question: the nearest line ending in `?` within a few meaningful lines
    // above the first option, else the nearest meaningful line.
    let above: Vec<&String> = lines[..first]
        .iter()
        .rev()
        .filter(|l| !l.is_empty() && !is_rule(l) && !is_footer(l))
        .take(6)
        .collect();
    let question = above
        .iter()
        .find(|l| l.ends_with('?'))
        .or(above.first())
        .map(|l| (*l).clone());

    let mut out: Vec<String> = question.into_iter().collect();
    for &i in &options {
        let o = parsed[i].as_ref()?;
        out.push(format!("{}. {}", o.number, o.label));
    }
    Some(out.join("\n"))
}

fn is_phrase(agent: AgentKind, line: &str) -> bool {
    let lower = line.to_lowercase();
    PHRASES
        .iter()
        .any(|(a, p)| *a == agent && lower.contains(p))
}

fn phrase_prompt(agent: AgentKind, lines: &[String]) -> Option<String> {
    let last = lines.iter().rposition(|l| {
        is_phrase(agent, l)
            || YES_NO_SUFFIXES
                .iter()
                .any(|s| l.to_lowercase().trim_end().ends_with(&s.to_lowercase()))
    })?;
    // Pull in a question phrase a few lines above a bare `[y/n]` line.
    let lo = last.saturating_sub(8);
    let start = (lo..last)
        .find(|&i| is_phrase(agent, &lines[i]))
        .unwrap_or(last);
    let mut out: Vec<&str> = lines[start..=last]
        .iter()
        .filter(|l| !l.is_empty())
        .map(String::as_str)
        .collect();
    for l in lines[last + 1..].iter().take(8) {
        if is_rule(l) || is_footer(l) {
            break;
        }
        if !l.is_empty() {
            out.push(l);
        }
    }
    Some(out.join("\n"))
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

    /// (fixture, agent, expected text, or None for "no prompt")
    #[allow(clippy::type_complexity)]
    const CASES: &[(&str, AgentKind, Option<&str>)] = &[
        (
            "claude_permission.txt",
            AgentKind::Claude,
            Some(
                "Do you want to proceed?\n1. Yes\n2. Yes, and don't ask again for npm test commands\n3. No, and tell Claude what to do differently (esc)",
            ),
        ),
        (
            "claude_ask_user_question.txt",
            AgentKind::Claude,
            Some("Which database should the service use?\n1. PostgreSQL\n2. SQLite\n3. Other"),
        ),
        (
            "claude_plan_approval.txt",
            AgentKind::Claude,
            Some(
                "Would you like to proceed?\n1. Yes, and auto-accept edits\n2. Yes, and manually approve edits\n3. No, keep planning",
            ),
        ),
        ("claude_idle.txt", AgentKind::Claude, None),
        ("claude_working.txt", AgentKind::Claude, None),
        ("real_claude_idle.txt", AgentKind::Claude, None),
        ("real_claude_typed.txt", AgentKind::Claude, None),
        ("real_codex_idle.txt", AgentKind::Codex, None),
        (
            "codex_approval.txt",
            AgentKind::Codex,
            Some(
                "Would you like to run the following command?\n1. Yes, proceed (y)\n2. Yes, and don't ask again for commands that start with `cargo test` (p)\n3. No, and tell Codex what to do differently (esc)",
            ),
        ),
        (
            "codex_allow_command.txt",
            AgentKind::Codex,
            Some("Allow command?\nrm -r build\n[y/n]"),
        ),
        ("codex_idle.txt", AgentKind::Codex, None),
    ];

    #[test]
    fn fixtures_match_expectations() {
        for (file, agent, expected) in CASES {
            let got = detect(*agent, &fixture(file));
            assert_eq!(got.as_deref(), *expected, "fixture {file}");
        }
    }

    #[test]
    fn other_agents_never_prompt() {
        assert_eq!(
            detect(AgentKind::Other, &fixture("claude_permission.txt")),
            None
        );
    }

    #[test]
    fn plain_numbered_list_without_cursor_is_not_a_menu() {
        let screen = "Here are the steps:\n1. Install\n2. Build\n3. Run\n";
        assert_eq!(detect(AgentKind::Claude, screen), None);
        assert_eq!(detect(AgentKind::Codex, screen), None);
    }

    #[test]
    fn single_option_is_not_a_menu() {
        assert_eq!(
            detect(AgentKind::Claude, "Pick\n\u{276f} 1. Only one\n"),
            None
        );
    }

    #[test]
    fn input_cursor_line_is_not_a_menu() {
        assert_eq!(
            detect(
                AgentKind::Claude,
                "\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n\u{276f} fix the build\n\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n"
            ),
            None
        );
    }

    #[test]
    fn only_the_tail_is_examined() {
        let mut screen = fixture("claude_permission.txt");
        for _ in 0..SCAN_LINES + 5 {
            screen.push_str("scrolled output line\n");
        }
        assert_eq!(detect(AgentKind::Claude, &screen), None);
    }
}
