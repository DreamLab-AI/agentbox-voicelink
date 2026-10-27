//! Quoting for the tmux command language.
//!
//! A control-mode client sends tmux *command lines*, which tmux parses like a
//! tiny shell: `;` separates commands, quotes group words, `$VAR` and `~`
//! expand, and a newline ends the line. Splicing untrusted text into such a
//! line without quoting is an injection hole. [`quote`] makes any string a
//! single, inert word; [`command_line`] assembles a whole command.
//!
//! Quoting stops the *command parser* from interpreting text. Some commands
//! additionally run their arguments through the *format* expander
//! (`display-message`, `-F` arguments, `rename-window`, ...) where `#{...}`
//! is live; use [`escape_format`] for text destined for those arguments.

use std::fmt::Write as _;

/// Quotes `s` as one tmux word.
///
/// Strings made only of letters, digits and `_ @ % : = . , / + -` are
/// returned unchanged. Anything else becomes a double-quoted word in which
/// `\`, `"` and `$` are backslash-escaped and control characters (including
/// newline and carriage return) are written as three-digit octal escapes, so
/// the result never contains a raw line break.
///
/// ```
/// use tmux_control::quote;
/// assert_eq!(quote("agentbox"), "agentbox");
/// assert_eq!(quote("a b"), r#""a b""#);
/// assert_eq!(quote("x; kill-server"), r#""x; kill-server""#);
/// assert_eq!(quote("line1\nline2"), r#""line1\012line2""#);
/// assert_eq!(quote("$HOME \"q\""), r#""\$HOME \"q\"""#);
/// assert_eq!(quote(""), r#""""#);
/// ```
pub fn quote(s: &str) -> String {
    let bare = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "_@%:=.,/+-".contains(c));
    if bare {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '$' => out.push_str("\\$"),
            c if c.is_ascii_control() => {
                // Writing to a String cannot fail.
                let _ = write!(out, "\\{:03o}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Escapes `#` as `##` so text survives tmux *format* expansion unchanged.
///
/// Combine with [`quote`] for arguments that are both parsed and
/// format-expanded.
///
/// ```
/// use tmux_control::escape_format;
/// assert_eq!(escape_format("#{pane_id} #(id)"), "##{pane_id} ##(id)");
/// ```
pub fn escape_format(s: &str) -> String {
    s.replace('#', "##")
}

/// Joins words into one command line, quoting each with [`quote`].
///
/// ```
/// use tmux_control::command_line;
/// let line = command_line(["send-keys", "-t", "%3", "-l", "--", "wipe it; echo"]);
/// assert_eq!(line, r#"send-keys -t %3 -l -- "wipe it; echo""#);
/// ```
pub fn command_line<I, S>(words: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    words
        .into_iter()
        .map(|w| quote(w.as_ref()))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_words_are_untouched() {
        for w in ["abc", "%12", "agentbox:1.0", "-t", "a=b", "/tmp/x.sock"] {
            assert_eq!(quote(w), w);
        }
    }

    #[test]
    fn hostile_strings_become_single_words() {
        for hostile in [
            "a;b",
            "a ; kill-server",
            "it's",
            "say \"hi\"",
            "#{pane_id}",
            "#(touch /tmp/x)",
            "$(id) $HOME ~",
            "back\\slash",
            "multi\nline",
            "cr\rlf",
            "tab\there",
            "\u{7f}del",
            "unicode \u{e9} \u{65e5}\u{672c}",
            "",
            "{ run }",
        ] {
            let q = quote(hostile);
            assert!(q.starts_with('"') && q.ends_with('"'), "{q}");
            assert!(!q.contains('\n') && !q.contains('\r'), "{q:?}");
            // Every interior quote and dollar must be preceded by an odd run of backslashes.
            let inner = &q[1..q.len() - 1];
            let mut run = 0;
            for c in inner.chars() {
                assert!(
                    !((c == '"' || c == '$') && run % 2 == 0),
                    "unescaped {c:?} in {q}"
                );
                run = if c == '\\' { run + 1 } else { 0 };
            }
        }
    }

    #[test]
    fn control_characters_are_octal() {
        assert_eq!(quote("a\nb"), r#""a\012b""#);
        assert_eq!(quote("\u{1b}"), r#""\033""#);
    }

    #[test]
    fn format_escape() {
        assert_eq!(escape_format("#{x}"), "##{x}");
        assert_eq!(escape_format("plain"), "plain");
    }

    #[test]
    fn command_line_quotes_every_word() {
        assert_eq!(
            command_line(["rename-window", "-t", "@1", "a;b"]),
            r#"rename-window -t @1 "a;b""#
        );
    }
}
