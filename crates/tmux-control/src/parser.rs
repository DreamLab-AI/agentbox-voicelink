//! Pure, I/O-free decoding of the control-mode stream.
//!
//! [`Decoder`] consumes one line at a time (without its terminator) and yields
//! [`Event`]s: command replies and notifications. It implements the CONTROL
//! MODE section of tmux(1):
//!
//! * A command's output is wrapped in `%begin time number flags` and
//!   `%end time number flags` (or `%error ...`). **Only** the matching
//!   `%end`/`%error` closes a block; every other line inside it, including
//!   lines that start with `%` (a pane id such as `%3` is ordinary output),
//!   is command output. tmux itself queues notifications until a block ends,
//!   so this never loses one.
//! * Outside a block, a line starting with `%` is a notification.
//! * `\r` line endings and the `\eP1000p` / `\e\\` wrapping used by `-CC` are
//!   stripped.

use std::fmt;

/// The reply to one command: the contents of a `%begin` ... `%end`/`%error`
/// block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    /// Server timestamp (seconds since the epoch) from the `%begin` line.
    pub time: u64,
    /// tmux's command sequence number.
    pub number: u64,
    /// The block flags. Bit 0 is set for replies to commands sent by this
    /// client; server-originated blocks (such as the one answering the
    /// initial `attach`) have it clear.
    pub flags: u32,
    /// The output lines on success, or the error text lines on `%error`.
    pub result: Result<Vec<String>, Vec<String>>,
}

impl Reply {
    /// Whether this reply answers a command sent by this client.
    pub fn is_ours(&self) -> bool {
        self.flags & 1 == 1
    }
}

/// One decoded item of the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// A completed command block.
    Reply(Reply),
    /// An asynchronous notification.
    Notification(Notification),
}

/// An asynchronous `%` notification from tmux.
///
/// Identifiers keep tmux's sigils: sessions `$1`, windows `@2`, panes `%3`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Notification {
    /// `%output pane data`: pane output (absent when the client uses the
    /// `no-output` flag). `data` is the raw bytes after octal-unescaping.
    Output {
        /// Pane id.
        pane: String,
        /// Output bytes (not necessarily valid UTF-8).
        data: Vec<u8>,
    },
    /// `%extended-output pane age ... : data` (flow control with
    /// `pause-after`).
    ExtendedOutput {
        /// Pane id.
        pane: String,
        /// Milliseconds the output was buffered.
        age_ms: u64,
        /// Output bytes.
        data: Vec<u8>,
    },
    /// `%session-window-changed session window`: a session's current window
    /// changed.
    SessionWindowChanged {
        /// Session id.
        session: String,
        /// New current window id.
        window: String,
    },
    /// `%window-pane-changed window pane`: a window's active pane changed.
    WindowPaneChanged {
        /// Window id.
        window: String,
        /// New active pane id.
        pane: String,
    },
    /// `%session-changed session name`: this client attached to another
    /// session.
    SessionChanged {
        /// Session id.
        session: String,
        /// Session name.
        name: String,
    },
    /// `%client-session-changed client session name`: some client switched
    /// session.
    ClientSessionChanged {
        /// Client name.
        client: String,
        /// Session id.
        session: String,
        /// Session name.
        name: String,
    },
    /// `%sessions-changed`: a session was created or destroyed.
    SessionsChanged,
    /// `%window-add window`: a window was linked into this client's session.
    WindowAdd {
        /// Window id.
        window: String,
    },
    /// `%window-close window`: a window of this client's session closed.
    WindowClose {
        /// Window id.
        window: String,
    },
    /// `%unlinked-window-add window`: a window appeared in another session.
    UnlinkedWindowAdd {
        /// Window id.
        window: String,
    },
    /// `%unlinked-window-close window`: a window of another session closed.
    UnlinkedWindowClose {
        /// Window id.
        window: String,
    },
    /// `%window-renamed window name`.
    WindowRenamed {
        /// Window id.
        window: String,
        /// New name.
        name: String,
    },
    /// `%layout-change window layout visible-layout [flags]`.
    LayoutChange {
        /// Window id.
        window: String,
        /// The window layout string.
        layout: String,
        /// The layout as visible (differs when a pane is zoomed).
        visible_layout: String,
        /// Window flags, when tmux sent them.
        flags: Option<String>,
    },
    /// `%pane-mode-changed pane`: a pane entered or left a mode.
    PaneModeChanged {
        /// Pane id.
        pane: String,
    },
    /// `%exit [reason]`: the control client is ending.
    Exit {
        /// Why, when tmux said.
        reason: Option<String>,
    },
    /// `%pause pane`: flow control paused a pane.
    Pause {
        /// Pane id.
        pane: String,
    },
    /// `%continue pane`: flow control resumed a pane.
    Continue {
        /// Pane id.
        pane: String,
    },
    /// `%subscription-changed name session window index pane : value`.
    SubscriptionChanged {
        /// Subscription name.
        name: String,
        /// Session id, or `None` when tmux sent `-`.
        session: Option<String>,
        /// Window id, or `None`.
        window: Option<String>,
        /// Window index (for `%window` subscriptions), or `None`.
        index: Option<String>,
        /// Pane id, or `None`.
        pane: Option<String>,
        /// The format's new value.
        value: String,
    },
    /// Any notification this crate does not model, or a line outside a block
    /// that is not a notification at all. Holds the whole line.
    Unknown(String),
}

/// Incremental line decoder; see the [crate documentation](crate).
///
/// ```
/// use tmux_control::{Decoder, Event, Notification};
///
/// let mut d = Decoder::default();
/// assert!(d.feed(b"%begin 1700000000 7 1").is_none());
/// assert!(d.feed(b"%0 first pane").is_none()); // output, not a notification
/// let Some(Event::Reply(r)) = d.feed(b"%end 1700000000 7 1") else { panic!() };
/// assert_eq!(r.result, Ok(vec!["%0 first pane".to_string()]));
/// assert_eq!(
///     d.feed(b"%window-add @3"),
///     Some(Event::Notification(Notification::WindowAdd { window: "@3".into() })),
/// );
/// ```
#[derive(Debug, Default)]
pub struct Decoder {
    open: Option<Block>,
}

#[derive(Debug)]
struct Block {
    time: u64,
    number: u64,
    flags: u32,
    lines: Vec<String>,
}

impl Decoder {
    /// Whether a `%begin` block is currently open (an end of stream now means
    /// a truncated reply).
    pub fn in_block(&self) -> bool {
        self.open.is_some()
    }

    /// Feeds one line, without its `\n`. Returns an [`Event`] when the line
    /// completes one.
    pub fn feed(&mut self, raw: &[u8]) -> Option<Event> {
        let line = normalise(raw);
        if let Some(block) = self.open.as_mut() {
            if let Some((kind, time, number, _flags)) = parse_guard(line)
                && kind != Guard::Begin
                && number == block.number
                && time == block.time
            {
                let block = self.open.take()?;
                let result = if kind == Guard::End {
                    Ok(block.lines)
                } else {
                    Err(block.lines)
                };
                return Some(Event::Reply(Reply {
                    time: block.time,
                    number: block.number,
                    flags: block.flags,
                    result,
                }));
            }
            block.lines.push(String::from_utf8_lossy(line).into_owned());
            return None;
        }
        if line.is_empty() {
            return None;
        }
        if let Some((Guard::Begin, time, number, flags)) = parse_guard(line) {
            self.open = Some(Block {
                time,
                number,
                flags,
                lines: Vec::new(),
            });
            return None;
        }
        Some(Event::Notification(parse_notification(line)))
    }
}

/// Removes a trailing `\r`, and the `-CC` DCS introducer / string terminator.
fn normalise(mut line: &[u8]) -> &[u8] {
    if let Some(rest) = line.strip_suffix(b"\r") {
        line = rest;
    }
    if let Some(rest) = line.strip_prefix(b"\x1bP1000p") {
        line = rest;
    }
    if let Some(rest) = line.strip_suffix(b"\x1b\\") {
        line = rest;
    }
    line
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Guard {
    Begin,
    End,
    Error,
}

fn parse_guard(line: &[u8]) -> Option<(Guard, u64, u64, u32)> {
    let text = std::str::from_utf8(line).ok()?;
    let mut it = text.split(' ');
    let kind = match it.next()? {
        "%begin" => Guard::Begin,
        "%end" => Guard::End,
        "%error" => Guard::Error,
        _ => return None,
    };
    let time = it.next()?.parse().ok()?;
    let number = it.next()?.parse().ok()?;
    let flags = it.next().and_then(|f| f.parse().ok()).unwrap_or(0);
    Some((kind, time, number, flags))
}

/// Reverses tmux's output escaping: bytes below 0x20 and `\` are sent as a
/// backslash followed by three octal digits.
///
/// ```
/// use tmux_control::unescape_output;
/// assert_eq!(unescape_output(br"a\015\012b\134c"), b"a\r\nb\\c");
/// assert_eq!(unescape_output(br"bad\9"), br"bad\9"); // malformed: kept literally
/// ```
pub fn unescape_output(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if data[i] == b'\\'
            && i + 4 <= data.len()
            && let Some(b) = octal3(&data[i + 1..i + 4])
        {
            out.push(b);
            i += 4;
            continue;
        }
        out.push(data[i]);
        i += 1;
    }
    out
}

fn octal3(d: &[u8]) -> Option<u8> {
    if d.len() != 3 || !d.iter().all(|c| (b'0'..=b'7').contains(c)) {
        return None;
    }
    let v = u32::from(d[0] - b'0') * 64 + u32::from(d[1] - b'0') * 8 + u32::from(d[2] - b'0');
    u8::try_from(v).ok()
}

/// Splits `line` at the first space: `(head, rest)`; `rest` is empty when
/// there is no space.
fn split1(line: &[u8]) -> (&[u8], &[u8]) {
    match line.iter().position(|&b| b == b' ') {
        Some(p) => (&line[..p], &line[p + 1..]),
        None => (line, &[]),
    }
}

fn s(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn opt(b: &str) -> Option<String> {
    (b != "-").then(|| b.to_owned())
}

/// Parses a `%` line outside any block. Never fails: unmodelled or malformed
/// lines become [`Notification::Unknown`].
pub fn parse_notification(line: &[u8]) -> Notification {
    let unknown = || Notification::Unknown(s(line));
    let (name, rest) = split1(line);
    match name {
        b"%output" => {
            let (pane, data) = split1(rest);
            if pane.is_empty() {
                return unknown();
            }
            Notification::Output {
                pane: s(pane),
                data: unescape_output(data),
            }
        }
        b"%extended-output" => {
            let (pane, rest) = split1(rest);
            // "age [reserved...] : data"
            let (head, data): (&[u8], &[u8]) = if let Some(p) = find(rest, b" : ") {
                (&rest[..p], &rest[p + 3..])
            } else if let Some(h) = rest.strip_suffix(b" :") {
                (h, &[])
            } else {
                return unknown();
            };
            let age = std::str::from_utf8(split1(head).0)
                .ok()
                .and_then(|a| a.parse().ok());
            match (pane.is_empty(), age) {
                (false, Some(age_ms)) => Notification::ExtendedOutput {
                    pane: s(pane),
                    age_ms,
                    data: unescape_output(data),
                },
                _ => unknown(),
            }
        }
        b"%sessions-changed" => Notification::SessionsChanged,
        b"%exit" => {
            let reason = s(rest);
            Notification::Exit {
                reason: (!reason.is_empty()).then_some(reason),
            }
        }
        _ => parse_text_notification(&s(line)).unwrap_or_else(unknown),
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn parse_text_notification(line: &str) -> Option<Notification> {
    let (name, rest) = line.split_once(' ').unwrap_or((line, ""));
    let one = |r: &str| (!r.is_empty()).then(|| r.to_owned());
    Some(match name {
        "%session-window-changed" => {
            let (session, window) = rest.split_once(' ')?;
            Notification::SessionWindowChanged {
                session: session.into(),
                window: window.into(),
            }
        }
        "%window-pane-changed" => {
            let (window, pane) = rest.split_once(' ')?;
            Notification::WindowPaneChanged {
                window: window.into(),
                pane: pane.into(),
            }
        }
        "%session-changed" => {
            let (session, name) = rest.split_once(' ')?;
            Notification::SessionChanged {
                session: session.into(),
                name: name.into(),
            }
        }
        "%client-session-changed" => {
            let mut it = rest.splitn(3, ' ');
            Notification::ClientSessionChanged {
                client: it.next()?.into(),
                session: it.next()?.into(),
                name: it.next()?.into(),
            }
        }
        "%window-add" => Notification::WindowAdd { window: one(rest)? },
        "%window-close" => Notification::WindowClose { window: one(rest)? },
        "%unlinked-window-add" => Notification::UnlinkedWindowAdd { window: one(rest)? },
        "%unlinked-window-close" => Notification::UnlinkedWindowClose { window: one(rest)? },
        "%window-renamed" => {
            let (window, name) = rest.split_once(' ')?;
            Notification::WindowRenamed {
                window: window.into(),
                name: name.into(),
            }
        }
        "%layout-change" => {
            let mut it = rest.splitn(4, ' ');
            Notification::LayoutChange {
                window: it.next()?.into(),
                layout: it.next()?.into(),
                visible_layout: it.next()?.into(),
                flags: it.next().map(str::to_owned),
            }
        }
        "%pane-mode-changed" => Notification::PaneModeChanged { pane: one(rest)? },
        "%pause" => Notification::Pause { pane: one(rest)? },
        "%continue" => Notification::Continue { pane: one(rest)? },
        "%subscription-changed" => {
            let (head, value) = rest
                .split_once(" : ")
                .or_else(|| rest.strip_suffix(" :").map(|h| (h, "")))?;
            let mut it = head.split(' ');
            Notification::SubscriptionChanged {
                name: it.next()?.into(),
                session: opt(it.next()?),
                window: opt(it.next()?),
                index: opt(it.next()?),
                pane: opt(it.next()?),
                value: value.into(),
            }
        }
        _ => return None,
    })
}

impl fmt::Display for Notification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(d: &mut Decoder, lines: &[&str]) -> Vec<Event> {
        lines.iter().filter_map(|l| d.feed(l.as_bytes())).collect()
    }

    fn notif(line: &str) -> Notification {
        match Decoder::default().feed(line.as_bytes()) {
            Some(Event::Notification(n)) => n,
            other => panic!("not a notification: {other:?}"),
        }
    }

    #[test]
    fn octal_escapes() {
        assert_eq!(unescape_output(br"hello\015\012"), b"hello\r\n");
        assert_eq!(unescape_output(br"\033[1mX\033[0m"), b"\x1b[1mX\x1b[0m");
        assert_eq!(unescape_output(br"a\134b"), b"a\\b");
        assert_eq!(unescape_output(br"\0"), br"\0");
        assert_eq!(unescape_output(br"\08x"), br"\08x");
        assert_eq!(unescape_output(br"end\"), br"end\");
        assert_eq!(unescape_output(b"caf\xc3\xa9"), "caf\u{e9}".as_bytes());
        assert_eq!(unescape_output(br"\400"), br"\400"); // > 0xff is malformed
    }

    #[test]
    fn output_notification() {
        assert_eq!(
            notif(r"%output %5 ls\015\012"),
            Notification::Output {
                pane: "%5".into(),
                data: b"ls\r\n".to_vec()
            }
        );
        assert_eq!(
            notif("%output %5 "),
            Notification::Output {
                pane: "%5".into(),
                data: vec![]
            }
        );
    }

    #[test]
    fn extended_output() {
        assert_eq!(
            notif(r"%extended-output %2 150 : hi\012"),
            Notification::ExtendedOutput {
                pane: "%2".into(),
                age_ms: 150,
                data: b"hi\n".to_vec()
            }
        );
        assert_eq!(
            notif("%extended-output %2 9 res : "),
            Notification::ExtendedOutput {
                pane: "%2".into(),
                age_ms: 9,
                data: vec![]
            }
        );
        assert!(matches!(
            notif("%extended-output %2 nope"),
            Notification::Unknown(_)
        ));
    }

    #[test]
    fn simple_notifications() {
        assert_eq!(
            notif("%session-window-changed $1 @4"),
            Notification::SessionWindowChanged {
                session: "$1".into(),
                window: "@4".into()
            }
        );
        assert_eq!(
            notif("%window-pane-changed @4 %9"),
            Notification::WindowPaneChanged {
                window: "@4".into(),
                pane: "%9".into()
            }
        );
        assert_eq!(
            notif("%session-changed $2 my session"),
            Notification::SessionChanged {
                session: "$2".into(),
                name: "my session".into()
            }
        );
        assert_eq!(
            notif("%client-session-changed /dev/pts/3 $2 work"),
            Notification::ClientSessionChanged {
                client: "/dev/pts/3".into(),
                session: "$2".into(),
                name: "work".into()
            }
        );
        assert_eq!(notif("%sessions-changed"), Notification::SessionsChanged);
        assert_eq!(
            notif("%window-add @7"),
            Notification::WindowAdd {
                window: "@7".into()
            }
        );
        assert_eq!(
            notif("%window-close @7"),
            Notification::WindowClose {
                window: "@7".into()
            }
        );
        assert_eq!(
            notif("%unlinked-window-add @7"),
            Notification::UnlinkedWindowAdd {
                window: "@7".into()
            }
        );
        assert_eq!(
            notif("%unlinked-window-close @7"),
            Notification::UnlinkedWindowClose {
                window: "@7".into()
            }
        );
        assert_eq!(
            notif("%window-renamed @7 new name here"),
            Notification::WindowRenamed {
                window: "@7".into(),
                name: "new name here".into()
            }
        );
        assert_eq!(
            notif("%pane-mode-changed %3"),
            Notification::PaneModeChanged { pane: "%3".into() }
        );
        assert_eq!(
            notif("%pause %3"),
            Notification::Pause { pane: "%3".into() }
        );
        assert_eq!(
            notif("%continue %3"),
            Notification::Continue { pane: "%3".into() }
        );
    }

    #[test]
    fn layout_change_with_and_without_flags() {
        assert_eq!(
            notif("%layout-change @1 c9f6,40x10,0,0 c9f6,40x10,0,0 *"),
            Notification::LayoutChange {
                window: "@1".into(),
                layout: "c9f6,40x10,0,0".into(),
                visible_layout: "c9f6,40x10,0,0".into(),
                flags: Some("*".into())
            }
        );
        assert!(matches!(
            notif("%layout-change @1 a b"),
            Notification::LayoutChange { flags: None, .. }
        ));
    }

    #[test]
    fn exit_with_and_without_reason() {
        assert_eq!(notif("%exit"), Notification::Exit { reason: None });
        assert_eq!(
            notif("%exit server exited unexpectedly"),
            Notification::Exit {
                reason: Some("server exited unexpectedly".into())
            }
        );
    }

    #[test]
    fn subscription() {
        assert_eq!(
            notif("%subscription-changed clock $1 @2 3 %4 : 12:34 pm"),
            Notification::SubscriptionChanged {
                name: "clock".into(),
                session: Some("$1".into()),
                window: Some("@2".into()),
                index: Some("3".into()),
                pane: Some("%4".into()),
                value: "12:34 pm".into()
            }
        );
        assert_eq!(
            notif("%subscription-changed s $1 - - - : v"),
            Notification::SubscriptionChanged {
                name: "s".into(),
                session: Some("$1".into()),
                window: None,
                index: None,
                pane: None,
                value: "v".into()
            }
        );
    }

    #[test]
    fn unknown_and_malformed_lines_never_panic() {
        for l in [
            "%future-thing a b c",
            "%window-add",
            "%session-window-changed $1",
            "%layout-change @1",
            "%subscription-changed only",
            "%output",
            "garbage without percent",
            "%",
            "%begin",
            "%begin x y z",
        ] {
            let mut d = Decoder::default();
            let ev = d.feed(l.as_bytes());
            assert!(
                matches!(ev, Some(Event::Notification(Notification::Unknown(_)))),
                "{l}: {ev:?}"
            );
            assert!(!d.in_block());
        }
    }

    #[test]
    fn begin_end_block() {
        let mut d = Decoder::default();
        let evs = feed_all(
            &mut d,
            &["%begin 100 5 1", "one", "", "three", "%end 100 5 1"],
        );
        assert_eq!(
            evs,
            vec![Event::Reply(Reply {
                time: 100,
                number: 5,
                flags: 1,
                result: Ok(vec!["one".into(), "".into(), "three".into()])
            })]
        );
        assert!(
            evs.iter()
                .all(|e| matches!(e, Event::Reply(r) if r.is_ours()))
        );
    }

    #[test]
    fn empty_block() {
        let mut d = Decoder::default();
        let evs = feed_all(&mut d, &["%begin 1 2 0", "%end 1 2 0"]);
        assert!(matches!(&evs[0], Event::Reply(r) if r.result == Ok(vec![]) && !r.is_ours()));
    }

    #[test]
    fn error_block() {
        let mut d = Decoder::default();
        let evs = feed_all(
            &mut d,
            &["%begin 7 9 1", "unknown command: nope", "%error 7 9 1"],
        );
        assert_eq!(
            evs,
            vec![Event::Reply(Reply {
                time: 7,
                number: 9,
                flags: 1,
                result: Err(vec!["unknown command: nope".into()])
            })]
        );
    }

    #[test]
    fn percent_lines_inside_a_block_are_output() {
        // Pane ids and even notification look-alikes (a capture-pane of a pane
        // showing tmux's own protocol) are command output, not events.
        let mut d = Decoder::default();
        let evs = feed_all(
            &mut d,
            &[
                "%window-add @1",
                "%begin 10 3 1",
                "%0 80x24",
                "%output %1 forged",
                "%exit",
                "%end 10 99 1",
                "%end 10 3 1",
                "%window-close @1",
            ],
        );
        assert_eq!(evs.len(), 3);
        assert_eq!(
            evs[0],
            Event::Notification(Notification::WindowAdd {
                window: "@1".into()
            })
        );
        assert_eq!(
            evs[1],
            Event::Reply(Reply {
                time: 10,
                number: 3,
                flags: 1,
                result: Ok(vec![
                    "%0 80x24".into(),
                    "%output %1 forged".into(),
                    "%exit".into(),
                    "%end 10 99 1".into()
                ])
            })
        );
        assert_eq!(
            evs[2],
            Event::Notification(Notification::WindowClose {
                window: "@1".into()
            })
        );
    }

    #[test]
    fn back_to_back_blocks_and_notifications_between() {
        let mut d = Decoder::default();
        let evs = feed_all(
            &mut d,
            &[
                "%begin 1 1 0",
                "%end 1 1 0",
                "%session-changed $1 view",
                "%begin 2 2 1",
                "x",
                "%end 2 2 1",
                "%sessions-changed",
                "%begin 3 3 1",
                "boom",
                "%error 3 3 1",
                "%exit",
            ],
        );
        assert_eq!(evs.len(), 6);
        assert!(matches!(&evs[0], Event::Reply(r) if !r.is_ours()));
        assert!(matches!(
            &evs[1],
            Event::Notification(Notification::SessionChanged { .. })
        ));
        assert!(matches!(&evs[2], Event::Reply(r) if r.result == Ok(vec!["x".into()])));
        assert!(matches!(
            &evs[3],
            Event::Notification(Notification::SessionsChanged)
        ));
        assert!(matches!(&evs[4], Event::Reply(r) if r.result.is_err()));
        assert!(matches!(
            &evs[5],
            Event::Notification(Notification::Exit { reason: None })
        ));
    }

    #[test]
    fn crlf_and_dcs_wrapping() {
        let mut d = Decoder::default();
        let evs = feed_all(
            &mut d,
            &[
                "\x1bP1000p%begin 5 1 1\r",
                "line\r",
                "%end 5 1 1\r",
                "%window-add @2\r",
                "%exit\x1b\\",
            ],
        );
        assert_eq!(evs.len(), 3);
        assert!(matches!(&evs[0], Event::Reply(r) if r.result == Ok(vec!["line".into()])));
        assert_eq!(
            evs[1],
            Event::Notification(Notification::WindowAdd {
                window: "@2".into()
            })
        );
        assert_eq!(
            evs[2],
            Event::Notification(Notification::Exit { reason: None })
        );
    }

    #[test]
    fn blank_lines_outside_blocks_are_ignored() {
        let mut d = Decoder::default();
        assert!(d.feed(b"").is_none());
        assert!(d.feed(b"\r").is_none());
    }

    #[test]
    fn truncated_block_is_visible() {
        let mut d = Decoder::default();
        d.feed(b"%begin 1 1 1");
        d.feed(b"partial");
        assert!(d.in_block());
    }

    #[test]
    fn invalid_utf8_in_blocks_is_lossy_but_output_is_raw() {
        let mut d = Decoder::default();
        d.feed(b"%begin 1 1 1");
        d.feed(b"bad \xff byte");
        let Some(Event::Reply(r)) = d.feed(b"%end 1 1 1") else {
            panic!()
        };
        assert_eq!(r.result.unwrap()[0], "bad \u{fffd} byte");
        assert_eq!(
            Decoder::default().feed(b"%output %1 \xff\\012"),
            Some(Event::Notification(Notification::Output {
                pane: "%1".into(),
                data: vec![0xff, b'\n']
            }))
        );
    }
}
