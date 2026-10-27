//! Typed tmux queries and actions for one view session, over a control client.

use tmux_control::{Client, Error, command_line};
use voicelink_proto::{AgentKind, Focus, PaneId};

/// One row of `list-panes -s`.
#[derive(Debug, Clone)]
pub struct PaneRow {
    pub pane: PaneId,
    pub window: u32,
    pub window_id: String,
    pub active: bool,
    pub command: String,
    pub window_name: String,
}

/// A control client attached to the view session.
pub struct Tmux {
    pub client: Client,
    pub view: String,
}

/// `true` for ids of the form `%<digits>`.
pub fn valid_pane_id(id: &str) -> bool {
    id.strip_prefix('%')
        .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

impl Tmux {
    /// Exact-match target for the view session's current window and pane.
    pub fn view_target(&self) -> String {
        format!("={}:", self.view)
    }

    pub async fn cmd<I, S>(&self, words: I) -> Result<Vec<String>, Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.client.command(command_line(words)).await
    }

    /// The view session's current window and its active pane.
    pub async fn focus(&self) -> Result<Option<Focus>, Error> {
        let fmt = "#{window_index}\t#{pane_id}\t#{pane_current_command}\t#{window_name}";
        let lines = self
            .cmd(["display-message", "-p", "-t", &self.view_target(), fmt])
            .await?;
        Ok(lines.first().and_then(|l| parse_focus(l)))
    }

    /// Every pane of the session group, window then pane order.
    pub async fn panes(&self) -> Result<Vec<PaneRow>, Error> {
        let fmt = "#{pane_id}\t#{window_index}\t#{window_id}\t#{pane_active}\t#{pane_current_command}\t#{window_name}";
        let lines = self
            .cmd(["list-panes", "-s", "-t", &self.view_target(), "-F", fmt])
            .await?;
        Ok(lines.iter().filter_map(|l| parse_pane_row(l)).collect())
    }

    /// `capture-pane -p` of the visible screen plus `history` lines above it;
    /// `ansi` adds `-e`.
    pub async fn capture(
        &self,
        pane: &str,
        history: u32,
        ansi: bool,
    ) -> Result<Vec<String>, Error> {
        let start = format!("-{history}");
        let mut args = vec!["capture-pane", "-p", "-t", pane];
        if history > 0 {
            args.extend(["-S", &start]);
        }
        if ansi {
            args.push("-e");
        }
        self.cmd(args).await
    }

    /// Plain text of the cursor's row, up to the cursor column.
    pub async fn cursor_row_text(&self, pane: &str) -> Result<String, Error> {
        let pos = self
            .cmd([
                "display-message",
                "-p",
                "-t",
                pane,
                "#{cursor_x}\t#{cursor_y}",
            ])
            .await?;
        let (x, y) = pos
            .first()
            .and_then(|l| l.split_once('\t'))
            .and_then(|(x, y)| Some((x.parse::<usize>().ok()?, y.parse::<usize>().ok()?)))
            .unwrap_or((0, 0));
        let screen = self.capture(pane, 0, false).await?;
        Ok(screen
            .get(y)
            .map(|row| row.chars().take(x).collect())
            .unwrap_or_default())
    }
}

/// Classifies a pane's foreground command.
pub fn agent_of(command: &str) -> AgentKind {
    AgentKind::from_command(command)
}

fn parse_focus(line: &str) -> Option<Focus> {
    let mut it = line.splitn(4, '\t');
    let window = it.next()?.parse().ok()?;
    let pane = it.next()?.to_owned();
    let command = it.next()?.to_owned();
    let window_name = it.next().unwrap_or_default().to_owned();
    if !valid_pane_id(&pane) {
        return None;
    }
    let agent = agent_of(&command);
    Some(Focus {
        window,
        window_name,
        pane,
        command,
        agent,
    })
}

fn parse_pane_row(line: &str) -> Option<PaneRow> {
    let mut it = line.splitn(6, '\t');
    let pane = it.next()?.to_owned();
    let window = it.next()?.parse().ok()?;
    let window_id = it.next()?.to_owned();
    let active = it.next()? == "1";
    let command = it.next()?.to_owned();
    let window_name = it.next().unwrap_or_default().to_owned();
    valid_pane_id(&pane).then_some(PaneRow {
        pane,
        window,
        window_id,
        active,
        command,
        window_name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pane_id_validation() {
        assert!(valid_pane_id("%0"));
        assert!(valid_pane_id("%123"));
        for bad in ["", "%", "12", "%1a", "%1 ", "%1;kill-server", "@1", "%-1"] {
            assert!(!valid_pane_id(bad), "{bad:?}");
        }
    }

    #[test]
    fn focus_parsing_keeps_spaces_and_tabs_in_the_name() {
        let f = parse_focus("3\t%9\tclaude\tmy window\tx").unwrap();
        assert_eq!(
            (f.window, f.pane.as_str(), f.command.as_str()),
            (3, "%9", "claude")
        );
        assert_eq!(f.window_name, "my window\tx");
        assert_eq!(f.agent, AgentKind::Claude);
        assert!(parse_focus("garbage").is_none());
        assert!(parse_focus("1\tnotapane\tfish\tn").is_none());
    }

    #[test]
    fn pane_row_parsing() {
        let r = parse_pane_row("%2\t1\t@4\t1\tcodex\tServices").unwrap();
        assert_eq!(
            (r.pane.as_str(), r.window, r.window_id.as_str(), r.active),
            ("%2", 1, "@4", true)
        );
        assert_eq!(r.window_name, "Services");
        assert!(parse_pane_row("%2\tx\t@4\t1\tc\tn").is_none());
    }
}
