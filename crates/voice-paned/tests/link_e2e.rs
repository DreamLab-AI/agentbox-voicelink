//! The client's real controller link (`voicelink::link`) and the wizard's
//! handshake (`voicelink::setup::remote`) against the real `voice-paned`
//! binary, over its stdin/stdout, on a private tmux server addressed by
//! socket name (`--socket-name`). This is the publishing gate: nothing else
//! drives the two halves of the wire contract together.
//!
//! SAFETY: only `Server` (a private `tmux -L` server under its own
//! `TMUX_TMPDIR`) is ever touched; the tests skip, with a message, when tmux
//! is not installed.

mod common;

use std::io;
use std::time::Duration;

use async_trait::async_trait;
use common::{Server, TIMEOUT};
use tokio::sync::mpsc::UnboundedReceiver;
use voicelink_client::link::{
    Connector, LinkConfig, LinkError, LinkHandle, LinkNotice, Transport, spawn,
};
use voicelink_client::setup::remote;
use voicelink_proto::{AgentKind, ErrorCode, EventKind, Focus, Key};

macro_rules! server {
    () => {
        match Server::start_named() {
            Some(s) => s,
            None => return,
        }
    };
}

/// Starts `voice-paned --stdio --socket-name <private> --spool <tmp>` per connection.
struct PanedConnector {
    command: std::process::Command,
}

#[async_trait]
impl Connector for PanedConnector {
    async fn connect(&self) -> io::Result<Transport> {
        let mut cmd = tokio::process::Command::from(clone_command(&self.command));
        let mut child = cmd
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true)
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("stdout"))?;
        Ok(Transport {
            reader: Box::new(stdout),
            writer: Box::new(stdin),
            keepalive: Some(Box::new(child)),
        })
    }
}

fn clone_command(c: &std::process::Command) -> std::process::Command {
    let mut n = std::process::Command::new(c.get_program());
    n.args(c.get_args());
    for (k, v) in c.get_envs() {
        match v {
            Some(v) => n.env(k, v),
            None => n.env_remove(k),
        };
    }
    n
}

fn config(view: &str) -> LinkConfig {
    LinkConfig {
        client: "voicelink/e2e".into(),
        view_session: view.into(),
        request_timeout: TIMEOUT,
        hello_timeout: TIMEOUT,
        backoff_min: Duration::from_millis(50),
        backoff_max: Duration::from_millis(200),
    }
}

async fn notice(rx: &mut UnboundedReceiver<LinkNotice>) -> LinkNotice {
    tokio::time::timeout(TIMEOUT, rx.recv())
        .await
        .expect("a notice in time")
        .expect("link open")
}

/// Next focus push matching `pred`, skipping other notices.
async fn focus_where(
    rx: &mut UnboundedReceiver<LinkNotice>,
    pred: impl Fn(&Focus) -> bool,
) -> Focus {
    loop {
        if let LinkNotice::Focus(f) = notice(rx).await
            && pred(&f)
        {
            return f;
        }
    }
}

async fn connect(srv: &Server, view: &str) -> (LinkHandle, UnboundedReceiver<LinkNotice>, Focus) {
    let (h, mut rx, _task) = spawn(
        PanedConnector {
            command: srv.controller_command(&["--session", "agentbox"]),
        },
        config(view),
    );
    let LinkNotice::Connected(focus) = notice(&mut rx).await else {
        panic!("expected Connected first")
    };
    (h, rx, focus)
}

fn view_pane(srv: &Server, view: &str) -> String {
    srv.out(&[
        "display-message",
        "-p",
        "-t",
        &format!("={view}:"),
        "#{pane_id}",
    ])
}

#[tokio::test]
async fn real_link_against_real_controller() {
    let srv = server!();
    let view = "remote-e2e";
    let (h, mut rx, focus) = connect(&srv, view).await;

    // Hello -> Welcome carries the view's focus.
    assert_eq!((focus.window, focus.window_name.as_str()), (0, "one"));
    assert_eq!(focus.command, "cat");
    assert_eq!(focus.agent, AgentKind::Other);
    assert_eq!(focus.pane, srv.panes_of(0)[0]);
    h.ping().await.unwrap();
    let listed = h.list_panes().await.unwrap();
    assert_eq!(listed.len(), 4);
    assert!(listed.iter().any(|p| p.focused && p.pane == focus.pane));

    // Focus is pushed after a select-window on the private server.
    srv.ok(&["select-window", "-t", &format!("={view}:1")]);
    let moved = focus_where(&mut rx, |f| f.window == 1).await;
    assert_eq!(moved.window_name, "two words");
    let a = moved.pane.clone();
    assert!(srv.panes_of(1).contains(&a));

    // Send lands in the focused pane, and only there.
    h.send("hello-link", true, false).await.unwrap();
    srv.wait_capture(&a, "typed text", |c| c.matches("hello-link").count() >= 2);
    for other in srv.panes_of(0).into_iter().chain(srv.panes_of(2)) {
        assert!(!srv.capture(&other).contains("hello-link"));
    }

    // Reads may address any pane.
    assert!(
        h.read_pane(&a, 20).await.unwrap().contains("hello-link"),
        "read_pane of the focused pane"
    );

    // SelectPane: first to another window, then to the non-active pane of
    // window 1. The view's window and pane move; the base session's does not.
    let w2 = srv.panes_of(2)[0].clone();
    h.select_pane(&w2).await.unwrap();
    focus_where(&mut rx, |f| f.window == 2).await;
    assert_eq!(srv.view_window(view), "2");

    let b = srv.panes_of(1).into_iter().find(|p| *p != a).unwrap();
    h.select_pane(&b).await.unwrap();
    let f = focus_where(&mut rx, |f| f.pane == b).await;
    assert_eq!(f.window, 1);
    assert_eq!(srv.view_window(view), "1");
    assert_eq!(view_pane(&srv, view), b);
    assert_eq!(srv.base_window(), "0", "the base session must not move");

    // The recorder's event reaches the client.
    let payload = r#"{"type":"agent-turn-complete","thread-id":"t","turn-id":"u","cwd":"/w","client":"codex_exec","input-messages":["hi"],"last-assistant-message":"probe ok"}"#;
    let out = srv.record(
        &[
            "--source",
            "codex",
            "--spool",
            srv.spool_path().to_str().unwrap(),
            payload,
        ],
        None,
        Some(&b),
    );
    assert!(out.status.success());
    let event = loop {
        if let LinkNotice::Event(e) = notice(&mut rx).await {
            break e;
        }
    };
    assert_eq!(
        (event.pane.as_str(), event.agent, event.kind),
        (b.as_str(), AgentKind::Codex, EventKind::TurnEnd)
    );
    assert_eq!(event.text.as_deref(), Some("probe ok"));

    // Dropping the handle ends the session cleanly.
    drop(h);
}

#[tokio::test]
async fn forced_send_to_an_agent_menu_is_refused_through_the_real_link() {
    let srv = server!();
    let view = "remote-menu";
    let agent = srv.fake_agent("codex", "agent");
    agent.draw("  Would you like to run the following command?\r\n\r\n  $ cargo test\r\n\r\n\u{203a} 1. Yes, proceed (y)\r\n  2. No, and tell Codex what to do differently (esc)\r\n");
    srv.wait_capture(&agent.pane, "the menu", |c| c.contains("1. Yes"));
    let (h, mut rx, _) = connect(&srv, view).await;

    h.select_pane(&agent.pane).await.unwrap();
    let f = focus_where(&mut rx, |f| f.pane == agent.pane).await;
    assert_eq!(f.agent, AgentKind::Codex);
    let before = srv.capture(&agent.pane);

    for force in [false, true] {
        let err = h
            .send("yes, run the tests", true, force)
            .await
            .expect_err("a menu must refuse Send");
        assert!(
            matches!(
                err,
                LinkError::Controller {
                    code: ErrorCode::PromptActive,
                    ..
                }
            ),
            "force={force}: {err:?}"
        );
    }
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(
        srv.capture(&agent.pane),
        before,
        "the menu must be untouched"
    );

    // The menu is answered with Keys.
    h.keys(vec![Key::Digit(2)]).await.unwrap();
    srv.wait_capture(&agent.pane, "the echoed digit", |c| c != before);
}

#[test]
fn wizard_handshake_against_the_real_controller() {
    let srv = server!();
    let command = srv.controller_command(&["--session", "agentbox"]);
    let welcome = remote::handshake_with(command, "remote-wizard").expect("handshake");
    assert!(welcome.server.starts_with("voice-paned/"), "{welcome:?}");
    assert_eq!(welcome.focus, "0:one cat");
    // A bad view name is refused with a readable reason.
    let command = srv.controller_command(&["--session", "agentbox"]);
    let err = remote::handshake_with(command, "agentbox").unwrap_err();
    assert!(err.contains("refused"), "{err}");
    // The handshake leaves the server and base session alone.
    assert_eq!(srv.base_window(), "0");
}
