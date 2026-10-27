//! End-to-end tests: the built `voice-paned` binary against a private tmux
//! server, driven with JSON lines over stdin/stdout.

mod common;

use common::{Server, TIMEOUT};
use voicelink_proto::{
    AgentKind, ClientMessage as C, ErrorCode, EventKind, Key, PROTOCOL_VERSION, ServerMessage as S,
};

macro_rules! server {
    () => {
        match Server::start() {
            Some(s) => s,
            None => return,
        }
    };
}

fn err_code(m: &S) -> ErrorCode {
    match m {
        S::Error { code, .. } => *code,
        other => panic!("expected an error, got {other:?}"),
    }
}

#[tokio::test]
async fn hello_welcome_ping_and_listing() {
    let srv = server!();
    let mut ctl = srv.controller();
    let focus = ctl.handshake("view-a").await;
    assert_eq!(
        (
            focus.window,
            focus.window_name.as_str(),
            focus.command.as_str()
        ),
        (0, "one", "cat")
    );
    assert_eq!(focus.agent, AgentKind::Other);
    assert_eq!(focus.pane, srv.panes_of(0)[0]);

    assert_eq!(ctl.request(C::Ping { id: 1 }, 1).await, S::Pong { id: 1 });

    let S::Panes { panes, .. } = ctl.request(C::ListPanes { id: 2 }, 2).await else {
        panic!()
    };
    assert_eq!(panes.len(), 4, "every pane of the group: {panes:?}");
    assert_eq!(
        panes.iter().map(|p| p.window).collect::<Vec<_>>(),
        [0, 1, 1, 2]
    );
    assert_eq!(panes.iter().filter(|p| p.focused).count(), 1);
    assert!(panes.iter().find(|p| p.focused).unwrap().pane == focus.pane);
    assert_eq!(panes[1].window_name, "two words");
    assert_eq!(
        panes.iter().filter(|p| p.window == 1 && p.active).count(),
        1
    );

    // The view is a grouped session: it sees the same windows.
    assert!(srv.sessions().contains(&"view-a".to_owned()));
    let status = ctl.close_and_wait().await;
    assert!(status.success());
}

#[tokio::test]
async fn handshake_errors() {
    let srv = server!();

    let mut ctl = srv.controller();
    ctl.send(&C::Hello {
        proto: PROTOCOL_VERSION + 1,
        client: "t".into(),
        view_session: "v".into(),
    })
    .await;
    assert_eq!(err_code(&ctl.recv().await), ErrorCode::ProtocolMismatch);
    assert!(!ctl.close_and_wait().await.success());

    let mut ctl = srv.controller();
    ctl.send(&C::ListPanes { id: 1 }).await;
    assert_eq!(err_code(&ctl.recv().await), ErrorCode::BadHandshake);
    ctl.close_and_wait().await;

    let mut ctl = srv.controller();
    ctl.send_raw("not json at all\n").await;
    assert_eq!(err_code(&ctl.recv().await), ErrorCode::BadHandshake);
    ctl.close_and_wait().await;

    // Unusable view names are refused without touching tmux.
    for bad in ["a.b", "a:b", "agentbox", ""] {
        let mut ctl = srv.controller();
        ctl.send(&C::Hello {
            proto: PROTOCOL_VERSION,
            client: "t".into(),
            view_session: bad.into(),
        })
        .await;
        assert_eq!(
            err_code(&ctl.recv().await),
            ErrorCode::BadRequest,
            "{bad:?}"
        );
        ctl.close_and_wait().await;
    }
    assert_eq!(srv.sessions(), ["agentbox"]);

    // A second hello is refused but the session continues.
    let mut ctl = srv.controller();
    ctl.handshake("v2").await;
    ctl.send(&C::Hello {
        proto: PROTOCOL_VERSION,
        client: "t".into(),
        view_session: "v2".into(),
    })
    .await;
    assert_eq!(err_code(&ctl.recv().await), ErrorCode::BadHandshake);
    assert_eq!(ctl.request(C::Ping { id: 5 }, 5).await, S::Pong { id: 5 });
}

#[tokio::test]
async fn missing_base_session_is_reported() {
    let srv = server!();
    let mut ctl = srv.controller_with(&["--session", "nope"]);
    ctl.send(&C::Hello {
        proto: PROTOCOL_VERSION,
        client: "t".into(),
        view_session: "v".into(),
    })
    .await;
    assert_eq!(err_code(&ctl.recv().await), ErrorCode::Tmux);
    assert!(!ctl.close_and_wait().await.success());
    assert_eq!(
        srv.sessions(),
        ["agentbox"],
        "no stray session must be created"
    );
}

#[tokio::test]
async fn invalid_requests_return_bad_request() {
    let srv = server!();
    let mut ctl = srv.controller();
    ctl.handshake("v").await;

    ctl.send_raw("garbage\n").await;
    assert_eq!(err_code(&ctl.recv().await), ErrorCode::BadRequest);

    ctl.send_raw("{\"type\":\"send\",\"id\":42,\"text\":5}\n")
        .await;
    match ctl.recv().await {
        S::Error { id, code, .. } => assert_eq!((id, code), (Some(42), ErrorCode::BadRequest)),
        other => panic!("{other:?}"),
    }

    let long = format!(
        "{{\"type\":\"ping\",\"id\":1,\"pad\":\"{}\"}}\n",
        "x".repeat(voicelink_proto::MAX_LINE_BYTES)
    );
    ctl.send_raw(&long).await;
    assert_eq!(err_code(&ctl.recv().await), ErrorCode::BadRequest);

    assert_eq!(ctl.request(C::Ping { id: 9 }, 9).await, S::Pong { id: 9 });
}

#[tokio::test]
async fn focus_is_pushed_when_the_view_window_changes() {
    let srv = server!();
    let mut ctl = srv.controller();
    let f0 = ctl.handshake("view-f").await;
    assert_eq!(f0.window, 0);

    srv.ok(&["select-window", "-t", "=view-f:1"]);
    let f1 = ctl
        .recv_until(|m| {
            if let S::Focus(f) = m {
                Some(f.clone())
            } else {
                None
            }
        })
        .await;
    assert_eq!((f1.window, f1.window_name.as_str()), (1, "two words"));
    assert_eq!(
        f1.pane,
        srv.out(&["display-message", "-p", "-t", "agentbox:1", "#{pane_id}"])
    );
    assert_eq!(
        srv.base_window(),
        "0",
        "base session must not follow the view"
    );

    // Changing the active pane of the current window is a focus change too.
    let other = srv.panes_of(1).into_iter().find(|p| *p != f1.pane).unwrap();
    srv.ok(&["select-pane", "-t", &other]);
    let f2 = ctl
        .recv_until(|m| {
            if let S::Focus(f) = m {
                Some(f.clone())
            } else {
                None
            }
        })
        .await;
    assert_eq!((f2.window, f2.pane.as_str()), (1, other.as_str()));

    // Identical focus is not pushed again: the next push is for window 2.
    srv.ok(&["select-pane", "-t", &other]);
    srv.ok(&["select-window", "-t", "=view-f:2"]);
    let f3 = ctl
        .recv_until(|m| {
            if let S::Focus(f) = m {
                Some(f.clone())
            } else {
                None
            }
        })
        .await;
    assert_eq!(f3.window, 2);
}

#[tokio::test]
async fn send_lands_only_in_the_focused_pane_and_is_sanitised() {
    let srv = server!();
    let mut ctl = srv.controller();
    let focus = ctl.handshake("view-s").await;
    let others: Vec<String> = (0..=2)
        .flat_map(|w| srv.panes_of(w))
        .filter(|p| *p != focus.pane)
        .collect();

    let reply = ctl
        .request(
            C::Send {
                id: 1,
                text: "hello\nworld\r\n; kill-server".into(),
                submit: true,
                force: false,
            },
            1,
        )
        .await;
    assert_eq!(reply, S::Ack { id: 1 });
    // One submission only: cat echoes the typed line once and prints it once.
    let screen = srv.wait_capture(&focus.pane, "echoed text", |c| {
        c.matches("hello world").count() >= 2
    });
    assert!(
        screen.contains("hello world  ; kill-server"),
        "newlines become spaces: {screen}"
    );
    assert_eq!(
        screen.matches("hello world").count(),
        2,
        "an embedded newline must not submit early"
    );
    for p in &others {
        assert!(!srv.capture(p).contains("hello"), "{p} must stay untouched");
    }
    assert!(
        srv.try_out(&["list-sessions"]).is_some(),
        "server must survive injection attempts"
    );
}

#[tokio::test]
async fn input_not_empty_guard_and_force() {
    let srv = server!();
    let mut ctl = srv.controller();
    let focus = ctl.handshake("view-i").await;

    assert_eq!(
        ctl.request(
            C::Send {
                id: 1,
                text: "abc".into(),
                submit: false,
                force: false
            },
            1
        )
        .await,
        S::Ack { id: 1 }
    );
    srv.wait_capture(&focus.pane, "abc", |c| c.contains("abc"));

    let r = ctl
        .request(
            C::Send {
                id: 2,
                text: "def".into(),
                submit: false,
                force: false,
            },
            2,
        )
        .await;
    assert_eq!(err_code(&r), ErrorCode::InputNotEmpty);
    assert!(
        !srv.capture(&focus.pane).contains("def"),
        "refused text must not be typed"
    );

    assert_eq!(
        ctl.request(
            C::Send {
                id: 3,
                text: "def".into(),
                submit: false,
                force: true
            },
            3
        )
        .await,
        S::Ack { id: 3 }
    );
    srv.wait_capture(&focus.pane, "abcdef", |c| c.contains("abcdef"));

    // Submitting empties the line; a plain send works again.
    assert_eq!(
        ctl.request(
            C::Keys {
                id: 4,
                keys: vec![Key::Enter]
            },
            4
        )
        .await,
        S::Ack { id: 4 }
    );
    srv.wait_capture(&focus.pane, "submitted line", |c| {
        c.matches("abcdef").count() >= 2
    });
    assert_eq!(
        ctl.request(
            C::Send {
                id: 5,
                text: "ghi".into(),
                submit: true,
                force: false
            },
            5
        )
        .await,
        S::Ack { id: 5 }
    );
    srv.wait_capture(&focus.pane, "ghi", |c| c.matches("ghi").count() >= 2);

    // Nothing to type at all.
    let r = ctl
        .request(
            C::Send {
                id: 6,
                text: "\n".into(),
                submit: false,
                force: true,
            },
            6,
        )
        .await;
    assert!(matches!(r, S::Ack { .. }) || err_code(&r) == ErrorCode::BadRequest);
}

#[tokio::test]
async fn keys_interrupt_and_bad_keys() {
    let srv = server!();
    let mut ctl = srv.controller();
    let focus = ctl.handshake("view-k").await;

    let r = ctl
        .request(
            C::Keys {
                id: 1,
                keys: vec![Key::Letter('y'), Key::Digit(12), Key::Enter],
            },
            1,
        )
        .await;
    assert_eq!(err_code(&r), ErrorCode::BadRequest);
    let r = ctl
        .request(
            C::Keys {
                id: 2,
                keys: vec![Key::Letter(';')],
            },
            2,
        )
        .await;
    assert_eq!(err_code(&r), ErrorCode::BadRequest);
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(
        srv.capture(&focus.pane).trim().is_empty(),
        "a rejected key list must send nothing"
    );

    let r = ctl
        .request(
            C::Keys {
                id: 3,
                keys: vec![
                    Key::Letter('y'),
                    Key::Digit(2),
                    Key::Space,
                    Key::Letter('Z'),
                    Key::Enter,
                ],
            },
            3,
        )
        .await;
    assert_eq!(r, S::Ack { id: 3 });
    srv.wait_capture(&focus.pane, "typed keys", |c| {
        c.matches("y2 Z").count() >= 2
    });

    assert_eq!(
        ctl.request(C::Interrupt { id: 4 }, 4).await,
        S::Ack { id: 4 }
    );
    let r = ctl
        .request(
            C::Keys {
                id: 5,
                keys: vec![],
            },
            5,
        )
        .await;
    assert_eq!(err_code(&r), ErrorCode::BadRequest);
}

#[tokio::test]
async fn select_pane_moves_the_view_not_the_base() {
    let srv = server!();
    let mut ctl = srv.controller();
    let f0 = ctl.handshake("view-p").await;
    assert_eq!(f0.window, 0);

    let w1 = srv.panes_of(1);
    let target = w1[1].clone();
    assert_eq!(
        ctl.request(
            C::SelectPane {
                id: 1,
                pane: target.clone()
            },
            1
        )
        .await,
        S::Ack { id: 1 }
    );
    let focus = ctl
        .recv_until(|m| {
            if let S::Focus(f) = m {
                Some(f.clone())
            } else {
                None
            }
        })
        .await;
    assert_eq!((focus.window, focus.pane.as_str()), (1, target.as_str()));
    assert_eq!(srv.view_window("view-p"), "1");
    assert_eq!(
        srv.base_window(),
        "0",
        "SelectPane must not change the base session's current window"
    );

    // Focus-only writes now go to the selected pane.
    assert_eq!(
        ctl.request(
            C::Send {
                id: 2,
                text: "routed".into(),
                submit: true,
                force: false
            },
            2
        )
        .await,
        S::Ack { id: 2 }
    );
    srv.wait_capture(&target, "routed text", |c| c.contains("routed"));
    assert!(!srv.capture(&w1[0]).contains("routed"));
    assert!(!srv.capture(&srv.panes_of(0)[0]).contains("routed"));

    for bad in ["%9999", "%", "x", "%1;kill-server", "@1"] {
        let r = ctl
            .request(
                C::SelectPane {
                    id: 3,
                    pane: bad.into(),
                },
                3,
            )
            .await;
        assert_eq!(err_code(&r), ErrorCode::NotFound, "{bad}");
    }
    assert!(srv.try_out(&["list-sessions"]).is_some());
}

#[tokio::test]
async fn read_pane_reads_any_pane() {
    let srv = server!();
    let mut ctl = srv.controller();
    let focus = ctl.handshake("view-r").await;
    let background = srv.panes_of(2)[0].clone();
    assert_ne!(background, focus.pane);
    srv.ok(&["send-keys", "-t", &background, "-l", "from the background"]);
    srv.ok(&["send-keys", "-t", &background, "Enter"]);
    srv.wait_capture(&background, "text", |c| {
        c.matches("from the background").count() >= 2
    });

    let S::PaneText { pane, text, .. } = ctl
        .request(
            C::ReadPane {
                id: 1,
                pane: background.clone(),
                lines: 50,
            },
            1,
        )
        .await
    else {
        panic!()
    };
    assert_eq!(pane, background);
    assert_eq!(
        text.lines().count(),
        2,
        "trailing blank lines are removed: {text:?}"
    );
    assert!(text.lines().all(|l| l == "from the background"));

    // Lines are an upper bound on the tail.
    let S::PaneText { text, .. } = ctl
        .request(
            C::ReadPane {
                id: 2,
                pane: background.clone(),
                lines: 1,
            },
            2,
        )
        .await
    else {
        panic!()
    };
    assert_eq!(text, "from the background");

    // Absurd line counts are clamped, not errors.
    assert!(matches!(
        ctl.request(
            C::ReadPane {
                id: 3,
                pane: background,
                lines: u32::MAX
            },
            3
        )
        .await,
        S::PaneText { .. }
    ));

    for bad in ["%9999", "%1;kill-server", "nonsense"] {
        let r = ctl
            .request(
                C::ReadPane {
                    id: 4,
                    pane: bad.into(),
                    lines: 10,
                },
                4,
            )
            .await;
        assert_eq!(err_code(&r), ErrorCode::NotFound, "{bad}");
    }
}

fn next_event(m: &S) -> Option<voicelink_proto::AgentEvent> {
    if let S::Event(e) = m {
        Some(e.clone())
    } else {
        None
    }
}

#[tokio::test]
async fn recorder_events_flow_through_the_tail_including_rotation() {
    let srv = server!();
    let mut ctl = srv.controller();
    ctl.handshake("view-e").await;
    let pane = srv.panes_of(1)[1].clone();

    let stop = r#"{"session_id":"s","transcript_path":"/t","cwd":"/w","prompt_id":"p","permission_mode":"default","effort":{"level":"high"},"hook_event_name":"Stop","stop_hook_active":false,"last_assistant_message":"Build is green.","background_tasks":[],"session_crons":[]}"#;
    let out = srv.record(
        &[
            "--source",
            "claude",
            "--spool",
            srv.spool_path().to_str().unwrap(),
        ],
        Some(stop),
        Some(&pane),
    );
    assert!(out.status.success());
    assert!(
        out.stdout.is_empty(),
        "recorder stdout reaches the model: {:?}",
        out.stdout
    );
    assert!(
        srv.spool_path().exists(),
        "recorder wrote nothing; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let e = ctl.recv_until(next_event).await;
    assert_eq!(
        (e.pane.as_str(), e.window, e.agent, e.kind),
        (
            pane.as_str(),
            Some(1),
            AgentKind::Claude,
            EventKind::TurnEnd
        )
    );
    assert_eq!(e.text.as_deref(), Some("Build is green."));

    let codex = r#"{"type":"agent-turn-complete","thread-id":"t","turn-id":"u","cwd":"/w","client":"codex_exec","input-messages":["hi"],"last-assistant-message":"probe ok"}"#;
    let out = srv.record(
        &[
            "--source",
            "codex",
            "--spool",
            srv.spool_path().to_str().unwrap(),
            codex,
        ],
        None,
        Some(&pane),
    );
    assert!(out.status.success() && out.stdout.is_empty());
    let e = ctl.recv_until(next_event).await;
    assert_eq!(
        (e.agent, e.kind, e.text.as_deref()),
        (AgentKind::Codex, EventKind::TurnEnd, Some("probe ok"))
    );

    // Notification types.
    let prompt = r#"{"hook_event_name":"Notification","notification_type":"permission_prompt","message":"Claude needs your permission to use Bash"}"#;
    srv.record(
        &[
            "--source",
            "claude",
            "--spool",
            srv.spool_path().to_str().unwrap(),
        ],
        Some(prompt),
        Some(&pane),
    );
    let idle = r#"{"hook_event_name":"Notification","notification_type":"idle_prompt","message":"waiting"}"#;
    srv.record(
        &[
            "--source",
            "claude",
            "--spool",
            srv.spool_path().to_str().unwrap(),
        ],
        Some(idle),
        Some(&pane),
    );
    let e = ctl.recv_until(next_event).await;
    assert_eq!(
        (e.kind, e.text.as_deref()),
        (
            EventKind::Prompt,
            Some("Claude needs your permission to use Bash")
        )
    );

    // Rotation: move the file aside; the next record creates a fresh one.
    let spool = srv.spool_path();
    std::fs::rename(&spool, spool.with_file_name("events.jsonl.1")).unwrap();
    let after = r#"{"type":"agent-turn-complete","last-assistant-message":"after rotation"}"#;
    srv.record(
        &[
            "--source",
            "codex",
            "--spool",
            spool.to_str().unwrap(),
            after,
        ],
        None,
        Some(&pane),
    );
    let e = ctl.recv_until(next_event).await;
    assert_eq!(e.text.as_deref(), Some("after rotation"));

    // Truncation.
    std::fs::write(&spool, "").unwrap();
    let again = r#"{"type":"agent-turn-complete","last-assistant-message":"after truncation"}"#;
    srv.record(
        &[
            "--source",
            "codex",
            "--spool",
            spool.to_str().unwrap(),
            again,
        ],
        None,
        Some(&pane),
    );
    let e = ctl.recv_until(next_event).await;
    assert_eq!(e.text.as_deref(), Some("after truncation"));
    // The idle_prompt notice was skipped: nothing else is pending.
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(600), ctl.try_recv())
            .await
            .is_err(),
        "no further events expected"
    );
}

#[tokio::test]
async fn recorder_is_silent_and_never_fails() {
    let srv = server!();
    let spool = srv.spool_path();
    let sp = spool.to_str().unwrap();

    // No pane: nothing written, exit 0, no output.
    let out = srv.record(
        &["--source", "claude", "--spool", sp],
        Some(r#"{"hook_event_name":"Stop"}"#),
        None,
    );
    assert!(out.status.success() && out.stdout.is_empty() && out.stderr.is_empty());
    assert!(!spool.exists());

    // Garbage payloads and bad arguments.
    let pane = srv.panes_of(0)[0].clone();
    for (args, stdin) in [
        (
            vec!["--source", "claude", "--spool", sp],
            "this is not json",
        ),
        (
            vec!["--source", "claude", "--spool", sp],
            r#"{"hook_event_name":"PreToolUse"}"#,
        ),
        (vec!["--source", "codex", "--spool", sp, "{broken"], ""),
        (vec!["--source", "nonsense"], ""),
        (vec!["--bogus-flag"], ""),
        (vec![], ""),
    ] {
        let out = srv.record(&args, Some(stdin), Some(&pane));
        assert!(out.status.success(), "{args:?}");
        assert!(out.stdout.is_empty(), "{args:?}");
    }
    assert!(!spool.exists() || std::fs::read_to_string(&spool).unwrap().is_empty());

    // An unreachable tmux only loses the window index.
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_voice-paned"));
    cmd.args([
        "record",
        "--source",
        "codex",
        "--spool",
        sp,
        r#"{"type":"agent-turn-complete","last-assistant-message":"x"}"#,
    ])
    .env("TMUX_PANE", "%0")
    .env("TMUX", format!("{}/missing,1,0", srv.dir.path().display()))
    .env("TMUX_TMPDIR", srv.dir.path());
    let out = cmd.output().unwrap();
    assert!(out.status.success() && out.stdout.is_empty());
    let line = std::fs::read_to_string(&spool).unwrap();
    assert!(
        line.contains("\"window\":null") && line.contains("\"pane\":\"%0\""),
        "{line}"
    );
}

#[tokio::test]
async fn both_start_orders_work() {
    let srv = server!();

    // Launcher first: the view session already exists when voice-paned starts.
    srv.ok(&["new-session", "-d", "-t", "agentbox", "-s", "pre-made"]);
    let mut ctl = srv.controller();
    let focus = ctl.handshake("pre-made").await;
    assert_eq!(focus.window, 0);
    assert_eq!(ctl.request(C::Ping { id: 1 }, 1).await, S::Pong { id: 1 });
    drop(ctl);

    // voice-paned first: it creates the view session.
    let mut ctl = srv.controller();
    ctl.handshake("fresh").await;
    assert!(srv.sessions().contains(&"fresh".to_owned()));
    assert!(ctl.control_check(&srv, "fresh"));
    drop(ctl);

    // Both at once: the launcher creates the session at the moment Hello arrives.
    for round in 0..5 {
        let view = format!("race-{round}");
        let mut ctl = srv.controller();
        let s = srv
            .cmd()
            .args(["new-session", "-d", "-t", "agentbox", "-s", &view])
            .spawn()
            .unwrap();
        ctl.send(&C::Hello {
            proto: PROTOCOL_VERSION,
            client: "t".into(),
            view_session: view.clone(),
        })
        .await;
        let welcome = ctl.recv().await;
        assert!(
            matches!(welcome, S::Welcome { .. }),
            "round {round}: {welcome:?}"
        );
        let mut s = s;
        let _ = s.wait();
    }
    assert_eq!(srv.base_window(), "0");
}

impl common::Controller {
    /// Whether exactly this controller's control client is attached to `view`.
    fn control_check(&self, srv: &Server, view: &str) -> bool {
        srv.out(&[
            "list-clients",
            "-t",
            &format!("={view}"),
            "-F",
            "#{client_flags}",
        ])
        .contains("control-mode")
    }
}

#[tokio::test]
async fn clean_exit_leaves_the_shared_windows_alone() {
    let srv = server!();
    let windows_before = srv.out(&[
        "list-windows",
        "-t",
        "agentbox",
        "-F",
        "#{window_id} #{window_name}",
    ]);
    let panes_before = srv.out(&["list-panes", "-s", "-t", "=agentbox:", "-F", "#{pane_id}"]);

    // EOF.
    let mut ctl = srv.controller();
    ctl.handshake("eof-view").await;
    assert!(ctl.close_and_wait().await.success());

    // Bye.
    let mut ctl = srv.controller();
    ctl.handshake("bye-view").await;
    ctl.send(&C::Bye).await;
    let status = tokio::time::timeout(TIMEOUT, ctl.child.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(status.success());

    // Hello then immediately EOF is fine too.
    let mut ctl = srv.controller();
    ctl.send(&C::Hello {
        proto: PROTOCOL_VERSION,
        client: "t".into(),
        view_session: "quick".into(),
    })
    .await;
    assert!(ctl.close_and_wait().await.success());

    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(srv.sessions().contains(&"agentbox".to_owned()));
    assert_eq!(
        srv.out(&[
            "list-windows",
            "-t",
            "agentbox",
            "-F",
            "#{window_id} #{window_name}"
        ]),
        windows_before
    );
    assert_eq!(
        srv.out(&["list-panes", "-s", "-t", "=agentbox:", "-F", "#{pane_id}"]),
        panes_before
    );
    assert_eq!(srv.base_window(), "0");
    assert_eq!(srv.control_clients(), 0, "no control client may linger");
}

#[tokio::test]
async fn prompt_detection_emits_once_per_menu_and_rearms() {
    use std::time::Duration;
    let srv = server!();
    // A pane whose foreground process is named `claude` (argv[0]), with the
    // menu drawn by writing to the pane's tty.
    srv.ok(&[
        "new-window",
        "-d",
        "-t",
        "agentbox:",
        "-n",
        "agent",
        "bash",
        "-c",
        "exec -a claude bash -c 'read x; read x'",
    ]);
    let pane = srv.out(&["list-panes", "-t", "agentbox:agent", "-F", "#{pane_id}"]);
    let tty = srv.out(&["display-message", "-p", "-t", &pane, "#{pane_tty}"]);
    let start = std::time::Instant::now();
    while srv.out(&[
        "display-message",
        "-p",
        "-t",
        &pane,
        "#{pane_current_command}",
    ]) != "claude"
    {
        assert!(
            start.elapsed() < TIMEOUT,
            "pane command never became `claude`"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    let draw = |text: &str| std::fs::write(&tty, text).unwrap();
    let menu = "Bash command\r\n\r\n Do you want to proceed?\r\n \u{276f} 1. Yes\r\n   2. No, and tell Claude what to do differently (esc)\r\n";

    let mut ctl = srv.controller();
    ctl.handshake("view-pd").await;
    draw(menu);
    let e = ctl.recv_until(next_event).await;
    assert_eq!(
        (e.kind, e.agent, e.pane.as_str()),
        (EventKind::Prompt, AgentKind::Claude, pane.as_str())
    );
    assert_eq!(
        e.text.as_deref(),
        Some(
            "Do you want to proceed?\n1. Yes\n2. No, and tell Claude what to do differently (esc)"
        )
    );

    // Same menu, still showing: not announced again.
    assert!(
        tokio::time::timeout(Duration::from_millis(4000), ctl.try_recv())
            .await
            .is_err()
    );

    // Menu gone, then back: announced again.
    draw("\u{1b}[2J\u{1b}[H");
    assert!(
        tokio::time::timeout(Duration::from_millis(3500), ctl.try_recv())
            .await
            .is_err()
    );
    draw(menu);
    let e = ctl.recv_until(next_event).await;
    assert_eq!(e.kind, EventKind::Prompt);
}

#[tokio::test]
async fn controller_never_arms_destroy_unattached_and_launcher_can() {
    let srv = server!();
    let mut ctl = srv.controller();
    ctl.handshake("lv").await;
    assert_eq!(
        srv.out(&["show-options", "-Av", "-t", "=lv:", "destroy-unattached"]),
        "off"
    );

    // The launcher attaches its terminal and arms the option once attached.
    let Some(pty) = common::PtyClient::attach(&srv, "=lv", 80, 24) else {
        return;
    };
    srv.ok(&["set-option", "-t", "=lv:", "destroy-unattached", "on"]);

    // Terminal detaches while the controller's control client remains: the
    // view survives and the controller keeps working.
    let name = srv.terminal_clients().remove(0);
    srv.ok(&["detach-client", "-t", &name]);
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert!(srv.sessions().contains(&"lv".to_owned()));
    assert_eq!(ctl.request(C::Ping { id: 1 }, 1).await, S::Pong { id: 1 });

    // The controller exits last: tmux (not the controller) destroys the view.
    assert!(ctl.close_and_wait().await.success());
    let start = std::time::Instant::now();
    while srv.sessions().contains(&"lv".to_owned()) {
        assert!(
            start.elapsed() < TIMEOUT,
            "armed view should go with its last client"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!(srv.base_window(), "0");
    drop(pty);

    // Unarmed: a controller-created view simply persists after exit.
    let mut ctl = srv.controller();
    ctl.handshake("persist").await;
    assert!(ctl.close_and_wait().await.success());
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert!(srv.sessions().contains(&"persist".to_owned()));
}

#[tokio::test]
async fn recorder_rotates_an_oversized_spool_itself() {
    let srv = server!();
    let spool = srv.spool_path();
    std::fs::create_dir_all(spool.parent().unwrap()).unwrap();
    let filler = "{\"v\":1,\"ts_ms\":1,\"pane\":\"%0\",\"window\":0,\"agent\":\"codex\",\"kind\":\"notice\",\"text\":\"filler\"}\n";
    std::fs::write(
        &spool,
        filler.repeat(voicelink_proto::SPOOL_MAX_BYTES as usize / filler.len() + 2),
    )
    .unwrap();
    let mut ctl = srv.controller();
    ctl.handshake("rot").await;
    let pane = srv.panes_of(0)[0].clone();
    let payload = r#"{"type":"agent-turn-complete","last-assistant-message":"post rotation"}"#;
    let out = srv.record(
        &[
            "--source",
            "codex",
            "--spool",
            spool.to_str().unwrap(),
            payload,
        ],
        None,
        Some(&pane),
    );
    assert!(out.status.success() && out.stdout.is_empty());
    assert!(
        spool.with_file_name("events.jsonl.1").exists(),
        "the recorder must rotate"
    );
    assert!(std::fs::metadata(&spool).unwrap().len() < 1000);
    let e = ctl.recv_until(next_event).await;
    assert_eq!(e.text.as_deref(), Some("post rotation"));
}

const CLAUDE_MENU: &str = "Bash command\r\n\r\n Do you want to proceed?\r\n \u{276f} 1. Yes\r\n   2. No, and tell Claude what to do differently (esc)\r\n";
const CODEX_MENU: &str = "  Would you like to run the following command?\r\n\r\n  $ cargo test --workspace\r\n\r\n\u{203a} 1. Yes, proceed (y)\r\n  2. No, and tell Codex what to do differently (esc)\r\n\r\n  Press enter to confirm or esc to cancel\r\n";

async fn prompt_active_case(command: &str, menu: &str) {
    use std::time::Duration;
    let srv = server!();
    let agent = srv.fake_agent(command, "agent");
    agent.draw(menu);
    srv.wait_capture(&agent.pane, "the menu", |c| c.contains("1."));
    let mut ctl = srv.controller();
    ctl.handshake("view-pa").await;
    assert_eq!(
        ctl.request(
            C::SelectPane {
                id: 1,
                pane: agent.pane.clone()
            },
            1
        )
        .await,
        S::Ack { id: 1 }
    );
    let before = srv.capture(&agent.pane);

    // Every flavour of Send is refused, `force` included: a leading `y` or a
    // digit would answer the menu, and a bare Enter accepts the highlighted option.
    let attempts = [
        ("yes, run the tests", true, false),
        ("yes, run the tests", true, true),
        ("2", false, true),
        ("", true, true),
        ("", true, false),
    ];
    for (n, (text, submit, force)) in attempts.into_iter().enumerate() {
        let id = 10 + n as u64;
        let r = ctl
            .request(
                C::Send {
                    id,
                    text: text.into(),
                    submit,
                    force,
                },
                id,
            )
            .await;
        assert_eq!(
            err_code(&r),
            ErrorCode::PromptActive,
            "{text:?} submit={submit} force={force}"
        );
    }
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(
        srv.capture(&agent.pane),
        before,
        "a refused Send must leave the pane untouched (tty echo is on)"
    );

    // Keys still answer the menu.
    assert_eq!(
        ctl.request(
            C::Keys {
                id: 30,
                keys: vec![Key::Digit(2)]
            },
            30
        )
        .await,
        S::Ack { id: 30 }
    );
    srv.wait_capture(&agent.pane, "the digit echoed", |c| c != before);

    // Once the menu is gone a forced Send types again.
    agent.draw("\u{1b}[2J\u{1b}[H");
    srv.wait_capture(&agent.pane, "a cleared screen", |c| !c.contains("1."));
    assert_eq!(
        ctl.request(
            C::Send {
                id: 31,
                text: "hello".into(),
                submit: false,
                force: true
            },
            31
        )
        .await,
        S::Ack { id: 31 }
    );
    srv.wait_capture(&agent.pane, "hello", |c| c.contains("hello"));
}

#[tokio::test]
async fn forced_send_never_types_into_a_claude_menu() {
    prompt_active_case("claude", CLAUDE_MENU).await;
}

#[tokio::test]
async fn forced_send_never_types_into_a_codex_menu() {
    prompt_active_case("codex", CODEX_MENU).await;
}

#[tokio::test]
async fn unknown_message_types_and_keys_are_refused_without_ending_the_session() {
    let srv = server!();
    let mut ctl = srv.controller();
    ctl.handshake("view-unk").await;
    // A message from the future, with an id and extra fields.
    ctl.send_raw("{\"type\":\"teleport\",\"id\":77,\"to\":\"%1\"}\n")
        .await;
    match ctl.recv().await {
        S::Error { id, code, message } => {
            assert_eq!((id, code), (Some(77), ErrorCode::BadRequest));
            assert!(message.contains("teleport"), "{message}");
        }
        other => panic!("{other:?}"),
    }
    // A known request with an extra field works.
    ctl.send_raw("{\"type\":\"ping\",\"id\":78,\"future\":true}\n")
        .await;
    assert_eq!(ctl.recv().await, S::Pong { id: 78 });
    // An unknown key (with a payload) is a BadRequest, nothing is pressed.
    ctl.send_raw("{\"type\":\"keys\",\"id\":79,\"keys\":[{\"key\":\"f5\",\"value\":1}]}\n")
        .await;
    match ctl.recv().await {
        S::Error { id, code, .. } => assert_eq!((id, code), (Some(79), ErrorCode::BadRequest)),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        ctl.request(C::Ping { id: 80 }, 80).await,
        S::Pong { id: 80 }
    );
}
