//! Pins the tmux behaviours the controller's design depends on, observed on a
//! private server (developed against tmux 3.8):
//!
//! * destroying a grouped view session never closes the shared windows;
//! * `destroy-unattached` set on a session that has no client destroys it
//!   straight away, so the controller sets it only after attaching;
//! * while a control client is attached the session survives, and what
//!   happens when the terminal client detaches while the control client
//!   remains (the session survives; when the control client then leaves, it
//!   is destroyed);
//! * `ignore-size` keeps the operator's pane sizes unchanged.

mod common;

use std::time::{Duration, Instant};

use common::{PtyClient, Server};
use tmux_control::{Client, ClientOptions, Socket};

macro_rules! server {
    () => {
        match Server::start() {
            Some(s) => s,
            None => return,
        }
    };
}

fn windows(srv: &Server) -> String {
    srv.out(&[
        "list-windows",
        "-t",
        "=agentbox",
        "-F",
        "#{window_id} #{window_name} #{window_panes}",
    ])
}

fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let start = Instant::now();
    while !cond() {
        assert!(
            start.elapsed() < common::TIMEOUT,
            "timed out waiting for {what}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

async fn attach(srv: &Server, view: &str, flags: &[&str]) -> Client {
    let mut opts = ClientOptions::new(view);
    for f in flags {
        opts = opts.flag(*f);
    }
    let (client, _events) = Client::attach(&Socket::Path(srv.sock.clone()), &opts).unwrap();
    // Keep draining notifications so the channel never matters.
    tokio::spawn(async move {
        let mut e = _events;
        while e.recv().await.is_some() {}
    });
    client.command("display-message -p ok").await.unwrap();
    client
}

#[tokio::test]
async fn destroying_a_view_session_never_closes_shared_windows() {
    let srv = server!();
    let before = windows(&srv);
    srv.ok(&["new-session", "-d", "-t", "agentbox", "-s", "v1"]);
    assert_eq!(
        srv.out(&[
            "list-windows",
            "-t",
            "=v1",
            "-F",
            "#{window_id} #{window_name} #{window_panes}"
        ]),
        before
    );

    // Kill the view session outright.
    srv.ok(&["kill-session", "-t", "=v1"]);
    assert_eq!(windows(&srv), before);

    // Detach of a control client attached to a view session.
    srv.ok(&["new-session", "-d", "-t", "agentbox", "-s", "v2"]);
    let client = attach(&srv, "v2", &["ignore-size", "no-output"]).await;
    client.detach().await;
    srv.ok(&["kill-session", "-t", "=v2"]);
    assert_eq!(windows(&srv), before);
    assert_eq!(srv.base_window(), "0");
}

#[tokio::test]
async fn destroy_unattached_must_be_set_after_attaching() {
    let srv = server!();

    // Observed on 3.8: setting it on a session with no clients destroys the
    // session at once (tmux sweeps unattached sessions when the setting
    // command's client goes away). Hence "set it immediately after creating
    // it" from a one-shot command is wrong.
    srv.ok(&["new-session", "-d", "-t", "agentbox", "-s", "early"]);
    srv.ok(&["set-option", "-t", "=early:", "destroy-unattached", "on"]);
    wait_until("the unattached view to vanish", || {
        !srv.sessions().contains(&"early".to_owned())
    });
    assert!(srv.sessions().contains(&"agentbox".to_owned()));

    // The controller's order: create, attach the control client, then set it
    // over that connection.
    srv.ok(&["new-session", "-d", "-t", "agentbox", "-s", "late"]);
    let client = attach(&srv, "late", &["ignore-size", "no-output"]).await;
    client
        .command("set-option -t =late: destroy-unattached on")
        .await
        .unwrap();

    // Trigger the sweep deliberately: commands that start and finish clients
    // and sessions.
    for i in 0..3 {
        srv.ok(&["list-sessions"]);
        srv.ok(&["new-session", "-d", "-s", &format!("tmp{i}"), "cat"]);
        srv.ok(&["kill-session", "-t", &format!("=tmp{i}")]);
    }
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        srv.sessions().contains(&"late".to_owned()),
        "attached control client must keep the view alive"
    );
    assert_eq!(srv.control_clients(), 1);

    // When the last client leaves, the session is destroyed - and the shared
    // windows are not.
    let before = windows(&srv);
    client.detach().await;
    wait_until(
        "the view to be destroyed after the last client left",
        || !srv.sessions().contains(&"late".to_owned()),
    );
    assert_eq!(windows(&srv), before);
}

#[tokio::test]
async fn terminal_detaching_while_the_control_client_remains() {
    let srv = server!();
    srv.ok(&["new-session", "-d", "-t", "agentbox", "-s", "shared"]);
    let Some(pty) = PtyClient::attach(&srv, "=shared", 80, 24) else {
        return;
    };
    let client = attach(&srv, "shared", &["ignore-size", "no-output"]).await;
    client
        .command("set-option -t =shared: destroy-unattached on")
        .await
        .unwrap();
    assert_eq!(srv.terminal_clients().len(), 1);

    // The terminal client detaches (named explicitly; `-s shared` would also
    // detach the control client).
    let name = srv.terminal_clients().remove(0);
    srv.ok(&["detach-client", "-t", &name]);
    wait_until("the terminal client to go", || {
        srv.terminal_clients().is_empty()
    });
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        srv.sessions().contains(&"shared".to_owned()),
        "the control client keeps the session alive"
    );
    assert_eq!(srv.control_clients(), 1);

    // The control client still works and still follows the view.
    client.command("select-window -t =shared:1").await.unwrap();
    assert_eq!(srv.view_window("shared"), "1");
    assert_eq!(srv.base_window(), "0");

    // Control client leaves with no terminal attached: destroyed.
    client.detach().await;
    wait_until("the view to be destroyed", || {
        !srv.sessions().contains(&"shared".to_owned())
    });
    drop(pty);
}

#[tokio::test]
async fn control_client_leaving_while_the_terminal_remains() {
    let srv = server!();
    srv.ok(&["new-session", "-d", "-t", "agentbox", "-s", "shared"]);
    let Some(pty) = PtyClient::attach(&srv, "=shared", 80, 24) else {
        return;
    };
    let client = attach(&srv, "shared", &["ignore-size", "no-output"]).await;
    client
        .command("set-option -t =shared: destroy-unattached on")
        .await
        .unwrap();
    client.detach().await;
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        srv.sessions().contains(&"shared".to_owned()),
        "the operator's terminal keeps the view alive"
    );
    drop(pty);
    wait_until("the view to go with its last client", || {
        !srv.sessions().contains(&"shared".to_owned())
    });
}

fn size_of(srv: &Server, target: &str) -> String {
    srv.out(&[
        "display-message",
        "-p",
        "-t",
        target,
        "#{window_width}x#{window_height}",
    ])
}

#[tokio::test]
async fn ignore_size_keeps_operator_pane_sizes() {
    let srv = server!();
    srv.ok(&["new-session", "-d", "-t", "agentbox", "-s", "sz"]);
    // A terminal of a different size than the windows were created with
    // (100x30); its status line takes one row.
    let Some(pty) = PtyClient::attach(&srv, "=sz", 120, 40) else {
        return;
    };
    wait_until("the displayed window to adopt the terminal size", || {
        size_of(&srv, "=sz:0") == "120x39"
    });
    let pane0 = srv.panes_of(0)[0].clone();
    let pane_size = |p: &str| {
        srv.out(&[
            "display-message",
            "-p",
            "-t",
            p,
            "#{pane_width}x#{pane_height}",
        ])
    };
    assert_eq!(pane_size(&pane0), "120x39");

    // Negative control. A plain control client that merely attaches and runs
    // commands does not disturb a "latest"-sized window, but one that declares
    // a size (`refresh-client -C`, as terminal-less front ends do) resizes the
    // operator's window. This is the case `ignore-size` exists for.
    let plain = attach(&srv, "sz", &["no-output"]).await;
    plain.command("refresh-client -C 60x20").await.unwrap();
    wait_until(
        "the unprotected size declaration to resize the window",
        || size_of(&srv, "=sz:0") != "120x39",
    );
    eprintln!(
        "without ignore-size, refresh-client -C resized the window to {}",
        size_of(&srv, "=sz:0")
    );
    plain.detach().await;
    wait_until("the window to recover", || {
        size_of(&srv, "=sz:0") == "120x39"
    });

    // With ignore-size even an explicit size declaration is ignored.
    let ignoring = attach(&srv, "sz", &["ignore-size", "no-output"]).await;
    ignoring.command("refresh-client -C 60x20").await.unwrap();
    std::thread::sleep(Duration::from_millis(400));
    eprintln!(
        "with ignore-size, refresh-client -C leaves the window at {}",
        size_of(&srv, "=sz:0")
    );
    assert_eq!(size_of(&srv, "=sz:0"), "120x39");
    ignoring.detach().await;

    // With ignore-size: select-window / select-pane / send-keys leave every
    // size alone. (The terminal follows the view's window, as intended, and
    // sizes the newly displayed window to itself.)
    let client = attach(&srv, "sz", &["ignore-size", "no-output"]).await;
    client
        .command("set-option -t =sz: destroy-unattached on")
        .await
        .unwrap();
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(size_of(&srv, "=sz:0"), "120x39");
    assert_eq!(pane_size(&pane0), "120x39");
    let pane = srv.panes_of(1)[1].clone();
    client.command("select-window -t =sz:1").await.unwrap();
    client
        .command(format!("select-pane -t {pane}"))
        .await
        .unwrap();
    client
        .command(format!("send-keys -t {pane} -l typed"))
        .await
        .unwrap();
    wait_until("window 1 to be shown at terminal size", || {
        size_of(&srv, "=sz:1") == "120x39"
    });
    client.command("select-window -t =sz:0").await.unwrap();
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(size_of(&srv, "=sz:0"), "120x39");
    assert_eq!(pane_size(&pane0), "120x39");
    assert_eq!(srv.control_clients(), 1);
    client.detach().await;
    drop(pty);
}
