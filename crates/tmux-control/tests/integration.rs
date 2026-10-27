//! Integration tests against a real, private tmux server.
//!
//! Every test starts its own server on a socket inside a temporary directory
//! (`tmux -S <dir>/sock`) and kills it on drop; nothing here ever addresses the
//! default server. Tests skip with a message when `tmux` is not on `PATH`.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use tmux_control::{
    Client, ClientOptions, Error, Notification, Notifications, Socket, command_line,
    ensure_grouped_session, has_session, quote, run,
};

struct Server {
    dir: tempfile::TempDir,
    sock: PathBuf,
}

impl Server {
    fn start(session: &str) -> Option<Server> {
        if Command::new("tmux")
            .arg("-V")
            .stdout(Stdio::null())
            .status()
            .is_err()
        {
            eprintln!("SKIP: tmux not found on PATH");
            return None;
        }
        let dir = tempfile::Builder::new()
            .prefix("tc-")
            .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
            .unwrap();
        let sock = dir.path().join("sock");
        let server = Server { dir, sock };
        let st = server
            .cmd()
            .args([
                "-f",
                "/dev/null",
                "new-session",
                "-d",
                "-x",
                "100",
                "-y",
                "30",
                "-s",
                session,
                "cat",
            ])
            .status()
            .unwrap();
        assert!(st.success());
        Some(server)
    }

    fn cmd(&self) -> Command {
        let mut c = Command::new("tmux");
        c.arg("-S")
            .arg(&self.sock)
            .env_remove("TMUX")
            .env("TMUX_TMPDIR", self.dir.path());
        c
    }

    fn out(&self, args: &[&str]) -> String {
        let o = self.cmd().args(args).output().unwrap();
        String::from_utf8_lossy(&o.stdout).trim_end().to_owned()
    }

    fn socket(&self) -> Socket {
        Socket::Path(self.sock.clone())
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.cmd().arg("kill-server").stderr(Stdio::null()).status();
    }
}

async fn next(rx: &mut Notifications) -> Notification {
    tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("timed out waiting for a notification")
        .expect("channel closed")
}

#[tokio::test]
async fn commands_return_output_and_errors() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    let (client, _rx) =
        Client::attach(&srv.socket(), &ClientOptions::new("base").flag("no-output")).unwrap();
    let lines = client.command("list-panes -F '#{pane_id}'").await.unwrap();
    assert_eq!(lines.len(), 1);
    assert!(lines[0].starts_with('%'));

    let err = client.command("no-such-command").await.unwrap_err();
    assert!(
        matches!(err, Error::Tmux(ref m) if m.contains("unknown command")),
        "{err:?}"
    );

    // The connection survives an error.
    assert_eq!(
        client.command("display-message -p ok").await.unwrap(),
        ["ok"]
    );

    // Replies stay matched to commands when several are in flight at once.
    let (a, b, c, d) = tokio::join!(
        client.command("display-message -p n0"),
        client.command("no-such-command"),
        client.command("display-message -p n2"),
        client.command("display-message -p n3"),
    );
    assert_eq!(a.unwrap(), ["n0"]);
    assert!(matches!(b, Err(Error::Tmux(_))));
    assert_eq!(c.unwrap(), ["n2"]);
    assert_eq!(d.unwrap(), ["n3"]);
    client.detach().await;
}

#[tokio::test]
async fn invalid_commands_are_refused_locally() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    let (client, _rx) = Client::attach(&srv.socket(), &ClientOptions::new("base")).unwrap();
    for bad in [
        "",
        "   ",
        "display-message -p a\ndisplay-message -p b",
        "x\r",
    ] {
        assert!(
            matches!(client.command(bad).await, Err(Error::InvalidCommand(_))),
            "{bad:?}"
        );
    }
    // Still alive afterwards: an empty line would have detached us.
    assert_eq!(
        client.command("display-message -p alive").await.unwrap(),
        ["alive"]
    );
}

#[tokio::test]
async fn notifications_are_typed() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    let (client, mut rx) =
        Client::attach(&srv.socket(), &ClientOptions::new("base").flag("no-output")).unwrap();
    client.command("display-message -p sync").await.unwrap();

    srv.out(&["new-window", "-d", "-t", "base", "cat"]);
    loop {
        if let Notification::WindowAdd { window } = next(&mut rx).await {
            assert!(window.starts_with('@'));
            break;
        }
    }
    srv.out(&["rename-window", "-t", "base:1", "renamed one"]);
    loop {
        // Automatic renames from the new window's command may arrive first.
        if let Notification::WindowRenamed { name, .. } = next(&mut rx).await {
            if name == "renamed one" {
                break;
            }
        }
    }
    client.command("select-window -t base:1").await.unwrap();
    loop {
        if let Notification::SessionWindowChanged { window, .. } = next(&mut rx).await {
            assert!(window.starts_with('@'));
            break;
        }
    }
}

#[tokio::test]
async fn output_notifications_are_decoded() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    let (client, mut rx) = Client::attach(&srv.socket(), &ClientOptions::new("base")).unwrap();
    client.command("display-message -p sync").await.unwrap();
    let pane = client
        .command("display-message -p '#{pane_id}'")
        .await
        .unwrap()
        .remove(0);
    client
        .command(format!("send-keys -t {pane} -l marker-xyz"))
        .await
        .unwrap();
    let mut seen = Vec::new();
    while !String::from_utf8_lossy(&seen).contains("marker-xyz") {
        if let Notification::Output { pane: p, data } = next(&mut rx).await {
            assert_eq!(p, pane);
            seen.extend(data);
        }
    }
}

#[tokio::test]
async fn quoting_survives_a_real_round_trip() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    srv.out(&["new-session", "-d", "-s", "sentinel", "cat"]);
    let (client, _rx) =
        Client::attach(&srv.socket(), &ClientOptions::new("base").flag("no-output")).unwrap();

    let payloads = [
        "plain",
        "x; kill-session -t sentinel",
        "x ;kill-session -t sentinel; y",
        "it's \"quoted\" and 'single'",
        "#{session_name} #(echo pwned) ##",
        "$HOME ${HOME} ~ $(id)",
        "back\\slash \\n \\012",
        "multi\nline\nvalue",
        "cr\rhere\ttab\u{1b}[31mesc",
        "unicode \u{e9}\u{65e5}\u{672c}\u{1f600}",
        "-t looks like a flag",
        "}{ %if 1",
        "trailing backslash\\",
    ];
    for (i, payload) in payloads.iter().enumerate() {
        let name = format!("buf{i}");
        client
            .command(command_line([
                "set-buffer",
                "-b",
                name.as_str(),
                "--",
                payload,
            ]))
            .await
            .unwrap();
        let out = srv
            .cmd()
            .args(["show-buffer", "-b", &name])
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            *payload,
            "payload {i}: {payload:?}"
        );
    }
    let sessions = srv.out(&["list-sessions", "-F", "#{session_name}"]);
    assert!(
        sessions.lines().any(|l| l == "sentinel"),
        "injection killed the sentinel: {sessions}"
    );
    assert_eq!(srv.out(&["list-buffers"]).lines().count(), payloads.len());
    assert_eq!(quote("a"), "a");
}

#[tokio::test]
async fn attach_to_missing_session_reports_why() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    let (client, mut rx) = Client::attach(&srv.socket(), &ClientOptions::new("nope")).unwrap();
    let err = client.command("display-message -p x").await.unwrap_err();
    assert!(
        matches!(&err, Error::Closed(Some(r)) if r.contains("can't find session")),
        "{err:?}"
    );
    // The channel ends (possibly after an Exit notification).
    loop {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
        {
            None => break,
            Some(Notification::Exit { .. }) => {}
            Some(other) => panic!("unexpected {other:?}"),
        }
    }
}

#[tokio::test]
async fn attach_matches_session_names_exactly() {
    let Some(srv) = Server::start("remote-laptop") else {
        return;
    };
    let (client, _rx) = Client::attach(&srv.socket(), &ClientOptions::new("remote")).unwrap();
    assert!(
        client.command("display-message -p x").await.is_err(),
        "prefix must not match"
    );
}

#[tokio::test]
async fn detach_leaves_server_and_session_alone() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    let (client, _rx) = Client::attach(&srv.socket(), &ClientOptions::new("base")).unwrap();
    client.command("display-message -p x").await.unwrap();
    assert!(
        srv.out(&["list-clients", "-F", "#{client_flags}"])
            .contains("control-mode")
    );
    client.detach().await;
    assert_eq!(srv.out(&["list-clients"]), "");
    assert_eq!(srv.out(&["list-sessions", "-F", "#{session_name}"]), "base");
}

#[tokio::test]
async fn dropping_the_client_disconnects_it() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    let (client, rx) = Client::attach(&srv.socket(), &ClientOptions::new("base")).unwrap();
    client.command("display-message -p x").await.unwrap();
    drop(client);
    drop(rx);
    for _ in 0..50 {
        if srv.out(&["list-clients"]).is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("control client still attached after drop");
}

#[tokio::test]
async fn exit_notification_when_session_is_destroyed() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    srv.out(&["new-session", "-d", "-s", "other", "cat"]);
    let (client, mut rx) = Client::attach(&srv.socket(), &ClientOptions::new("base")).unwrap();
    client.command("display-message -p x").await.unwrap();
    srv.out(&["kill-session", "-t", "=base"]);
    let mut saw_exit = false;
    while let Ok(Some(n)) = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
        if matches!(n, Notification::Exit { .. }) {
            saw_exit = true;
        }
    }
    assert!(saw_exit, "no %exit seen");
    assert!(matches!(
        client.command("display-message -p x").await,
        Err(Error::Closed(_))
    ));
}

#[tokio::test]
async fn ensure_grouped_session_is_idempotent_and_race_tolerant() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    srv.out(&["new-window", "-d", "-t", "base", "cat"]);
    let sock = srv.socket();
    assert!(ensure_grouped_session(&sock, "base", "view").await.unwrap());
    assert!(!ensure_grouped_session(&sock, "base", "view").await.unwrap());
    assert!(has_session(&sock, "view").await.unwrap());
    assert!(
        !has_session(&sock, "vie").await.unwrap(),
        "exact match only"
    );
    // Same group: both list both windows.
    assert_eq!(srv.out(&["list-windows", "-t", "=view"]).lines().count(), 2);

    // Race: many concurrent callers, exactly one creates.
    let mut handles = Vec::new();
    for _ in 0..8 {
        let s = sock.clone();
        handles.push(tokio::spawn(async move {
            ensure_grouped_session(&s, "base", "racer").await
        }));
    }
    let mut created = 0;
    for h in handles {
        if h.await.unwrap().unwrap() {
            created += 1;
        }
    }
    assert_eq!(created, 1);

    assert!(matches!(
        ensure_grouped_session(&sock, "base", "bad.name").await,
        Err(Error::InvalidSessionName(_))
    ));
    assert!(matches!(
        ensure_grouped_session(&sock, "missing", "other-view").await,
        Err(Error::Tmux(_))
    ));
}

#[tokio::test]
async fn run_returns_stdout_or_stderr() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    let out = run(&srv.socket(), &["list-sessions", "-F", "#{session_name}"])
        .await
        .unwrap();
    assert_eq!(out.trim(), "base");
    let err = run(&srv.socket(), &["frobnicate"]).await.unwrap_err();
    assert!(
        matches!(err, Error::Tmux(ref m) if m.contains("unknown command")),
        "{err:?}"
    );
}

#[tokio::test]
async fn compound_commands_and_hooks_do_not_misalign_replies() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    srv.out(&["new-window", "-d", "-t", "base", "cat"]);
    let (client, _rx) =
        Client::attach(&srv.socket(), &ClientOptions::new("base").flag("no-output")).unwrap();
    client
        .command("set-hook -g after-select-window 'display-message -p hooked'")
        .await
        .unwrap();

    // Hook output must not be mistaken for the reply to the next command.
    client.command("select-window -t base:1").await.unwrap();
    assert_eq!(
        client.command("display-message -p sync").await.unwrap(),
        ["sync"]
    );
    client.command("select-window -t base:0").await.unwrap();
    assert_eq!(
        client.command("display-message -p sync2").await.unwrap(),
        ["sync2"]
    );

    // A `;` list is refused rather than allowed to shift later replies.
    let r = client
        .command("display-message -p a ; display-message -p b")
        .await;
    println!("compound: {r:?}");
    assert_eq!(client.command("display-message -p c").await.unwrap(), ["c"]);
    let r = client.command("if-shell -F 1 'display-message -p x'").await;
    println!("if-shell: {r:?}");
    assert_eq!(client.command("display-message -p d").await.unwrap(), ["d"]);
}

#[tokio::test]
async fn escape_format_survives_format_expansion() {
    let Some(srv) = Server::start("base") else {
        return;
    };
    let (client, _rx) = Client::attach(&srv.socket(), &ClientOptions::new("base")).unwrap();
    let raw = "#{session_name} #(echo hi) ##";
    let line = command_line(["display-message", "-p", &tmux_control::escape_format(raw)]);
    assert_eq!(client.command(line).await.unwrap(), [raw]);
}
