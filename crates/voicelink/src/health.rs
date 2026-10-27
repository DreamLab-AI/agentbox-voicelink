//! Reachability check of every configured backend, run by `voicelink connect`
//! before the terminal is handed to tmux.
//!
//! The probes are `voicelink_backends::probe_*`; this module only decides what
//! to probe and builds the message that tells the operator how to bring each
//! unreachable endpoint up. Hints are chosen from the configured URL and
//! forward names, never from the error text: through an ssh forward a dead
//! remote service looks like a closed connection rather than a refused one.

use std::path::Path;

use voicelink_backends::{LlmConfig, SttConfig, TtsConfig, probe_llm, probe_stt, probe_tts};

use crate::config::{Config, Slot};
use crate::stack::ComposeCommand;

/// One endpoint that did not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    /// Which slot it serves.
    pub slot: Slot,
    /// The endpoint (`/models` base URL or WebSocket URL).
    pub url: String,
    /// The probe's error, which names the exact URL it tried.
    pub error: String,
}

/// The configured endpoint of each slot.
pub fn endpoints(cfg: &Config) -> [(Slot, String); 3] {
    let stt = match &cfg.stt {
        SttConfig::OpenAi { base_url, .. } => base_url.clone(),
        SttConfig::Nemotron { url } => url.clone(),
    };
    let LlmConfig::OpenAi { base_url: llm, .. } = &cfg.llm;
    let TtsConfig::OpenAi { base_url: tts, .. } = &cfg.tts;
    [
        (Slot::Stt, stt),
        (Slot::Llm, llm.clone()),
        (Slot::Tts, tts.clone()),
    ]
}

/// Probes the three backends concurrently and returns those that failed.
/// Each probe is bounded by `voicelink_backends::PROBE_TIMEOUT`.
pub async fn check(cfg: &Config) -> Vec<Failure> {
    let (stt, llm, tts) = tokio::join!(
        probe_stt(&cfg.stt),
        probe_llm(&cfg.llm),
        probe_tts(&cfg.tts)
    );
    let urls = endpoints(cfg);
    [stt, llm, tts]
        .into_iter()
        .zip(urls)
        .filter_map(|(r, (slot, url))| {
            r.err().map(|e| Failure {
                slot,
                url,
                error: e.to_string(),
            })
        })
        .collect()
}

/// Whether `url` points at loopback port `port` (a forward's local end).
fn uses_port(url: &str, port: u16) -> bool {
    let needle = format!("127.0.0.1:{port}");
    url.match_indices(&needle)
        .any(|(i, _)| !url[i + needle.len()..].starts_with(|c: char| c.is_ascii_digit()))
}

/// How to bring the endpoint of `slot` at `url` up.
///
/// `compose_file` is where `compose.yml` was found, if anywhere.
pub fn hint(cfg: &Config, slot: Slot, url: &str, compose_file: Option<&Path>) -> String {
    use crate::setup::choices::{COMPOSE_LLM, COMPOSE_STT, COMPOSE_TTS};
    if let Some(f) = cfg
        .connection
        .forwards
        .iter()
        .find(|f| uses_port(url, f.local_port))
    {
        return match f.name.as_str() {
            "nemotron" | "pocket" => format!(
                "the agentbox speech plane is down ({} on the agentbox host, {}): check `supervisorctl status` there, or pick another source with `voicelink setup`",
                if f.name == "nemotron" {
                    "Nemotron speech-to-text"
                } else {
                    "Pocket TTS"
                },
                f.remote
            ),
            "loom" => format!(
                "the agentbox Ontology Loom is down (host port {}): start it on the agentbox host, or pick another source with `voicelink setup`",
                f.remote
            ),
            other => format!(
                "the service behind the ssh forward {other:?} ({}) is down",
                f.remote
            ),
        };
    }
    let compose_url = match slot {
        Slot::Stt => COMPOSE_STT,
        Slot::Llm => COMPOSE_LLM,
        Slot::Tts => COMPOSE_TTS,
    };
    if url == compose_url {
        return match (&cfg.stack, compose_file) {
            (Some(_), Some(f)) => match ComposeCommand::from_config(cfg, f) {
                Some(cmd) => format!("the compose stack is not running: {}", cmd.display()),
                None => "the compose stack is not running".to_string(),
            },
            (Some(stack), None) => format!(
                "the compose stack is not running: docker compose -f <path>/compose.yml --profile {} up -d (compose/compose.yml was not found next to the binary; set VOICELINK_COMPOSE)",
                stack.profile.name()
            ),
            (None, _) => "the compose stack is not running: re-run `voicelink setup`".to_string(),
        };
    }
    let port = url
        .rsplit_once(':')
        .map(|(_, rest)| rest.split('/').next().unwrap_or(""))
        .unwrap_or("");
    match (slot, port) {
        (_, "11434") => "start Ollama (`ollama serve`, or open the Ollama app)".into(),
        (_, "1234") => "start LM Studio's local server (Developer tab, Start Server)".into(),
        (Slot::Llm, "8080") => "start llama-server with --jinja (tool calls need it)".into(),
        (Slot::Stt, "8080") => {
            "start whisper-server with --inference-path /v1/audio/transcriptions".into()
        }
        (Slot::Tts, "8880") => "start Kokoro-FastAPI (port 8880)".into(),
        _ => format!("start the server that should answer at {url}"),
    }
}

/// The message printed when `failures` is not empty.
pub fn report(cfg: &Config, failures: &[Failure], compose_file: Option<&Path>) -> String {
    let mut out = format!(
        "voicelink: not connecting, {} of 3 model endpoints did not answer:\n",
        failures.len()
    );
    for f in failures {
        out.push_str(&format!(
            "\n  {} ({})\n    {}\n    -> {}\n",
            f.slot.label(),
            f.url,
            f.error,
            hint(cfg, f.slot, &f.url, compose_file)
        ));
    }
    out.push_str(
        "\nStart what is missing and run `voicelink connect` again, change the source with `voicelink setup`, or attach anyway with `voicelink connect --skip-health`.",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::choices::{Choices, Source, Tier};

    fn cfg(stt: Source, llm: Source, tts: Source) -> Config {
        let mut c = Choices::defaults_for(Tier::Nvidia, "laptop");
        c.stt_source = stt;
        c.llm_source = llm;
        c.tts_source = tts;
        c.to_config()
    }

    fn fail(cfg: &Config, slot: Slot) -> Failure {
        let (s, url) = endpoints(cfg)
            .into_iter()
            .find(|(s, _)| *s == slot)
            .unwrap();
        Failure {
            slot: s,
            error: format!("{url}/models: connection refused or host unreachable"),
            url,
        }
    }

    #[test]
    fn compose_endpoints_get_the_exact_docker_command() {
        let c = cfg(Source::Compose, Source::Compose, Source::Compose);
        let f = fail(&c, Slot::Stt);
        let msg = report(&c, &[f], Some(Path::new("/opt/vl/compose/compose.yml")));
        assert!(msg.contains("1 of 3"), "{msg}");
        assert!(
            msg.contains("speech-to-text (http://127.0.0.1:8181/v1)"),
            "{msg}"
        );
        assert!(
            msg.contains("docker compose -f /opt/vl/compose/compose.yml --profile cuda up -d llm-cuda stt-cuda tts-cuda"),
            "{msg}"
        );
        assert!(msg.contains("--skip-health"), "{msg}");
        // Without the file the hint still names the profile and the override.
        let h = hint(&c, Slot::Llm, "http://127.0.0.1:8180/v1", None);
        assert!(
            h.contains("--profile cuda") && h.contains("VOICELINK_COMPOSE"),
            "{h}"
        );
    }

    #[test]
    fn remote_endpoints_blame_the_agentbox_service_by_forward_name() {
        let c = cfg(Source::Remote, Source::Remote, Source::Remote);
        let stt = hint(
            &c,
            Slot::Stt,
            "ws://127.0.0.1:18897/api/asr-streaming",
            None,
        );
        assert!(
            stt.contains("agentbox speech plane is down") && stt.contains("Nemotron"),
            "{stt}"
        );
        let tts = hint(&c, Slot::Tts, "http://127.0.0.1:18898/v1", None);
        assert!(
            tts.contains("speech plane") && tts.contains("Pocket"),
            "{tts}"
        );
        let llm = hint(&c, Slot::Llm, "http://127.0.0.1:18084/v1", None);
        assert!(
            llm.contains("Ontology Loom") && llm.contains("127.0.0.1:8084"),
            "{llm}"
        );
    }

    #[test]
    fn local_servers_get_product_specific_advice() {
        let c = cfg(Source::Local, Source::Local, Source::Local);
        assert!(hint(&c, Slot::Llm, "http://127.0.0.1:11434/v1", None).contains("Ollama"));
        assert!(hint(&c, Slot::Llm, "http://localhost:1234/v1", None).contains("LM Studio"));
        assert!(hint(&c, Slot::Llm, "http://127.0.0.1:8080/v1", None).contains("--jinja"));
        assert!(hint(&c, Slot::Stt, "http://127.0.0.1:8080/v1", None).contains("--inference-path"));
        assert!(hint(&c, Slot::Tts, "http://127.0.0.1:8880/v1", None).contains("Kokoro-FastAPI"));
        assert_eq!(
            hint(&c, Slot::Llm, "https://llm.example.net/v1", None),
            "start the server that should answer at https://llm.example.net/v1"
        );
    }

    #[test]
    fn the_report_lists_every_failure_with_its_own_error() {
        let c = cfg(Source::Remote, Source::Local, Source::Compose);
        let fs = [
            fail(&c, Slot::Stt),
            fail(&c, Slot::Llm),
            fail(&c, Slot::Tts),
        ];
        let msg = report(&c, &fs, None);
        assert!(msg.contains("3 of 3"));
        for f in &fs {
            assert!(msg.contains(&f.error), "{msg}");
        }
    }

    #[tokio::test]
    async fn check_reports_dead_endpoints_and_names_them() {
        let mut c = cfg(Source::Local, Source::Local, Source::Local);
        c.stt = SttConfig::OpenAi {
            base_url: "http://127.0.0.1:1/v1".into(),
            model: "m".into(),
            api_key_env: None,
        };
        c.llm = LlmConfig::OpenAi {
            base_url: "http://127.0.0.1:1/v1".into(),
            model: "m".into(),
            api_key_env: None,
            extra: Default::default(),
        };
        c.tts = TtsConfig::OpenAi {
            base_url: "http://127.0.0.1:1/v1".into(),
            model: "m".into(),
            voice: "v".into(),
            api_key_env: None,
        };
        let f = check(&c).await;
        assert_eq!(f.len(), 3);
        assert!(
            f.iter()
                .all(|f| f.error.contains("http://127.0.0.1:1/v1/models"))
        );
    }
}
