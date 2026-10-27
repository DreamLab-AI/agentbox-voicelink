//! Live tests against real estate services. Run with
//! `VOICELINK_LIVE=1 cargo test -p voicelink-backends --test live -- --ignored --nocapture`
//! (add `--features whisper` for the embedded whisper round trip).

use std::time::Instant;
use tokio::sync::mpsc;
#[cfg(feature = "whisper")]
use voicelink_backends::SpeechToText;
use voicelink_backends::{
    AudioClip, ChatMessage, ChatRequest, LlmConfig, Role, SttConfig, ToolSpec, TtsConfig,
    build_llm, build_stt, build_tts,
};

const SENTENCE: &str = "The quick brown fox jumps over the lazy dog near the river bank.";
const KEYWORDS: [&str; 4] = ["quick", "fox", "lazy", "river"];

fn live() -> bool {
    std::env::var("VOICELINK_LIVE").is_ok_and(|v| v == "1")
}

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

async fn synth() -> AudioClip {
    let tts = build_tts(&TtsConfig::OpenAi {
        base_url: env("VOICELINK_TTS_URL", "http://pocket-tts:8000/v1"),
        model: "pocket-tts".into(),
        voice: "alba".into(),
        api_key_env: None,
    })
    .unwrap();
    let (tx, mut rx) = mpsc::channel(256);
    let t = Instant::now();
    let task = tokio::spawn(async move { tts.synthesize(SENTENCE, tx).await });
    let mut first = None;
    let (mut rate, mut samples, mut chunks) = (0, Vec::new(), 0);
    while let Some(c) = rx.recv().await {
        first.get_or_insert_with(|| t.elapsed());
        rate = c.sample_rate;
        chunks += 1;
        samples.extend(c.samples);
    }
    task.await.unwrap().unwrap();
    println!(
        "TTS pocket: first chunk {:?}, total {:?}, {chunks} chunks, {} samples @ {rate} Hz ({:.2}s audio)",
        first.unwrap(),
        t.elapsed(),
        samples.len(),
        samples.len() as f32 / rate as f32
    );
    assert!(samples.len() > rate as usize, "implausibly short audio");
    AudioClip {
        sample_rate: rate,
        samples,
    }
}

fn assert_keywords(who: &str, text: &str) {
    let lower = text.to_lowercase();
    println!("{who} transcript: {text:?}");
    for k in KEYWORDS {
        assert!(lower.contains(k), "{who}: missing {k:?} in {text:?}");
    }
}

#[tokio::test]
#[ignore = "needs VOICELINK_LIVE=1 and estate services"]
async fn pocket_tts_to_nemotron() {
    if !live() {
        return;
    }
    let clip = synth().await;
    let stt = build_stt(&SttConfig::Nemotron {
        url: env(
            "VOICELINK_NEMOTRON_URL",
            "ws://speech-to-text:8000/api/asr-streaming",
        ),
    })
    .unwrap();
    let t = Instant::now();
    let tr = stt.transcribe(&clip).await.unwrap();
    println!("Nemotron latency {:?}", t.elapsed());
    assert_keywords("nemotron", &tr.text);
}

#[cfg(feature = "whisper")]
#[tokio::test]
#[ignore = "needs VOICELINK_LIVE=1, estate TTS and a ggml model"]
async fn pocket_tts_to_embedded_whisper() {
    if !live() {
        return;
    }
    let clip = synth().await;
    let stt: Box<dyn SpeechToText> = build_stt(&SttConfig::Whisper {
        model_path: env(
            "VOICELINK_WHISPER_MODEL",
            "/home/devuser/workspace/.tmp/vl-models/ggml-tiny.en.bin",
        ),
        language: Some("en".into()),
        threads: 4,
        gpu: false,
    })
    .unwrap();
    println!("backend: {}", stt.name());
    let t = Instant::now();
    let tr = stt.transcribe(&clip).await.unwrap();
    println!("whisper latency {:?}", t.elapsed());
    assert_keywords("whisper", &tr.text);
}

#[tokio::test]
#[ignore = "needs VOICELINK_LIVE=1 and the Ontology Loom"]
async fn loom_chat_and_tool_call() {
    if !live() {
        return;
    }
    let llm = build_llm(&LlmConfig::OpenAi {
        base_url: env("VOICELINK_LLM_URL", "http://192.168.2.132:8084/v1"),
        model: env("VOICELINK_LLM_MODEL", "loom"),
        api_key_env: None,
        extra: serde_json::json!({"loom_options":{"scaffold":false}})
            .as_object()
            .cloned()
            .unwrap(),
    })
    .unwrap();
    let user = |t: &str| ChatMessage {
        role: Role::User,
        content: Some(t.into()),
        tool_calls: vec![],
        tool_call_id: None,
    };
    let t = Instant::now();
    let plain = llm
        .chat(&ChatRequest {
            messages: vec![user("Reply with exactly one word: pong")],
            tools: vec![],
            temperature: 0.0,
            max_tokens: 2048,
        })
        .await
        .unwrap();
    println!("loom plain reply {:?} in {:?}", plain.content, t.elapsed());
    assert!(
        plain
            .content
            .as_deref()
            .is_some_and(|c| c.to_lowercase().contains("pong"))
    );

    let t = Instant::now();
    let tool = llm
        .chat(&ChatRequest {
            messages: vec![user("Use the type_text tool to type the text `ls -la` into the terminal.")],
            tools: vec![ToolSpec {
                name: "type_text".into(),
                description: "Type text into the focused terminal pane".into(),
                parameters: serde_json::json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}),
            }],
            temperature: 0.0,
            max_tokens: 2048,
        })
        .await
        .unwrap();
    println!("loom tool reply {tool:?} in {:?}", t.elapsed());
    assert_eq!(tool.tool_calls.len(), 1);
    assert_eq!(tool.tool_calls[0].name, "type_text");
    assert!(
        tool.tool_calls[0].arguments["text"]
            .as_str()
            .unwrap()
            .contains("ls")
    );
}
