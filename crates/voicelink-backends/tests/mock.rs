//! Backends against in-process mock servers.

use axum::{
    Json, Router,
    body::Body,
    extract::{
        Multipart, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use voicelink_backends::{
    AudioClip, BackendError, ChatMessage, ChatRequest, LlmConfig, Role, SttConfig, ToolSpec,
    TtsConfig, build_llm, build_stt, build_tts, decode_wav, encode_wav,
};

async fn serve(app: Router) -> String {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    format!("127.0.0.1:{}", addr.port())
}

fn req(messages: Vec<ChatMessage>) -> ChatRequest {
    ChatRequest {
        messages,
        tools: vec![ToolSpec {
            name: "send_keys".into(),
            description: "type".into(),
            parameters: json!({"type":"object","properties":{"text":{"type":"string"}}}),
        }],
        temperature: 0.2,
        max_tokens: 64,
    }
}

fn user(t: &str) -> ChatMessage {
    ChatMessage {
        role: Role::User,
        content: Some(t.into()),
        tool_calls: vec![],
        tool_call_id: None,
    }
}

#[derive(Clone, Default)]
struct Seen {
    body: Arc<Mutex<Option<Value>>>,
    auth: Arc<Mutex<Option<String>>>,
    reply: Arc<Mutex<Value>>,
}

async fn chat_handler(State(s): State<Seen>, h: HeaderMap, Json(b): Json<Value>) -> Response {
    *s.body.lock().unwrap() = Some(b);
    *s.auth.lock().unwrap() = h
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let r = s.reply.lock().unwrap().clone();
    if r.get("fail").is_some() {
        return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response();
    }
    Json(r).into_response()
}

async fn chat_server(reply: Value) -> (String, Seen) {
    let seen = Seen::default();
    *seen.reply.lock().unwrap() = reply;
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_handler))
        .with_state(seen.clone());
    (format!("http://{}/v1", serve(app).await), seen)
}

fn llm(base: &str, extra: Value, key_env: Option<&str>) -> Box<dyn voicelink_backends::ChatModel> {
    build_llm(&LlmConfig::OpenAi {
        base_url: base.into(),
        model: "m".into(),
        api_key_env: key_env.map(str::to_owned),
        extra: extra.as_object().cloned().unwrap_or_default(),
    })
    .unwrap()
}

#[tokio::test]
async fn chat_parses_string_arguments_and_serialises_history() {
    let (base, seen) = chat_server(json!({"choices":[{"message":{
        "role":"assistant","content":null,
        "tool_calls":[{"id":"abc","type":"function","function":{"name":"send_keys","arguments":"{\"text\":\"ls\"}"}}]
    }}]})).await;
    let m = llm(
        &format!("{base}/"),
        json!({"loom_options":{"scaffold":false}}),
        Some("HOME"),
    );
    let history = vec![
        user("list files"),
        ChatMessage {
            role: Role::Assistant,
            content: None,
            tool_calls: vec![voicelink_backends::ToolCall {
                id: "t1".into(),
                name: "send_keys".into(),
                arguments: json!({"text":"pwd"}),
            }],
            tool_call_id: None,
        },
        ChatMessage {
            role: Role::Tool,
            content: Some("/home".into()),
            tool_calls: vec![],
            tool_call_id: Some("t1".into()),
        },
    ];
    let out = m.chat(&req(history)).await.unwrap();
    assert_eq!(out.role, Role::Assistant);
    assert_eq!(out.content, None);
    assert_eq!(out.tool_calls.len(), 1);
    assert_eq!(out.tool_calls[0].id, "abc");
    assert_eq!(out.tool_calls[0].arguments, json!({"text":"ls"}));

    let b = seen.body.lock().unwrap().clone().unwrap();
    assert_eq!(b["model"], "m");
    assert_eq!(b["max_tokens"], 64);
    assert_eq!(b["loom_options"], json!({"scaffold":false}));
    assert_eq!(b["tools"][0]["type"], "function");
    assert_eq!(b["tools"][0]["function"]["name"], "send_keys");
    assert_eq!(b["messages"][1]["content"], Value::Null);
    assert_eq!(
        b["messages"][1]["tool_calls"][0]["function"]["arguments"],
        "{\"text\":\"pwd\"}"
    );
    assert_eq!(b["messages"][1]["tool_calls"][0]["type"], "function");
    assert_eq!(b["messages"][2]["role"], "tool");
    assert_eq!(b["messages"][2]["tool_call_id"], "t1");
    let home = std::env::var("HOME").unwrap();
    assert_eq!(
        seen.auth.lock().unwrap().clone().unwrap(),
        format!("Bearer {}", home.trim())
    );
}

#[tokio::test]
async fn chat_parses_object_arguments_and_missing_ids() {
    let (base, _) = chat_server(json!({"choices":[{"message":{
        "content":"on it",
        "tool_calls":[
            {"function":{"name":"a","arguments":{"x":1}}},
            {"id":"","function":{"name":"b","arguments":""}},
            {"id":"keep","function":{"name":"c"}}
        ]
    }}]}))
    .await;
    let out = llm(&base, json!({}), None)
        .chat(&req(vec![user("x")]))
        .await
        .unwrap();
    assert_eq!(out.content.as_deref(), Some("on it"));
    let ids: Vec<_> = out.tool_calls.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids, ["call_0", "call_1", "keep"]);
    assert_eq!(out.tool_calls[0].arguments, json!({"x":1}));
    assert_eq!(out.tool_calls[1].arguments, json!({}));
    assert_eq!(out.tool_calls[2].arguments, json!({}));
}

#[tokio::test]
async fn chat_plain_reply_omits_tools_when_none_offered() {
    let (base, seen) = chat_server(json!({"choices":[{"message":{"content":"hi"}}]})).await;
    let mut r = req(vec![user("x")]);
    r.tools.clear();
    let out = llm(&base, json!({}), None).chat(&r).await.unwrap();
    assert_eq!(out.content.as_deref(), Some("hi"));
    assert!(out.tool_calls.is_empty());
    assert!(
        seen.body
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .get("tools")
            .is_none()
    );
}

#[tokio::test]
async fn chat_server_error_and_garbage() {
    let (base, _) = chat_server(json!({"fail":true})).await;
    let e = llm(&base, json!({}), None)
        .chat(&req(vec![user("x")]))
        .await
        .unwrap_err();
    assert!(
        matches!(&e, BackendError::Backend(m) if m.contains("500") && m.contains("boom")),
        "{e}"
    );
    let (base, _) = chat_server(json!({"nothing":1})).await;
    let e = llm(&base, json!({}), None)
        .chat(&req(vec![user("x")]))
        .await
        .unwrap_err();
    assert!(matches!(e, BackendError::Backend(_)));
    let e = llm("http://127.0.0.1:1/v1", json!({}), None)
        .chat(&req(vec![user("x")]))
        .await
        .unwrap_err();
    assert!(matches!(e, BackendError::Transport(_)), "{e}");
}

#[tokio::test]
async fn stt_openai_posts_16k_wav_multipart() {
    type Parts = Vec<(String, Vec<u8>)>;
    #[derive(Clone, Default)]
    struct St(Arc<Mutex<Parts>>);
    async fn h(State(s): State<St>, mut mp: Multipart) -> Json<Value> {
        while let Some(f) = mp.next_field().await.unwrap() {
            let n = f.name().unwrap().to_owned();
            let data = f.bytes().await.unwrap().to_vec();
            s.0.lock().unwrap().push((n, data));
        }
        Json(json!({"text":"  hello there \n"}))
    }
    let st = St::default();
    let app = Router::new()
        .route("/v1/audio/transcriptions", post(h))
        .with_state(st.clone());
    let base = format!("http://{}/v1", serve(app).await);
    let stt = build_stt(&SttConfig::OpenAi {
        base_url: base,
        model: "whisper-1".into(),
        api_key_env: None,
    })
    .unwrap();
    let clip = AudioClip {
        sample_rate: 48_000,
        samples: vec![0.1; 48_000],
    };
    let t = stt.transcribe(&clip).await.unwrap();
    assert_eq!(t.text, "hello there");
    let parts = st.0.lock().unwrap().clone();
    let get = |n: &str| parts.iter().find(|p| p.0 == n).unwrap().1.clone();
    assert_eq!(get("model"), b"whisper-1");
    assert_eq!(get("response_format"), b"json");
    let wav = decode_wav(&get("file")).unwrap();
    assert_eq!(wav.sample_rate, 16_000);
    assert_eq!(wav.samples.len(), 16_000);
}

fn pcm_bytes(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (((i % 100) as i16 - 50) * 100).to_le_bytes())
        .collect()
}

async fn collect(rx: &mut mpsc::Receiver<AudioClip>) -> Vec<AudioClip> {
    let mut v = vec![];
    while let Some(c) = rx.recv().await {
        v.push(c);
    }
    v
}

fn tts_cfg(base: String) -> TtsConfig {
    TtsConfig::OpenAi {
        base_url: base,
        model: "pocket-tts".into(),
        voice: "alba".into(),
        api_key_env: None,
    }
}

#[tokio::test]
async fn tts_streams_pcm_in_100ms_chunks_with_odd_splits() {
    async fn h(Json(b): Json<Value>) -> Response {
        assert_eq!(b["response_format"], "pcm");
        assert_eq!(b["stream"], true);
        assert_eq!(b["voice"], "alba");
        assert_eq!(b["input"], "hello");
        let bytes = pcm_bytes(8_000); // 0.5 s at 16 kHz
        // Split at odd offsets to exercise sample straddling.
        let parts: Vec<_> = [&bytes[..333], &bytes[333..4_001], &bytes[4_001..]]
            .iter()
            .map(|p| Ok::<_, std::io::Error>(p.to_vec()))
            .collect();
        let mut r = Response::new(Body::from_stream(futures_util::stream::iter(parts)));
        r.headers_mut()
            .insert("x-sample-rate", HeaderValue::from_static("16000"));
        r.headers_mut()
            .insert("content-type", HeaderValue::from_static("audio/pcm"));
        r
    }
    let base = format!(
        "http://{}/v1",
        serve(Router::new().route("/v1/audio/speech", post(h))).await
    );
    let tts = build_tts(&tts_cfg(base)).unwrap();
    let (tx, mut rx) = mpsc::channel(64);
    tts.synthesize("hello", tx).await.unwrap();
    let clips = collect(&mut rx).await;
    assert!(clips.iter().all(|c| c.sample_rate == 16_000));
    assert_eq!(clips.iter().map(|c| c.samples.len()).sum::<usize>(), 8_000);
    assert_eq!(clips[0].samples.len(), 1_600);
    let expect: Vec<f32> = (0..8_000)
        .map(|i| (((i % 100) as i16 - 50) * 100) as f32 / 32768.0)
        .collect();
    let got: Vec<f32> = clips.into_iter().flat_map(|c| c.samples).collect();
    assert_eq!(got, expect);
}

#[tokio::test]
async fn tts_default_rate_without_header() {
    async fn h() -> Response {
        ([("content-type", "audio/pcm")], pcm_bytes(2_400)).into_response()
    }
    let base = format!(
        "http://{}/v1",
        serve(Router::new().route("/v1/audio/speech", post(h))).await
    );
    let (tx, mut rx) = mpsc::channel(8);
    build_tts(&tts_cfg(base))
        .unwrap()
        .synthesize("x", tx)
        .await
        .unwrap();
    let clips = collect(&mut rx).await;
    assert_eq!(clips.len(), 1);
    assert_eq!(clips[0].sample_rate, 24_000);
}

#[tokio::test]
async fn tts_cancels_promptly_when_sink_closes() {
    async fn h() -> Response {
        let s = futures_util::stream::unfold(0u32, |n| async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Some((Ok::<_, std::io::Error>(pcm_bytes(480)), n + 1))
        });
        let mut r = Response::new(Body::from_stream(s));
        r.headers_mut()
            .insert("content-type", HeaderValue::from_static("audio/pcm"));
        r
    }
    let base = format!(
        "http://{}/v1",
        serve(Router::new().route("/v1/audio/speech", post(h))).await
    );
    let tts = build_tts(&tts_cfg(base)).unwrap();
    let (tx, mut rx) = mpsc::channel(1);
    let task = tokio::spawn(async move { tts.synthesize("forever", tx).await });
    let first = rx.recv().await.unwrap();
    assert_eq!(first.samples.len(), 2_400);
    drop(rx);
    let t = Instant::now();
    let r = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .expect("did not stop")
        .unwrap();
    assert!(matches!(r, Err(BackendError::Cancelled)), "{r:?}");
    assert!(t.elapsed() < Duration::from_secs(1));
}

#[tokio::test]
async fn tts_cancels_when_sink_already_closed() {
    let (tx, rx) = mpsc::channel(1);
    drop(rx);
    let r = build_tts(&tts_cfg("http://127.0.0.1:1/v1".into()))
        .unwrap()
        .synthesize("x", tx)
        .await;
    assert!(matches!(r, Err(BackendError::Cancelled)));
}

#[tokio::test]
async fn tts_falls_back_to_complete_wav() {
    async fn h() -> Response {
        let clip = AudioClip {
            sample_rate: 22_050,
            samples: vec![0.25; 4_410],
        };
        ([("content-type", "audio/wav")], encode_wav(&clip)).into_response()
    }
    let base = format!(
        "http://{}/v1",
        serve(Router::new().route("/v1/audio/speech", post(h))).await
    );
    let (tx, mut rx) = mpsc::channel(16);
    build_tts(&tts_cfg(base))
        .unwrap()
        .synthesize("x", tx)
        .await
        .unwrap();
    let clips = collect(&mut rx).await;
    assert!(clips.iter().all(|c| c.sample_rate == 22_050));
    assert_eq!(clips.iter().map(|c| c.samples.len()).sum::<usize>(), 4_410);
    assert_eq!(clips[0].samples.len(), 2_205);
}

#[tokio::test]
async fn tts_server_error_and_bad_content_type() {
    async fn bad() -> Response {
        (StatusCode::BAD_REQUEST, "Unknown voice").into_response()
    }
    async fn mp3() -> Response {
        ([("content-type", "audio/mpeg")], vec![1u8, 2, 3]).into_response()
    }
    let base = format!(
        "http://{}/v1",
        serve(Router::new().route("/v1/audio/speech", post(bad))).await
    );
    let (tx, _rx) = mpsc::channel(1);
    let e = build_tts(&tts_cfg(base))
        .unwrap()
        .synthesize("x", tx)
        .await
        .unwrap_err();
    assert!(
        matches!(&e, BackendError::Backend(m) if m.contains("400") && m.contains("Unknown voice")),
        "{e}"
    );
    let base = format!(
        "http://{}/v1",
        serve(Router::new().route("/v1/audio/speech", post(mp3))).await
    );
    let (tx, _rx) = mpsc::channel(1);
    let e = build_tts(&tts_cfg(base))
        .unwrap()
        .synthesize("x", tx)
        .await
        .unwrap_err();
    assert!(matches!(e, BackendError::Backend(_)));
}

async fn nemotron_ws(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(|mut s: WebSocket| async move {
        let pack = |v: Value| Message::Binary(rmp_serde::to_vec_named(&v).unwrap().into());
        s.send(pack(json!({"type":"Ready"}))).await.unwrap();
        let (mut frames, mut after_marker) = (0usize, 0usize);
        let mut marker = None;
        while let Some(Ok(Message::Binary(b))) = s.recv().await {
            let v: Value = rmp_serde::from_slice(&b).unwrap();
            match v["type"].as_str().unwrap() {
                "Audio" => {
                    assert_eq!(v["pcm"].as_array().unwrap().len(), 1_920);
                    if marker.is_some() {
                        after_marker += 1
                    } else {
                        frames += 1
                    }
                    s.send(pack(
                        json!({"type":"Step","step_idx":frames,"prs":[0.0,0.0,0.02]}),
                    ))
                    .await
                    .unwrap();
                }
                "Marker" => marker = Some(v["id"].clone()),
                other => panic!("unexpected {other}"),
            }
            if marker.is_some() && after_marker >= 6 {
                for w in [" hello", " wor", "ld"] {
                    s.send(pack(json!({"type":"Word","text":w,"start_time":0.0})))
                        .await
                        .unwrap();
                }
                s.send(pack(
                    json!({"type":"Marker","id":marker.clone().unwrap(),"frames":frames}),
                ))
                .await
                .unwrap();
                break;
            }
        }
    })
}

#[tokio::test]
async fn nemotron_marker_protocol() {
    let addr =
        serve(Router::new().route("/api/asr-streaming", axum::routing::get(nemotron_ws))).await;
    let stt = build_stt(&SttConfig::Nemotron {
        url: format!("ws://{addr}/api/asr-streaming"),
    })
    .unwrap();
    // 1 s at 16 kHz, resampled to 24 kHz = 25 frames of 80 ms (rounded up).
    let t = stt
        .transcribe(&AudioClip {
            sample_rate: 16_000,
            samples: vec![0.05; 16_000],
        })
        .await
        .unwrap();
    assert_eq!(t.text, "hello world");
}

#[tokio::test]
async fn nemotron_error_and_unreachable() {
    async fn h(ws: WebSocketUpgrade) -> Response {
        ws.on_upgrade(|mut s: WebSocket| async move {
            let m = rmp_serde::to_vec_named(
                &json!({"type":"Error","message":"ASR is serving another stream"}),
            )
            .unwrap();
            let _ = s.send(Message::Binary(m.into())).await;
        })
    }
    let addr = serve(Router::new().route("/ws", axum::routing::get(h))).await;
    let stt = build_stt(&SttConfig::Nemotron {
        url: format!("ws://{addr}/ws"),
    })
    .unwrap();
    let clip = AudioClip {
        sample_rate: 24_000,
        samples: vec![0.0; 2_400],
    };
    let e = stt.transcribe(&clip).await.unwrap_err();
    assert!(
        matches!(&e, BackendError::Backend(m) if m.contains("another stream")),
        "{e}"
    );
    // A busy server is still a live one: the probe must not report an outage.
    voicelink_backends::probe_stt(&SttConfig::Nemotron {
        url: format!("ws://{addr}/ws"),
    })
    .await
    .unwrap();
    let stt = build_stt(&SttConfig::Nemotron {
        url: "ws://127.0.0.1:1/x".into(),
    })
    .unwrap();
    assert!(matches!(
        stt.transcribe(&clip).await.unwrap_err(),
        BackendError::Transport(_)
    ));
}

#[test]
fn factories_reject_bad_configs() {
    assert!(matches!(
        build_tts(&TtsConfig::OpenAi {
            base_url: "ftp://x".into(),
            model: "m".into(),
            voice: "v".into(),
            api_key_env: None
        }),
        Err(BackendError::Config(m)) if m.contains("http")
    ));
    assert!(
        build_stt(&SttConfig::Nemotron {
            url: "http://x".into()
        })
        .is_err()
    );
    assert!(
        build_stt(&SttConfig::OpenAi {
            base_url: "not a url".into(),
            model: "m".into(),
            api_key_env: None
        })
        .is_err()
    );
    assert!(
        build_stt(&SttConfig::OpenAi {
            base_url: "http://x/v1".into(),
            model: "m".into(),
            api_key_env: Some("VL_SURELY_UNSET_VAR".into())
        })
        .is_err()
    );
}

#[tokio::test]
async fn nemotron_rejects_utterance_over_limit_without_connecting() {
    // Nothing listens on port 9: the guard must answer before any connection.
    let stt = build_stt(&SttConfig::Nemotron {
        url: "ws://127.0.0.1:9/api/asr-streaming".into(),
    })
    .unwrap();
    let rate = 16_000;
    let len = (voicelink_backends::NEMOTRON_MAX_UTTERANCE.as_secs() as usize + 1) * rate;
    let clip = AudioClip {
        sample_rate: rate as u32,
        samples: vec![0.0; len],
    };
    match stt.transcribe(&clip).await {
        Err(BackendError::Backend(m)) => assert!(m.contains("exceeds"), "{m}"),
        other => panic!("expected the length guard, got {other:?}"),
    }
}

#[tokio::test]
async fn probe_openai_distinguishes_up_down_and_http_errors() {
    use voicelink_backends::{probe_llm, probe_stt, probe_tts};
    let app = Router::new()
        .route("/ok/v1/models", get(|| async { Json(json!({"data": []})) }))
        .route(
            "/denied/v1/models",
            get(|| async { StatusCode::UNAUTHORIZED }),
        )
        .route(
            "/broken/v1/models",
            get(|| async { StatusCode::INTERNAL_SERVER_ERROR }),
        );
    let host = format!("http://{}", serve(app).await);
    let stt = |p: &str| SttConfig::OpenAi {
        base_url: format!("{host}/{p}/v1"),
        model: "m".into(),
        api_key_env: None,
    };
    probe_stt(&stt("ok")).await.unwrap();
    // No `/models` route (whisper-server, Pocket): a 404 still proves it is up.
    probe_stt(&stt("absent")).await.unwrap();
    for (p, code) in [("denied", "401"), ("broken", "500")] {
        match probe_stt(&stt(p)).await {
            Err(BackendError::Backend(m)) => {
                assert!(
                    m.contains(&format!("{host}/{p}/v1/models")) && m.contains(code),
                    "{m}"
                )
            }
            other => panic!("expected an HTTP error, got {other:?}"),
        }
    }
    let llm = LlmConfig::OpenAi {
        base_url: format!("{host}/ok/v1/"),
        model: "m".into(),
        api_key_env: None,
        extra: serde_json::Map::new(),
    };
    probe_llm(&llm).await.unwrap();
    // A chat base URL without its `/v1` 404s on `/models`: that is a
    // misconfiguration, not a live server.
    let llm_no_v1 = LlmConfig::OpenAi {
        base_url: format!("{host}/ok"),
        model: "m".into(),
        api_key_env: None,
        extra: serde_json::Map::new(),
    };
    match probe_llm(&llm_no_v1).await {
        Err(BackendError::Backend(m)) => assert!(m.contains("404") && m.contains("/v1"), "{m}"),
        other => panic!("expected a 404 error, got {other:?}"),
    }
    let tts = TtsConfig::OpenAi {
        base_url: "http://127.0.0.1:1/v1".into(),
        model: "m".into(),
        voice: "v".into(),
        api_key_env: None,
    };
    match probe_tts(&tts).await {
        Err(BackendError::Transport(m)) => {
            assert!(
                m.contains("http://127.0.0.1:1/v1/models") && m.contains("refused"),
                "{m}"
            )
        }
        other => panic!("expected a transport error, got {other:?}"),
    }
}

#[tokio::test]
async fn probe_nemotron_waits_for_ready_and_names_the_url() {
    use voicelink_backends::probe_stt;
    let host = serve(Router::new().route("/api/asr-streaming", get(nemotron_ws))).await;
    let url = format!("ws://{host}/api/asr-streaming");
    probe_stt(&SttConfig::Nemotron { url }).await.unwrap();
    let dead = "ws://127.0.0.1:1/api/asr-streaming";
    match probe_stt(&SttConfig::Nemotron { url: dead.into() }).await {
        Err(BackendError::Transport(m)) => assert!(m.contains(dead), "{m}"),
        other => panic!("expected a transport error, got {other:?}"),
    }
}
