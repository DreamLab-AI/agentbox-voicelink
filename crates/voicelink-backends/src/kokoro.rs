//! Embedded Kokoro-82M text-to-speech (feature `kokoro`).
//!
//! Text is phonemised by `misaki-rs` (dictionary and tagger only: its
//! GPL espeak-ng fallback is disabled) and synthesised by the Kokoro ONNX model
//! through ONNX Runtime.

use crate::kokoro_vocab::VOCAB;
use crate::{AudioClip, BackendError, Result, TextToSpeech};
use async_trait::async_trait;
use misaki_rs::{G2P, Language};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

const RATE: u32 = 24_000;
/// Longest phoneme sequence the model accepts (style table length).
const MAX_TOKENS: usize = 510;
/// Target phonemes per synthesis segment, to keep latency to first audio low.
const SEGMENT_TOKENS: usize = 300;
const STYLE_DIM: usize = 256;

struct Engine {
    _env: ort::environment::Environment,
    session: Mutex<Session>,
    g2p: G2P,
    style: Vec<f32>,
    vocab: HashMap<char, i64>,
    input_name: &'static str,
}

pub(crate) struct KokoroTts {
    engine: Arc<Engine>,
    voice: String,
}

/// ONNX Runtime allows one environment per process: create it once and share it.
fn environment() -> Result<ort::environment::Environment> {
    static ENV: std::sync::OnceLock<std::result::Result<ort::environment::Environment, String>> =
        std::sync::OnceLock::new();
    ENV.get_or_init(|| ort::init().build().map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| BackendError::Config(format!("ONNX Runtime: {e}")))
}

fn cfg<E: std::fmt::Display>(what: &str) -> impl Fn(E) -> BackendError + '_ {
    move |e| BackendError::Config(format!("{what}: {e}"))
}

/// Parses a little-endian `float32` `.npy` payload.
fn parse_npy(bytes: &[u8]) -> Result<Vec<f32>> {
    let bad = || BackendError::Config("voices file: malformed .npy entry".into());
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad());
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        _ => return Err(bad()),
    };
    let header =
        std::str::from_utf8(bytes.get(start..start + hlen).ok_or_else(bad)?).map_err(|_| bad())?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err(BackendError::Config(
            "voices file: expected C-order little-endian float32".into(),
        ));
    }
    let data = &bytes[start + hlen..];
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn load_voice(path: &str, voice: &str) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path).map_err(cfg("voices file"))?;
    let mut zip =
        zip::ZipArchive::new(file).map_err(cfg("voices file (expected an .npz archive)"))?;
    let mut names: Vec<String> = zip
        .file_names()
        .filter_map(|n| n.strip_suffix(".npy").map(str::to_owned))
        .collect();
    names.sort();
    let mut entry = zip.by_name(&format!("{voice}.npy")).map_err(|_| {
        BackendError::Config(format!(
            "voice {voice:?} not in voices file; available: {}",
            names.join(", ")
        ))
    })?;
    let mut buf = Vec::new();
    entry.read_to_end(&mut buf).map_err(cfg("voices file"))?;
    let v = parse_npy(&buf)?;
    if v.len() < STYLE_DIM || v.len() % STYLE_DIM != 0 {
        return Err(BackendError::Config(
            "voices file: unexpected style shape".into(),
        ));
    }
    Ok(v)
}

impl KokoroTts {
    pub(crate) fn new(model_path: &str, voices_path: &str, voice: &str) -> Result<Self> {
        if !std::path::Path::new(model_path).is_file() {
            return Err(BackendError::Config(format!(
                "Kokoro model not found: {model_path}"
            )));
        }
        let style = load_voice(voices_path, voice)?;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
        let env = environment()?;
        let load = || -> std::result::Result<Session, String> {
            let mut b = Session::builder(&env).map_err(|e| e.to_string())?;
            b = b.with_intra_threads(threads).map_err(|e| e.to_string())?;
            b.commit_from_file(model_path).map_err(|e| e.to_string())
        };
        let session = load().map_err(cfg("Kokoro model"))?;
        let input_name = if session.inputs().iter().any(|i| i.name() == "input_ids") {
            "input_ids"
        } else {
            "tokens"
        };
        let lang = if voice.starts_with('b') {
            Language::EnglishGB
        } else {
            Language::EnglishUS
        };
        Ok(Self {
            engine: Arc::new(Engine {
                _env: env,
                session: Mutex::new(session),
                g2p: G2P::new(lang),
                style,
                vocab: VOCAB.iter().copied().collect(),
                input_name,
            }),
            voice: voice.to_owned(),
        })
    }
}

/// Splits text into sentence-sized pieces.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?' | '\n') && chars.peek().is_none_or(|n| n.is_whitespace()) {
            if !cur.trim().is_empty() {
                out.push(cur.trim().to_owned());
            }
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_owned());
    }
    out
}

impl Engine {
    /// Phonemises a sentence into vocabulary token ids.
    fn tokens(&self, sentence: &str) -> Result<Vec<i64>> {
        let (phonemes, _) = self
            .g2p
            .g2p(sentence)
            .map_err(|e| BackendError::Backend(format!("phonemiser: {e}")))?;
        Ok(phonemes
            .chars()
            .filter_map(|c| self.vocab.get(&c).copied())
            .collect())
    }

    fn infer(&self, tokens: &[i64]) -> Result<Vec<f32>> {
        let n = tokens.len().min(MAX_TOKENS);
        let mut ids = Vec::with_capacity(n + 2);
        ids.push(0);
        ids.extend_from_slice(&tokens[..n]);
        ids.push(0);
        let rows = self.style.len() / STYLE_DIM;
        let row = n.min(rows - 1);
        let style = self.style[row * STYLE_DIM..(row + 1) * STYLE_DIM].to_vec();
        let be = |e: ort::Error| BackendError::Backend(format!("kokoro inference: {e}"));
        let ids_len = ids.len();
        let inputs = ort::inputs![
            self.input_name => Tensor::from_array(([1usize, ids_len], ids)).map_err(be)?,
            "style" => Tensor::from_array(([1usize, STYLE_DIM], style)).map_err(be)?,
            "speed" => Tensor::from_array(([1usize], vec![1.0f32])).map_err(be)?,
        ];
        let mut session = self
            .session
            .lock()
            .map_err(|_| BackendError::Backend("kokoro session poisoned".into()))?;
        let outputs = session.run(inputs).map_err(be)?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>().map_err(be)?;
        Ok(data.to_vec())
    }
}

#[async_trait]
impl TextToSpeech for KokoroTts {
    async fn synthesize(&self, text: &str, sink: mpsc::Sender<AudioClip>) -> Result<()> {
        let step = (RATE / 10) as usize;
        let mut batches: Vec<Vec<i64>> = Vec::new();
        // Phonemise up front (cheap) so segments can be merged to a sensible size.
        let eng = Arc::clone(&self.engine);
        let text = text.to_owned();
        let sentence_tokens = tokio::task::spawn_blocking(move || {
            sentences(&text)
                .iter()
                .map(|s| eng.tokens(s))
                .collect::<Result<Vec<_>>>()
        })
        .await
        .map_err(|e| BackendError::Backend(format!("kokoro task: {e}")))??;
        for t in sentence_tokens {
            for piece in t.chunks(MAX_TOKENS) {
                match batches.last_mut() {
                    Some(b) if b.len() + piece.len() < SEGMENT_TOKENS => {
                        b.push(16); // a space token between sentences
                        b.extend_from_slice(piece);
                    }
                    _ => batches.push(piece.to_vec()),
                }
            }
        }
        for tokens in batches {
            if sink.is_closed() {
                return Err(BackendError::Cancelled);
            }
            let eng = Arc::clone(&self.engine);
            let samples = tokio::task::spawn_blocking(move || eng.infer(&tokens))
                .await
                .map_err(|e| BackendError::Backend(format!("kokoro task: {e}")))??;
            for c in samples.chunks(step) {
                sink.send(AudioClip {
                    sample_rate: RATE,
                    samples: c.to_vec(),
                })
                .await
                .map_err(|_| BackendError::Cancelled)?;
            }
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("kokoro:{}", self.voice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sentences() {
        assert_eq!(
            sentences("Hello there. How are you? Fine!\nOk"),
            ["Hello there.", "How are you?", "Fine!", "Ok"]
        );
        assert_eq!(sentences("v1.5 is out"), ["v1.5 is out"]);
        assert!(sentences("  ").is_empty());
    }

    #[test]
    fn parses_npy_v1() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
        let mut h = header.as_bytes().to_vec();
        while !(10 + h.len() + 1).is_multiple_of(64) {
            h.push(b' ');
        }
        h.push(b'\n');
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend_from_slice(&(h.len() as u16).to_le_bytes());
        b.extend_from_slice(&h);
        b.extend_from_slice(&1.5f32.to_le_bytes());
        b.extend_from_slice(&(-2.0f32).to_le_bytes());
        assert_eq!(parse_npy(&b).unwrap(), [1.5, -2.0]);
        assert!(parse_npy(b"nope").is_err());
    }

    #[test]
    fn vocab_covers_ipa_and_space() {
        let v: HashMap<char, i64> = VOCAB.iter().copied().collect();
        assert_eq!(v[&' '], 16);
        assert!(v.contains_key(&'ə') && v.contains_key(&'ˈ'));
    }
}
