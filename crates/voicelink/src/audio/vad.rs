//! Voice-activity scoring and utterance segmentation.
//!
//! The scorer is pure Rust (`earshot`, a small neural VAD, ~40 KiB model, no
//! ONNX runtime). It works on 256-sample frames (16 ms at 16 kHz). The
//! [`Segmenter`] is a plain state machine over those scores, so it is tested on
//! synthetic signals with a scripted scorer.

use std::collections::VecDeque;

use voicelink_backends::AudioClip;

use super::AudioEvent;

/// Sample rate of the speech path.
pub(crate) const SPEECH_RATE: u32 = 16_000;
/// Samples per VAD frame (16 ms).
pub(crate) const VAD_FRAME: usize = 256;
const FRAME_MS: u32 = 16;

/// Scores one frame of 16 kHz mono audio with a speech probability in 0..=1.
pub(crate) trait VoiceScorer: Send {
    /// `frame` holds exactly [`VAD_FRAME`] samples.
    fn score(&mut self, frame: &[f32]) -> f32;
}

/// `earshot`-backed scorer.
pub(crate) struct EarshotScorer {
    detector: Box<earshot::Detector>,
}

impl EarshotScorer {
    pub(crate) fn new() -> Self {
        EarshotScorer {
            detector: earshot::Detector::default_boxed(),
        }
    }
}

impl VoiceScorer for EarshotScorer {
    fn score(&mut self, frame: &[f32]) -> f32 {
        self.detector.predict_f32(frame)
    }
}

/// Segmentation parameters, derived from `config.audio`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SegmenterConfig {
    /// Score at or above which a frame is voiced.
    pub threshold: f32,
    /// Trailing silence that closes an utterance.
    pub end_silence_ms: u32,
    /// Shortest speech (excluding trailing silence) worth transcribing.
    pub min_utterance_ms: u32,
    /// Audio kept from before speech was detected.
    pub preroll_ms: u32,
    /// Consecutive voiced frames needed to declare speech started.
    pub start_frames: u32,
    /// Stricter start requirement while our own audio is playing: the echo
    /// canceller needs a moment to lock on at the start of each reply, and
    /// that residue must not read as the operator barging in.
    pub start_frames_while_playing: u32,
    /// Hard cap on one utterance.
    pub max_utterance_ms: u32,
}

impl SegmenterConfig {
    /// Builds a segmenter configuration from the user's audio settings.
    pub(crate) fn from_audio(a: &crate::config::Audio) -> Self {
        SegmenterConfig {
            threshold: a.vad_threshold,
            end_silence_ms: a.end_silence_ms,
            min_utterance_ms: a.min_utterance_ms,
            preroll_ms: 240,
            start_frames: 3,
            start_frames_while_playing: 12,
            max_utterance_ms: a.max_utterance_ms,
        }
    }
}

enum State {
    Idle { voiced_run: u32 },
    Speaking { silence_frames: u32, frames: u32 },
}

/// Turns a stream of frame scores into [`AudioEvent`]s.
pub(crate) struct Segmenter {
    cfg: SegmenterConfig,
    state: State,
    preroll: VecDeque<Vec<f32>>,
    buf: Vec<f32>,
    far_active: bool,
}

impl Segmenter {
    pub(crate) fn new(cfg: SegmenterConfig) -> Self {
        Segmenter {
            cfg,
            state: State::Idle { voiced_run: 0 },
            preroll: VecDeque::new(),
            buf: Vec::new(),
            far_active: false,
        }
    }

    /// Tells the segmenter whether our own audio is currently playing.
    pub(crate) fn set_far_active(&mut self, active: bool) {
        self.far_active = active;
    }

    /// Discards any partial utterance (used when muting).
    pub(crate) fn reset(&mut self) {
        self.state = State::Idle { voiced_run: 0 };
        self.preroll.clear();
        self.buf.clear();
    }

    /// Feeds one frame and its score; returns at most one event.
    pub(crate) fn push(&mut self, frame: &[f32], score: f32) -> Option<AudioEvent> {
        let voiced = score >= self.cfg.threshold;
        match &mut self.state {
            State::Idle { voiced_run } => {
                self.preroll.push_back(frame.to_vec());
                let keep = (self.cfg.preroll_ms / FRAME_MS).max(self.cfg.start_frames) as usize;
                while self.preroll.len() > keep {
                    self.preroll.pop_front();
                }
                *voiced_run = if voiced { *voiced_run + 1 } else { 0 };
                let need = if self.far_active {
                    self.cfg
                        .start_frames_while_playing
                        .max(self.cfg.start_frames)
                } else {
                    self.cfg.start_frames
                };
                if *voiced_run >= need {
                    self.buf.clear();
                    for f in self.preroll.drain(..) {
                        self.buf.extend_from_slice(&f);
                    }
                    self.state = State::Speaking {
                        silence_frames: 0,
                        frames: self.cfg.start_frames,
                    };
                    return Some(AudioEvent::SpeechStarted);
                }
                None
            }
            State::Speaking {
                silence_frames,
                frames,
            } => {
                self.buf.extend_from_slice(frame);
                *frames += 1;
                *silence_frames = if voiced { 0 } else { *silence_frames + 1 };
                let ended = *silence_frames * FRAME_MS >= self.cfg.end_silence_ms;
                let capped = *frames * FRAME_MS >= self.cfg.max_utterance_ms;
                if !(ended || capped) {
                    return None;
                }
                if capped && !ended {
                    // Force-cut: hand over everything captured so far and keep
                    // capturing into a new utterance, with no gap and no loss.
                    *frames = 0;
                    *silence_frames = 0;
                    return Some(AudioEvent::Utterance(AudioClip {
                        sample_rate: SPEECH_RATE,
                        samples: std::mem::take(&mut self.buf),
                    }));
                }
                let speech_ms = (*frames - *silence_frames) * FRAME_MS;
                // Keep a short tail of silence so the STT sees a natural end.
                let tail_frames = (*silence_frames).min(200 / FRAME_MS) as usize;
                let drop_frames = *silence_frames as usize - tail_frames;
                let mut samples = std::mem::take(&mut self.buf);
                samples.truncate(samples.len().saturating_sub(drop_frames * VAD_FRAME));
                self.state = State::Idle { voiced_run: 0 };
                self.preroll.clear();
                (speech_ms >= self.cfg.min_utterance_ms).then_some(AudioEvent::Utterance(
                    AudioClip {
                        sample_rate: SPEECH_RATE,
                        samples,
                    },
                ))
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Energy-based scorer so tone and noise bursts count as "speech".
    pub(crate) struct EnergyScorer;
    impl VoiceScorer for EnergyScorer {
        fn score(&mut self, frame: &[f32]) -> f32 {
            let rms = (frame.iter().map(|v| v * v).sum::<f32>() / frame.len() as f32).sqrt();
            if rms > 0.05 { 1.0 } else { 0.0 }
        }
    }

    fn cfg() -> SegmenterConfig {
        SegmenterConfig {
            threshold: 0.5,
            end_silence_ms: 320,
            min_utterance_ms: 160,
            preroll_ms: 64,
            start_frames: 2,
            start_frames_while_playing: 6,
            max_utterance_ms: 30_000,
        }
    }

    fn feed(seg: &mut Segmenter, voiced: bool, frames: usize) -> Vec<AudioEvent> {
        let mut out = Vec::new();
        for _ in 0..frames {
            let f = vec![if voiced { 0.3 } else { 0.0 }; VAD_FRAME];
            if let Some(e) = seg.push(&f, EnergyScorer.score(&f)) {
                out.push(e);
            }
        }
        out
    }

    #[test]
    fn silence_emits_nothing() {
        let mut s = Segmenter::new(cfg());
        assert!(feed(&mut s, false, 200).is_empty());
    }

    #[test]
    fn burst_then_silence_yields_start_then_utterance_with_preroll() {
        let mut s = Segmenter::new(cfg());
        feed(&mut s, false, 10);
        let started = feed(&mut s, true, 20);
        assert_eq!(started, vec![AudioEvent::SpeechStarted]);
        let end = feed(&mut s, false, 20);
        assert_eq!(end.len(), 1);
        let AudioEvent::Utterance(clip) = &end[0] else {
            panic!("expected utterance")
        };
        assert_eq!(clip.sample_rate, SPEECH_RATE);
        // 4 preroll frames (64 ms) at most, 20 voiced, 12 tail frames trimmed to 12 (200 ms/16).
        let frames = clip.samples.len() / VAD_FRAME;
        assert!((20..=40).contains(&frames), "{frames}");
        // Preroll silence precedes the speech.
        assert_eq!(clip.samples[0], 0.0);
        assert!(clip.samples.iter().any(|v| *v > 0.2));
    }

    #[test]
    fn short_blip_is_dropped_but_still_signalled() {
        let mut s = Segmenter::new(cfg());
        let mut events = feed(&mut s, true, 2); // exactly start_frames
        events.extend(feed(&mut s, false, 30));
        assert_eq!(events, vec![AudioEvent::SpeechStarted]);
    }

    #[test]
    fn brief_pause_inside_speech_does_not_split() {
        let mut s = Segmenter::new(cfg());
        let mut events = feed(&mut s, true, 10);
        events.extend(feed(&mut s, false, 10)); // 160 ms < end_silence
        events.extend(feed(&mut s, true, 10));
        events.extend(feed(&mut s, false, 30));
        let utterances = events
            .iter()
            .filter(|e| matches!(e, AudioEvent::Utterance(_)))
            .count();
        assert_eq!(utterances, 1);
    }

    #[test]
    fn seventy_seconds_of_tone_become_30_30_and_10_second_clips_without_loss() {
        let mut c = cfg();
        c.max_utterance_ms = 30_000;
        let mut s = Segmenter::new(c);
        let frames = 70 * 16_000 / VAD_FRAME + 5; // 70 s plus rounding up to whole frames
        let tone: Vec<f32> = (0..VAD_FRAME)
            .map(|i| 0.3 + (i % 2) as f32 * 0.01)
            .collect();
        let mut clips: Vec<Vec<f32>> = Vec::new();
        let mut started = 0;
        let mut feed_frame = |s: &mut Segmenter, f: &[f32], score: f32| match s.push(f, score) {
            Some(AudioEvent::SpeechStarted) => started += 1,
            Some(AudioEvent::Utterance(c)) => clips.push(c.samples),
            None => {}
        };
        let voiced_frames = 70 * 16_000 / VAD_FRAME; // 4375 exactly
        for _ in 0..voiced_frames {
            feed_frame(&mut s, &tone, 1.0);
        }
        for _ in 0..(frames - voiced_frames + 40) {
            feed_frame(&mut s, &vec![0.0; VAD_FRAME], 0.0);
        }
        assert_eq!(started, 1, "a force-cut continues the same speech");
        assert_eq!(clips.len(), 3);
        assert_eq!(clips[0].len(), 30 * 16_000);
        assert_eq!(clips[1].len(), 30 * 16_000);
        // The last clip is the remaining 10 s plus the short silence tail.
        let last = clips[2].len();
        assert!(
            (10 * 16_000..=10 * 16_000 + 3_300).contains(&last),
            "{last}"
        );
        let voiced: usize = clips
            .iter()
            .map(|c| c.iter().filter(|v| **v > 0.1).count())
            .sum();
        assert_eq!(
            voiced,
            voiced_frames * VAD_FRAME,
            "no voiced sample lost or duplicated"
        );
    }

    #[test]
    fn long_speech_is_cut_at_the_cap() {
        let mut c = cfg();
        c.max_utterance_ms = 1000;
        let mut s = Segmenter::new(c);
        let events = feed(&mut s, true, 100);
        assert!(events.iter().any(|e| matches!(e, AudioEvent::Utterance(_))));
    }

    #[test]
    fn stricter_start_requirement_while_our_audio_plays() {
        let mut s = Segmenter::new(cfg());
        s.set_far_active(true);
        assert!(
            feed(&mut s, true, 5).is_empty(),
            "5 voiced frames are not enough while playing"
        );
        assert_eq!(feed(&mut s, true, 1), vec![AudioEvent::SpeechStarted]);
        let mut s = Segmenter::new(cfg());
        s.set_far_active(false);
        assert_eq!(feed(&mut s, true, 2), vec![AudioEvent::SpeechStarted]);
    }

    #[test]
    fn reset_discards_partial_utterance() {
        let mut s = Segmenter::new(cfg());
        feed(&mut s, true, 10);
        s.reset();
        assert!(feed(&mut s, false, 40).is_empty());
    }

    #[test]
    fn earshot_scores_silence_low_on_256_sample_frames() {
        let mut sc = EarshotScorer::new();
        let mut last = 1.0;
        for _ in 0..20 {
            last = sc.score(&vec![0.0; VAD_FRAME]);
        }
        assert!(last < 0.2, "silence scored {last}");
    }
}
