//! Capture pipeline: device-rate mono -> 16 kHz -> echo cancellation -> VAD ->
//! utterance segmentation. Pure: no devices, driven by plain slices.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::AudioEvent;
use super::aec::{AEC_FRAME, AEC_RATE, EchoCanceller};
use super::playback::FarTap;
use super::resample::StreamResampler;
use super::vad::{SPEECH_RATE, Segmenter, SegmenterConfig, VAD_FRAME, VoiceScorer};

/// Expected output-to-input path delay handed to the echo canceller.
const INITIAL_DELAY_MS: i32 = 100;
/// How long after the last audible far-end frame we still treat playback as active.
const FAR_HOLD_MS: u32 = 300;

/// Everything between the microphone callback and [`AudioEvent`]s.
pub(crate) struct CapturePipeline {
    input_rs: StreamResampler,
    far_rs: StreamResampler,
    tap: Arc<FarTap>,
    aec: Option<EchoCanceller>,
    scorer: Box<dyn VoiceScorer>,
    seg: Segmenter,
    muted: Arc<AtomicBool>,
    near16: Vec<f32>,
    far16: VecDeque<f32>,
    cleaned: Vec<f32>,
    /// Milliseconds the far-end stays 'recent' after its last audible frame.
    far_hold_ms: u32,
}

impl CapturePipeline {
    /// Creates a pipeline for an input running at `input_rate`.
    pub(crate) fn new(
        input_rate: u32,
        tap: Arc<FarTap>,
        echo_cancel: bool,
        scorer: Box<dyn VoiceScorer>,
        seg_cfg: SegmenterConfig,
        muted: Arc<AtomicBool>,
    ) -> anyhow::Result<Self> {
        debug_assert_eq!(AEC_RATE, SPEECH_RATE);
        Ok(CapturePipeline {
            input_rs: StreamResampler::new(input_rate, SPEECH_RATE)?,
            far_rs: StreamResampler::new(tap.rate(), AEC_RATE)?,
            tap,
            aec: if echo_cancel {
                Some(EchoCanceller::new(INITIAL_DELAY_MS)?)
            } else {
                None
            },
            scorer,
            seg: Segmenter::new(seg_cfg),
            muted,
            near16: Vec::new(),
            far16: VecDeque::new(),
            cleaned: Vec::new(),
            far_hold_ms: 0,
        })
    }

    /// Feeds mono samples at the input rate and appends resulting events.
    pub(crate) fn push_input(&mut self, mono: &[f32], events: &mut Vec<AudioEvent>) {
        // Keep the far-end flowing even while muted so it stays aligned.
        let played = self.tap.drain();
        let mut far = Vec::new();
        self.far_rs.process(&played, &mut far);
        self.far16.extend(far);
        // Bound the backlog (500 ms) if capture stalls.
        let over = self.far16.len().saturating_sub(AEC_RATE as usize / 2);
        self.far16.drain(..over);

        if self.muted.load(Ordering::Relaxed) {
            self.near16.clear();
            self.cleaned.clear();
            self.far16.clear();
            self.seg.reset();
            return;
        }

        self.input_rs.process(mono, &mut self.near16);

        // Echo cancellation in 10 ms frames, 1:1 with the reference (zeros when idle).
        let mut produced = 0;
        while self.near16.len() - produced >= AEC_FRAME {
            let near = &self.near16[produced..produced + AEC_FRAME];
            match self.aec.as_mut() {
                Some(aec) => {
                    let far: Vec<f32> = (0..AEC_FRAME)
                        .map(|_| self.far16.pop_front().unwrap_or(0.0))
                        .collect();
                    let loud = far.iter().any(|v| v.abs() > 0.002);
                    self.far_hold_ms = if loud {
                        FAR_HOLD_MS
                    } else {
                        self.far_hold_ms.saturating_sub(10)
                    };
                    let out = aec.process(&far, near);
                    self.cleaned.extend(out);
                }
                None => self.cleaned.extend_from_slice(near),
            }
            produced += AEC_FRAME;
        }
        self.near16.drain(..produced);

        // VAD in 16 ms frames.
        let mut consumed = 0;
        while self.cleaned.len() - consumed >= VAD_FRAME {
            let frame = &self.cleaned[consumed..consumed + VAD_FRAME];
            self.seg.set_far_active(self.far_hold_ms > 0);
            let score = self.scorer.score(frame);
            if let Some(ev) = self.seg.push(frame, score) {
                events.push(ev);
            }
            consumed += VAD_FRAME;
        }
        self.cleaned.drain(..consumed);
    }
}

#[cfg(test)]
mod tests {
    use super::super::aec::tests::noise;
    use super::super::playback::{AudioSink, PlaybackQueue};
    use super::super::vad::tests::EnergyScorer;
    use super::*;
    use voicelink_backends::AudioClip;

    fn seg_cfg() -> SegmenterConfig {
        SegmenterConfig {
            threshold: 0.5,
            end_silence_ms: 400,
            min_utterance_ms: 200,
            preroll_ms: 100,
            start_frames: 2,
            start_frames_while_playing: 12,
            max_utterance_ms: 30_000,
        }
    }

    fn pipeline(rate: u32, aec: bool, muted: Arc<AtomicBool>) -> (CapturePipeline, Arc<FarTap>) {
        let tap = FarTap::new(rate);
        let p = CapturePipeline::new(
            rate,
            Arc::clone(&tap),
            aec,
            Box::new(EnergyScorer),
            seg_cfg(),
            muted,
        )
        .unwrap();
        (p, tap)
    }

    fn feed(p: &mut CapturePipeline, samples: &[f32], rate: u32) -> Vec<AudioEvent> {
        let mut ev = Vec::new();
        for chunk in samples.chunks(rate as usize / 100 + 7) {
            p.push_input(chunk, &mut ev);
        }
        ev
    }

    #[test]
    fn noise_burst_between_silences_becomes_one_utterance_at_48k() {
        let (mut p, _) = pipeline(48_000, false, Arc::new(AtomicBool::new(false)));
        let mut audio = vec![0.0; 48_000];
        audio.extend(noise(48_000, 0.4, 1)); // 1 s burst
        audio.extend(vec![0.0; 48_000]);
        let ev = feed(&mut p, &audio, 48_000);
        assert_eq!(ev.len(), 2, "{ev:?}");
        assert_eq!(ev[0], AudioEvent::SpeechStarted);
        let AudioEvent::Utterance(clip) = &ev[1] else {
            panic!()
        };
        assert_eq!(clip.sample_rate, 16_000);
        let secs = clip.samples.len() as f32 / 16_000.0;
        assert!((1.0..1.8).contains(&secs), "{secs}");
    }

    #[test]
    fn muted_input_is_ignored_and_unmute_resumes() {
        let muted = Arc::new(AtomicBool::new(true));
        let (mut p, _) = pipeline(16_000, false, Arc::clone(&muted));
        let burst = noise(16_000, 0.4, 2);
        assert!(feed(&mut p, &burst, 16_000).is_empty());
        muted.store(false, Ordering::Relaxed);
        let mut audio = burst;
        audio.extend(vec![0.0; 16_000]);
        let ev = feed(&mut p, &audio, 16_000);
        assert!(ev.iter().any(|e| matches!(e, AudioEvent::Utterance(_))));
    }

    /// Plays 6 s of noise, loops it back to the mic 50 ms late at half level
    /// (no one is talking), and returns the 10 ms tick of the first speech event
    /// of any kind in a 9 s run.
    fn first_echo_event_tick(echo_cancel: bool) -> Option<usize> {
        let rate = 16_000;
        let q = PlaybackQueue::new(rate);
        let mut p = CapturePipeline::new(
            rate,
            q.tap(),
            echo_cancel,
            Box::new(EnergyScorer),
            seg_cfg(),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        q.push(AudioClip {
            sample_rate: rate,
            samples: noise(rate as usize * 6, 0.3, 9),
        });
        let delay = 800usize;
        let mut history = vec![0.0f32; delay];
        for t in 0..900 {
            let mut out = vec![0.0; 160];
            q.fill(&mut out);
            history.extend_from_slice(&out);
            let mic: Vec<f32> = history[history.len() - 160 - delay..history.len() - delay]
                .iter()
                .map(|v| v * 0.5)
                .collect();
            let mut ev = Vec::new();
            p.push_input(&mic, &mut ev);
            if !ev.is_empty() {
                return Some(t);
            }
        }
        None
    }

    #[test]
    fn echo_is_heard_as_speech_without_aec_but_not_with_it() {
        // Control: the raw echo trips the VAD almost at once.
        let control = first_echo_event_tick(false).expect("control must hear the echo");
        assert!(control < 40, "control triggered at tick {control}");
        // With AEC the played signal never reaches the segmenter.
        assert_eq!(first_echo_event_tick(true), None);
    }
}
