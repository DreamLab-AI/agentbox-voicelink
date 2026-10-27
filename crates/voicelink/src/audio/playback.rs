//! Playback queue shared between the speaker task and the output callback.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use voicelink_backends::AudioClip;

use super::resample::StreamResampler;

/// How long after the last played sample we still report "playing" (device
/// buffers keep draining after the queue empties).
const TAIL: Duration = Duration::from_millis(250);

/// What the speaker task needs from an audio output.
pub(crate) trait AudioSink: Send + Sync {
    /// Queues a clip of any sample rate for playback.
    fn push(&self, clip: AudioClip);
    /// Drops everything queued, immediately (barge-in).
    fn stop(&self);
    /// Whether audio is queued or was audible a moment ago.
    fn is_playing(&self) -> bool;
}

/// Reference signal for the echo canceller: exactly what the output callback
/// emitted, at the device rate.
pub(crate) struct FarTap {
    rate: u32,
    buf: Mutex<VecDeque<f32>>,
}

impl FarTap {
    pub(crate) fn new(rate: u32) -> Arc<Self> {
        Arc::new(FarTap {
            rate,
            buf: Mutex::new(VecDeque::new()),
        })
    }

    pub(crate) fn rate(&self) -> u32 {
        self.rate
    }

    fn lock(&self) -> MutexGuard<'_, VecDeque<f32>> {
        self.buf.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn push(&self, samples: &[f32]) {
        let mut b = self.lock();
        b.extend(samples.iter().copied());
        // Bound the backlog (2 s) in case capture stalls.
        let cap = self.rate as usize * 2;
        let over = b.len().saturating_sub(cap);
        b.drain(..over);
    }

    /// Takes everything accumulated since the last call.
    pub(crate) fn drain(&self) -> Vec<f32> {
        self.lock().drain(..).collect()
    }
}

/// Idle time after the last pushed chunk before the resampler tail is flushed.
const FLUSH_AFTER: Duration = Duration::from_millis(150);

struct State {
    queue: VecDeque<f32>,
    last_audible: Option<Instant>,
    /// One continuous resampler per source rate, so chunks of a streamed
    /// reply join without clicks.
    stream: Option<(u32, StreamResampler)>,
    last_push: Option<Instant>,
}

/// Mono output queue at the device sample rate.
pub(crate) struct PlaybackQueue {
    device_rate: u32,
    state: Mutex<State>,
    tap: Arc<FarTap>,
}

impl PlaybackQueue {
    /// Creates a queue for a device running at `device_rate`.
    pub(crate) fn new(device_rate: u32) -> Self {
        PlaybackQueue {
            device_rate,
            state: Mutex::new(State {
                queue: VecDeque::new(),
                last_audible: None,
                stream: None,
                last_push: None,
            }),
            tap: FarTap::new(device_rate),
        }
    }

    /// The far-end reference fed by [`fill`](Self::fill).
    pub(crate) fn tap(&self) -> Arc<FarTap> {
        Arc::clone(&self.tap)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Fills one mono output buffer; pads with silence when the queue runs dry.
    /// Called from the device callback.
    pub(crate) fn fill(&self, out: &mut [f32]) {
        let mut st = self.lock();
        if st.queue.is_empty() && st.last_push.is_some_and(|t| t.elapsed() >= FLUSH_AFTER) {
            let st = &mut *st;
            if let Some((_, rs)) = st.stream.as_mut() {
                let mut tail = Vec::new();
                rs.flush(&mut tail);
                st.queue.extend(tail);
            }
            st.last_push = None;
        }
        let n = out.len().min(st.queue.len());
        for (o, s) in out.iter_mut().zip(st.queue.drain(..n)) {
            *o = s;
        }
        out[n..].fill(0.0);
        if n > 0 {
            st.last_audible = Some(Instant::now());
            drop(st);
            self.tap.push(&out[..n]);
        }
    }
}

impl AudioSink for PlaybackQueue {
    fn push(&self, clip: AudioClip) {
        let mut st = self.lock();
        let st = &mut *st;
        st.last_push = Some(Instant::now());
        if clip.sample_rate == self.device_rate {
            st.queue.extend(clip.samples);
            return;
        }
        if st
            .stream
            .as_ref()
            .is_none_or(|(rate, _)| *rate != clip.sample_rate)
        {
            match StreamResampler::new(clip.sample_rate, self.device_rate) {
                Ok(rs) => st.stream = Some((clip.sample_rate, rs)),
                Err(e) => {
                    tracing::warn!("dropping clip: {e}");
                    return;
                }
            }
        }
        let mut out = Vec::new();
        if let Some((_, rs)) = st.stream.as_mut() {
            rs.process(&clip.samples, &mut out);
        }
        st.queue.extend(out);
    }

    fn stop(&self) {
        let mut st = self.lock();
        st.queue.clear();
        st.last_audible = None;
        st.stream = None;
        st.last_push = None;
    }

    fn is_playing(&self) -> bool {
        let st = self.lock();
        !st.queue.is_empty() || st.last_audible.is_some_and(|t| t.elapsed() < TAIL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(rate: u32, n: usize) -> AudioClip {
        AudioClip {
            sample_rate: rate,
            samples: vec![0.25; n],
        }
    }

    #[test]
    fn clip_is_resampled_to_device_rate_and_drained_in_order() {
        let q = PlaybackQueue::new(48_000);
        q.push(clip(24_000, 2400)); // 100 ms -> 4800 samples
        let mut out = vec![1.0; 4000];
        q.fill(&mut out);
        assert!(out.iter().skip(500).all(|v| (*v - 0.25).abs() < 0.05));
        let mut rest = vec![1.0; 2000];
        q.fill(&mut rest);
        assert_eq!(rest[799], 0.25f32.min(rest[799])); // still playing the tail
        assert_eq!(rest[1999], 0.0, "padded with silence once drained");
    }

    #[test]
    fn streamed_chunks_join_without_clicks_and_the_tail_is_flushed() {
        let q = PlaybackQueue::new(48_000);
        let tone: Vec<f32> = (0..24_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24_000.0).sin() * 0.5)
            .collect();
        for chunk in tone.chunks(1200) {
            q.push(AudioClip {
                sample_rate: 24_000,
                samples: chunk.to_vec(),
            });
        }
        std::thread::sleep(Duration::from_millis(200));
        let mut all: Vec<f32> = Vec::new();
        loop {
            let mut buf = vec![0.0f32; 960];
            q.fill(&mut buf);
            if buf.iter().all(|v| *v == 0.0) {
                break;
            }
            all.extend(buf);
        }
        // About one second of audio came out, tail included.
        assert!(all.len() >= 47_000, "{}", all.len());
        // Away from the very start, no sample-to-sample jump beyond the tone's own slope.
        let max_step = all[400..46_000]
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_step < 0.045, "click of {max_step}");
    }

    #[test]
    fn stop_clears_the_queue_immediately() {
        let q = PlaybackQueue::new(16_000);
        q.push(clip(16_000, 16_000));
        assert!(q.is_playing());
        q.stop();
        assert!(!q.is_playing());
        let mut out = vec![1.0; 160];
        q.fill(&mut out);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn tap_receives_only_what_was_played() {
        let q = PlaybackQueue::new(16_000);
        q.push(clip(16_000, 1600));
        let mut out = vec![0.0; 160];
        q.fill(&mut out);
        q.stop(); // the remaining 1440 samples were never played
        q.fill(&mut out);
        assert_eq!(q.tap().drain().len(), 160);
    }

    #[test]
    fn playing_flag_lingers_briefly_after_the_queue_empties() {
        let q = PlaybackQueue::new(16_000);
        q.push(clip(16_000, 160));
        let mut out = vec![0.0; 160];
        q.fill(&mut out);
        assert!(q.is_playing());
    }
}
