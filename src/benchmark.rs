//! Explicit benchmark data and scopes; never used by ordinary generation.
use crate::kv_cache::CachedSession;
use crate::model::ModelConfig;
use crate::sampling::{Sampler, Sampling};
use crate::tokenizer::ChatTokenizer;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub name: String,
    pub prompt: Vec<usize>,
    pub fixed_output: Vec<usize>,
    pub continuation: Vec<usize>,
    pub capacity: usize,
    pub warmups: usize,
    pub repetitions: usize,
}
impl Workload {
    pub fn validate(&self, config: ModelConfig) -> Result<(), String> {
        if self.prompt.is_empty()
            || self.fixed_output.is_empty()
            || self.continuation.is_empty()
            || self.capacity == 0
            || self.capacity > config.max_tokens
            || self.warmups > 10
            || self.repetitions == 0
            || self.repetitions > 100
        {
            return Err("invalid workload dimensions/repetition bounds".into());
        }
        let total = self
            .prompt
            .len()
            .checked_add(self.fixed_output.len())
            .and_then(|n| n.checked_add(self.continuation.len()))
            .and_then(|n| n.checked_add(1))
            .ok_or("context arithmetic overflow")?;
        if total > self.capacity {
            return Err("whole workload including continuation exceeds admitted capacity".into());
        }
        if self
            .prompt
            .iter()
            .chain(&self.fixed_output)
            .chain(&self.continuation)
            .any(|&id| id >= config.vocab_size)
        {
            return Err("invalid workload token".into());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, Serialize)]
pub enum Mode {
    TeacherForced,
    Greedy,
}
#[derive(Debug, Serialize)]
pub struct Sample {
    pub mode: Mode,
    pub emitted: Vec<usize>,
    pub stopped_by_eos: bool,
    pub prefill_ns: u128,
    pub backend_decode_ns: Vec<u128>,
    pub sampling_ns: Vec<u128>,
    pub text_ns: Vec<u128>,
    pub ttft_ns: Option<u128>,
    pub first_logits_ready_ns: u128,
    pub first_visible_text_ns: Option<u128>,
    pub token_times_ns: Vec<u128>,
    pub decode_tokens_per_second: Option<f64>,
    pub backend_decode_tokens_per_second: Option<f64>,
    pub continuation_backend_ns: u128,
    pub continuation_next_token_ns: u128,
    pub continuation_emitted_token: usize,
    pub pending_after_continuation: usize,
    pub processed_before_continuation: usize,
    pub pending_emitted_tokens: usize,
    pub processed_after_continuation: usize,
}
/// Requested token count N; exactly N-1 decode evaluations after first logits.
/// No fabricated throughput for a single token or a zero duration.
pub fn throughput(tokens: usize, elapsed_ns: u128) -> Option<f64> {
    if tokens < 2 || elapsed_ns == 0 {
        None
    } else {
        Some((tokens - 1) as f64 * 1e9 / elapsed_ns as f64)
    }
}
/// Measures one warm, cache-reset request. Tokenization/model loading/cache
/// construction are outside these scopes; tracing stays out of this path.
pub fn run_sample(
    session: &mut CachedSession<'_>,
    workload: &Workload,
    mode: Mode,
    tokenizer: Option<&ChatTokenizer>,
) -> Result<Sample, String> {
    workload.validate(session.model_config())?;
    if workload.capacity != session.capacity()
        || workload.prompt.is_empty()
        || workload.fixed_output.is_empty()
        || workload.continuation.is_empty()
    {
        return Err("invalid admitted benchmark request".into());
    }
    let mut sampler = Sampler::new(Sampling::Greedy).map_err(|e| format!("{e:?}"))?;
    let mut stream = tokenizer.map(ChatTokenizer::text_stream);
    let mut emitted = Vec::with_capacity(workload.fixed_output.len());
    let mut backend_decode_ns = Vec::with_capacity(workload.fixed_output.len() - 1);
    let mut sampling_ns = Vec::with_capacity(workload.fixed_output.len());
    let mut text_ns = Vec::with_capacity(workload.fixed_output.len());
    let mut token_times_ns = Vec::with_capacity(workload.fixed_output.len());
    let mut first_visible_text_ns = None;
    let mut stopped_by_eos = false;
    session.reset();
    let mut logits = vec![0.0; session.model_config().vocab_size];
    let request = Instant::now();
    let clock = Instant::now();
    session
        .prefill_into(&workload.prompt, &mut logits)
        .map_err(|e| format!("{e:?}"))?;
    let first_logits_ready_ns = request.elapsed().as_nanos();
    let prefill_ns = clock.elapsed().as_nanos();
    for &forced in &workload.fixed_output {
        if let Some(&previous) = emitted.last() {
            let clock = Instant::now();
            session
                .step_into(previous, &mut logits)
                .map_err(|e| format!("{e:?}"))?;
            backend_decode_ns.push(clock.elapsed().as_nanos());
        }
        let clock = Instant::now();
        let id = match mode {
            Mode::TeacherForced => forced,
            Mode::Greedy => sampler.sample(&logits).map_err(|e| format!("{e:?}"))?,
        };
        sampling_ns.push(if matches!(mode, Mode::Greedy) {
            clock.elapsed().as_nanos()
        } else {
            0
        });
        emitted.push(id);
        token_times_ns.push(request.elapsed().as_nanos());
        let clock = Instant::now();
        if let Some(stream) = &mut stream
            && stream.push(id).map_err(|e| format!("{e:?}"))?.is_some()
            && first_visible_text_ns.is_none()
        {
            first_visible_text_ns = Some(request.elapsed().as_nanos());
        }
        text_ns.push(if tokenizer.is_some() {
            clock.elapsed().as_nanos()
        } else {
            0
        });
        if matches!(mode, Mode::Greedy) && tokenizer.is_some_and(|t| t.stop_tokens().contains(&id))
        {
            stopped_by_eos = true;
            break;
        }
    }
    let ttft_ns = matches!(mode, Mode::Greedy).then_some(token_times_ns[0]);
    let interval = token_times_ns.last().unwrap() - token_times_ns[0];
    let backend_interval = backend_decode_ns.iter().sum();
    let processed_before_continuation = session.processed();
    let pending_emitted_tokens = workload.prompt.len() + emitted.len() - session.processed();
    if pending_emitted_tokens != 1 {
        return Err("benchmark history/cache invariant violated".into());
    }
    let mut append = Vec::with_capacity(1 + workload.continuation.len());
    append.push(*emitted.last().unwrap());
    append.extend_from_slice(&workload.continuation);
    let clock = Instant::now();
    session
        .append_into(&append, &mut logits)
        .map_err(|e| format!("{e:?}"))?;
    let continuation_backend_ns = clock.elapsed().as_nanos();
    let next = sampler.sample(&logits).map_err(|e| format!("{e:?}"))?;
    std::hint::black_box(next);
    let continuation_next_token_ns = clock.elapsed().as_nanos();
    std::hint::black_box(logits);
    Ok(Sample {
        mode,
        decode_tokens_per_second: throughput(emitted.len(), interval),
        backend_decode_tokens_per_second: throughput(emitted.len(), backend_interval),
        emitted,
        stopped_by_eos,
        prefill_ns,
        backend_decode_ns,
        sampling_ns,
        text_ns,
        ttft_ns,
        first_logits_ready_ns,
        first_visible_text_ns,
        token_times_ns,
        continuation_backend_ns,
        continuation_next_token_ns,
        continuation_emitted_token: next,
        pending_after_continuation: 1,
        processed_before_continuation,
        pending_emitted_tokens,
        processed_after_continuation: session.processed(),
    })
}
#[derive(Debug, Serialize)]
pub struct Distribution {
    pub samples: usize,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub median_absolute_deviation: f64,
}
pub fn distribution(values: &[f64]) -> Result<Distribution, String> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err("finite nonempty observations required".into());
    }
    fn median(sorted: &[f64]) -> f64 {
        let middle = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            sorted[middle - 1] * 0.5 + sorted[middle] * 0.5
        } else {
            sorted[middle]
        }
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let center = median(&sorted);
    let mut deviations: Vec<_> = sorted.iter().map(|v| (v - center).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    Ok(Distribution {
        samples: values.len(),
        median: center,
        min: sorted[0],
        max: *sorted.last().unwrap(),
        median_absolute_deviation: median(&deviations),
    })
}
/// Reusable workspace plus up to two caller-owned vocabulary rows (legacy
/// append can overlap the outer row). Allocator/tokenizer/OS overhead is separate.
pub fn scratch_estimate(config: ModelConfig, capacity: usize) -> Result<usize, String> {
    let workspace =
        crate::kv_cache::workspace_bytes(config, capacity).map_err(|e| format!("{e:?}"))?;
    config
        .vocab_size
        .checked_mul(2)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .and_then(|n| n.checked_add(workspace))
        .ok_or("overflow".into())
}
