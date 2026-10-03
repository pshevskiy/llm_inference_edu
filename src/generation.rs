//! Bounded uncached greedy baseline. Each step recomputes the full prefix.
use crate::model::{Decoder, ModelError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    StopToken,
    OutputLimit,
}
#[derive(Debug)]
pub enum GenerationError {
    EmptyPrompt,
    InvalidToken(usize),
    OutputLimit,
    ContextLimit,
    Model(ModelError),
}
pub struct Completion {
    /// Full history, including the final emitted token (not necessarily evaluated).
    pub token_ids: Vec<usize>,
    pub generated_ids: Vec<usize>,
    pub stop: StopReason,
}

/// Greedy ties choose the lowest ID. Context includes prompt plus generated
/// tokens; requests that cannot fit are rejected, never silently truncated.
/// Stop tokens are appended to history but are not evaluated after stopping.
pub fn uncached_greedy(
    model: &Decoder,
    prompt: &[usize],
    stop_tokens: &[usize],
    max_new: usize,
) -> Result<Completion, GenerationError> {
    let c = model.config();
    if prompt.is_empty() {
        return Err(GenerationError::EmptyPrompt);
    }
    if max_new == 0 {
        return Err(GenerationError::OutputLimit);
    }
    for &id in prompt.iter().chain(stop_tokens) {
        if id >= c.vocab_size {
            return Err(GenerationError::InvalidToken(id));
        }
    }
    if prompt
        .len()
        .checked_add(max_new)
        .is_none_or(|n| n > c.max_tokens)
    {
        return Err(GenerationError::ContextLimit);
    }
    let mut history = Vec::with_capacity(prompt.len() + max_new);
    history.extend_from_slice(prompt);
    let mut generated_ids = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let output = model
            .forward(&history, false)
            .map_err(GenerationError::Model)?;
        // Forward already validates finiteness and the exact vocabulary width.
        let mut next = 0;
        for i in 1..output.logits.len() {
            if output.logits[i] > output.logits[next] {
                next = i;
            }
        }
        history.push(next);
        generated_ids.push(next);
        if stop_tokens.contains(&next) {
            return Ok(Completion {
                token_ids: history,
                generated_ids,
                stop: StopReason::StopToken,
            });
        }
    }
    Ok(Completion {
        token_ids: history,
        generated_ids,
        stop: StopReason::OutputLimit,
    })
}

use crate::kv_cache::{CacheError, CachedSession};
use crate::sampling::{Sampler, Sampling, SamplingError};
use std::sync::atomic::{AtomicBool, Ordering};
#[derive(Debug)]
pub enum StateError {
    InvalidParameters,
    NotStarted,
    AlreadyStarted,
    Cancelled,
    Cache(CacheError),
    Sampling(SamplingError),
}

/// History owns every admitted/emitted ID. Cache owns only the evaluated prefix.
/// At a successful emission exactly one history token is pending evaluation.
pub struct GenerationSession<'a> {
    cache: CachedSession<'a>,
    sampler: Sampler,
    history: Vec<usize>,
    logits: Vec<f32>,
    logits_ready: bool,
    stops: Vec<usize>,
    max_new: usize,
    generated: usize,
    stop: Option<StopReason>,
    vocab: usize,
}
impl<'a> GenerationSession<'a> {
    pub fn new(
        model: &'a Decoder,
        capacity: usize,
        policy: Sampling,
        stops: &[usize],
        max_new: usize,
    ) -> Result<Self, StateError> {
        let vocab = model.config().vocab_size;
        if max_new == 0 || max_new >= capacity || stops.iter().any(|&id| id >= vocab) {
            return Err(StateError::InvalidParameters);
        }
        let sampler = Sampler::new(policy).map_err(StateError::Sampling)?;
        sampler
            .validate_vocab(vocab)
            .map_err(StateError::Sampling)?;
        Ok(Self {
            cache: CachedSession::new(model, capacity).map_err(StateError::Cache)?,
            sampler,
            history: Vec::with_capacity(capacity),
            logits: vec![0.0; vocab],
            logits_ready: false,
            stops: stops.to_vec(),
            max_new,
            generated: 0,
            stop: None,
            vocab,
        })
    }
    pub fn start(&mut self, prompt: &[usize]) -> Result<(), StateError> {
        self.start_with_cancel(prompt, &AtomicBool::new(false))
    }
    /// Admits the whole prompt before evaluation. Cancellation during serial
    /// prefill retains its exact history and valid processed prefix; next_token
    /// resumes the remaining prompt before sampling anything.
    pub fn start_with_cancel(
        &mut self,
        prompt: &[usize],
        cancel: &AtomicBool,
    ) -> Result<(), StateError> {
        if !self.history.is_empty() {
            return Err(StateError::AlreadyStarted);
        }
        if prompt.is_empty()
            || prompt.iter().any(|&id| id >= self.vocab)
            || prompt
                .len()
                .checked_add(self.max_new)
                .is_none_or(|n| n > self.cache.capacity())
        {
            return Err(StateError::InvalidParameters);
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(StateError::Cancelled);
        }
        if !self.cache.is_valid() {
            return Err(StateError::Cache(CacheError::InvalidState));
        }
        self.history.extend_from_slice(prompt);
        self.evaluate_pending(cancel)
    }
    /// Admit a complete external context suffix at a stopped generation boundary.
    /// The emitted pending token is already in history and is evaluated exactly
    /// once before the new suffix; callers must not append it a second time.
    /// Sampling state is retained; output budget restarts for the next segment.
    /// On pre-admission error/cancellation all state stays unchanged. Cancellation
    /// after admission preserves the entire new history and exact valid prefix.
    pub fn append_context(
        &mut self,
        suffix: &[usize],
        max_new: usize,
        cancel: &AtomicBool,
    ) -> Result<(), StateError> {
        self.admit_context(suffix, max_new, cancel)?;
        self.evaluate_pending(cancel)
    }
    fn admit_context(
        &mut self,
        suffix: &[usize],
        max_new: usize,
        cancel: &AtomicBool,
    ) -> Result<(), StateError> {
        if self.history.is_empty()
            || self.stop.is_none()
            || suffix.is_empty()
            || max_new == 0
            || max_new >= self.cache.capacity()
            || suffix.iter().any(|&id| id >= self.vocab)
            || self
                .history
                .len()
                .checked_add(suffix.len())
                .and_then(|n| n.checked_add(max_new))
                .is_none_or(|n| n > self.cache.capacity())
        {
            return Err(StateError::InvalidParameters);
        }
        if !self.cache.is_valid() {
            return Err(StateError::Cache(CacheError::InvalidState));
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(StateError::Cancelled);
        }
        self.history.extend_from_slice(suffix);
        self.max_new = max_new;
        self.generated = 0;
        self.stop = None;
        Ok(())
    }
    /// Editing any existing history invalidates its KV prefix. Validate first,
    /// then reset/replay from zero; no cache prefix survives an accepted edit.
    /// A cancelled partial replay retains the replacement history for resume.
    pub fn replay_history(
        &mut self,
        history: &[usize],
        max_new: usize,
        cancel: &AtomicBool,
    ) -> Result<(), StateError> {
        self.admit_replay(history, max_new, cancel)?;
        self.evaluate_pending(cancel)
    }
    fn admit_replay(
        &mut self,
        history: &[usize],
        max_new: usize,
        cancel: &AtomicBool,
    ) -> Result<(), StateError> {
        if history.is_empty()
            || max_new == 0
            || max_new >= self.cache.capacity()
            || history.iter().any(|&id| id >= self.vocab)
            || history
                .len()
                .checked_add(max_new)
                .is_none_or(|n| n > self.cache.capacity())
        {
            return Err(StateError::InvalidParameters);
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(StateError::Cancelled);
        }
        self.reset();
        self.max_new = max_new;
        // No second pre-admission cancellation check after reset: replacement
        // history must survive a cancellation arriving after accepted editing.
        self.history.extend_from_slice(history);
        Ok(())
    }
    fn evaluate_pending(&mut self, cancel: &AtomicBool) -> Result<(), StateError> {
        while self.cache.processed() < self.history.len() {
            if cancel.load(Ordering::Relaxed) {
                return Err(StateError::Cancelled);
            }
            let token = self.history[self.cache.processed()];
            self.logits_ready = false;
            self.cache
                .step_into(token, &mut self.logits)
                .map_err(StateError::Cache)?;
            self.logits_ready = true;
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(StateError::Cancelled);
        }
        Ok(())
    }
    pub fn history(&self) -> &[usize] {
        &self.history
    }
    pub fn processed(&self) -> usize {
        self.cache.processed()
    }
    pub fn is_valid(&self) -> bool {
        self.cache.is_valid()
    }
    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop
    }
    pub fn generated(&self) -> usize {
        self.generated
    }
    pub fn reset(&mut self) {
        self.cache.reset();
        self.history.clear();
        self.logits_ready = false;
        self.generated = 0;
        self.stop = None;
        self.sampler.reset();
    }
    /// Cancellation is cooperative at token boundaries (before stepping and
    /// again before sampling). It preserves history, seed stream and valid KV.
    /// A token step already in progress finishes; no partially valid cache.
    pub fn next_token(&mut self, cancel: &AtomicBool) -> Result<Option<usize>, StateError> {
        if cancel.load(Ordering::Relaxed) {
            return Err(StateError::Cancelled);
        }
        if !self.cache.is_valid() {
            return Err(StateError::Cache(CacheError::InvalidState));
        }
        if self.history.is_empty() {
            return Err(StateError::NotStarted);
        }
        if self.stop.is_some() {
            return Ok(None);
        }
        self.evaluate_pending(cancel)?;
        if !self.logits_ready {
            return Err(StateError::NotStarted);
        }
        let next = self
            .sampler
            .sample(&self.logits)
            .map_err(StateError::Sampling)?;
        self.history.push(next);
        self.generated += 1;
        self.logits_ready = false;
        if self.stops.contains(&next) {
            self.stop = Some(StopReason::StopToken);
        } else if self.generated == self.max_new {
            self.stop = Some(StopReason::OutputLimit);
        }
        Ok(Some(next))
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;
    #[test]
    fn interrupted_prefill_resumes_missing_prompt_tokens_in_order() {
        let model = Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap();
        let mut state = GenerationSession::new(&model, 32, Sampling::Greedy, &[], 1).unwrap();
        // Exact reachable state at a cancellation boundary after first token:
        // the entire prompt is admitted, only its first token has been evaluated.
        state.history.extend_from_slice(&[1, 5, 7]);
        state.cache.step_into(1, &mut state.logits).unwrap();
        state.logits_ready = true;
        let cancel = AtomicBool::new(true);
        assert!(matches!(
            state.next_token(&cancel),
            Err(StateError::Cancelled)
        ));
        assert_eq!(state.processed(), 1);
        assert_eq!(state.history(), [1, 5, 7]);
        cancel.store(false, Ordering::Relaxed);
        assert_eq!(state.next_token(&cancel).unwrap(), Some(41));
        assert_eq!(state.processed(), 3);
        assert_eq!(state.history(), [1, 5, 7, 41]);
    }
    #[test]
    fn continued_and_edited_logits_match_independent_numpy() {
        let model = Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap();
        let mut state = GenerationSession::new(&model, 32, Sampling::Greedy, &[], 1).unwrap();
        let cancel = AtomicBool::new(false);
        state.start(&[1, 5, 7]).unwrap();
        state.next_token(&cancel).unwrap();
        state.append_context(&[2, 3], 1, &cancel).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/t25-continuation.json")).unwrap();
        for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
            if i == 1 {
                let ids: Vec<usize> = case["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();
                state.replay_history(&ids, 1, &cancel).unwrap();
            }
            for (&actual, expected) in state.logits.iter().zip(case["logits"].as_array().unwrap()) {
                let expected = expected.as_f64().unwrap() as f32;
                assert!(actual.is_finite());
                assert!(
                    (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                    "{i}: {actual} vs {expected}"
                );
            }
            assert_eq!(state.processed(), 6);
        }
    }
    #[test]
    fn cancellation_after_context_or_edit_admission_retains_exact_history() {
        let model = Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap();
        let mut state = GenerationSession::new(&model, 32, Sampling::Greedy, &[], 1).unwrap();
        let cancel = AtomicBool::new(false);
        state.start(&[1, 5, 7]).unwrap();
        state.next_token(&cancel).unwrap();
        state.admit_context(&[2, 3], 1, &cancel).unwrap();
        // Deterministic reachable boundary after the formerly pending 41 step.
        state.cache.step_into(41, &mut state.logits).unwrap();
        state.logits_ready = true;
        cancel.store(true, Ordering::Relaxed);
        assert!(matches!(
            state.evaluate_pending(&cancel),
            Err(StateError::Cancelled)
        ));
        assert_eq!(state.history(), [1, 5, 7, 41, 2, 3]);
        assert_eq!(state.processed(), 4);
        cancel.store(false, Ordering::Relaxed);
        assert_eq!(state.next_token(&cancel).unwrap(), Some(14));
        state
            .admit_replay(&[2, 5, 7, 41, 2, 3], 1, &cancel)
            .unwrap();
        cancel.store(true, Ordering::Relaxed);
        assert!(matches!(
            state.evaluate_pending(&cancel),
            Err(StateError::Cancelled)
        ));
        assert_eq!(state.history(), [2, 5, 7, 41, 2, 3]);
        assert_eq!(state.processed(), 0);
        cancel.store(false, Ordering::Relaxed);
        state.next_token(&cancel).unwrap();
        assert_eq!(state.processed(), 6);
    }
}
