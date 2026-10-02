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
