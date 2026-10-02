//! Model-bound, fixed-capacity KV state; layout `[kv_head,position,head_dim]`.
use crate::LinearError;
use crate::attention::{AttentionConfig, AttentionWeights};
use crate::matrix::ProjectionError;
use crate::model::{Decoder, ModelConfig};
use crate::ops::{OpError, residual_add, rmsnorm, silu, softmax};
use crate::quantization::QuantError;
use crate::rope::{RopeError, rope};
use serde::Serialize;
use std::time::Instant;

/// Diagnostic scopes include operator validation/allocation and are excluded
/// from ordinary benchmark repetitions. Units are monotonic wall-clock ns.
#[derive(Default, Debug, Serialize)]
pub struct LayerProfile {
    pub attention_norm_ns: u128,
    pub qkv_ns: u128,
    pub rope_ns: u128,
    pub attention_mix_ns: u128,
    pub attention_output_ns: u128,
    pub residual_norm_ns: u128,
    pub gate_up_ns: u128,
    pub silu_product_ns: u128,
    pub down_residual_ns: u128,
    pub total_ns: u128,
}
#[derive(Default, Debug, Serialize)]
pub struct StepProfile {
    pub embedding_ns: u128,
    pub layers: Vec<LayerProfile>,
    pub final_norm_ns: u128,
    pub logits_ns: u128,
    pub total_ns: u128,
}
fn begin(enabled: bool) -> Option<Instant> {
    enabled.then(Instant::now)
}
fn elapsed(start: Option<Instant>) -> u128 {
    start.map_or(0, |s| s.elapsed().as_nanos())
}

struct LayerCache {
    key: Vec<f32>,
    value: Vec<f32>,
}
#[derive(Debug)]
pub enum CacheError {
    InvalidCapacity,
    DimensionOverflow,
    Allocation,
    ResourceBudget,
    InvalidToken,
    InvalidOutputLength,
    EmptyInput,
    NotEmpty,
    Full,
    InvalidState,
    NonFinite,
    Operation(OpError),
    Linear(LinearError),
    Quantized(QuantError),
    Rotary(RopeError),
}
impl From<ProjectionError> for CacheError {
    fn from(e: ProjectionError) -> Self {
        match e {
            ProjectionError::Float(e) => Self::Linear(e),
            ProjectionError::Quantized(e) => Self::Quantized(e),
        }
    }
}
fn finite(values: &[f32]) -> Result<(), CacheError> {
    if values.iter().any(|x| !x.is_finite()) {
        Err(CacheError::NonFinite)
    } else {
        Ok(())
    }
}
fn zeros(len: usize) -> Result<Vec<f32>, CacheError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| CacheError::Allocation)?;
    values.resize(len, 0.0);
    Ok(values)
}

struct AttentionScratch {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    rotated_query: Vec<f32>,
    rotated_key: Vec<f32>,
    context: Vec<f32>,
    output: Vec<f32>,
    scores: Vec<f32>,
    probabilities: Vec<f32>,
}
struct StepScratch {
    hidden: Vec<f32>,
    norm: Vec<f32>,
    residual: Vec<f32>,
    ffn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    activated: Vec<f32>,
    attention: AttentionScratch,
}
/// Resident reusable vector payload, excluding caller logits and allocator overhead.
pub(crate) fn workspace_bytes(c: ModelConfig, capacity: usize) -> Result<usize, CacheError> {
    c.hidden_dim
        .checked_mul(8)
        .and_then(|n| {
            c.kv_heads
                .checked_mul(c.head_dim)
                .and_then(|k| k.checked_mul(3))
                .and_then(|k| n.checked_add(k))
        })
        .and_then(|n| c.ff_dim.checked_mul(3).and_then(|f| n.checked_add(f)))
        .and_then(|n| capacity.checked_mul(2).and_then(|p| n.checked_add(p)))
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or(CacheError::DimensionOverflow)
}
impl StepScratch {
    fn new(c: ModelConfig, capacity: usize) -> Result<Self, CacheError> {
        let kv = c
            .kv_heads
            .checked_mul(c.head_dim)
            .ok_or(CacheError::DimensionOverflow)?;
        let h = c.hidden_dim;
        Ok(Self {
            hidden: zeros(h)?,
            norm: zeros(h)?,
            residual: zeros(h)?,
            ffn: zeros(h)?,
            gate: zeros(c.ff_dim)?,
            up: zeros(c.ff_dim)?,
            activated: zeros(c.ff_dim)?,
            attention: AttentionScratch {
                query: zeros(h)?,
                key: zeros(kv)?,
                value: zeros(kv)?,
                rotated_query: zeros(h)?,
                rotated_key: zeros(kv)?,
                context: zeros(h)?,
                output: zeros(h)?,
                scores: zeros(capacity)?,
                probabilities: zeros(capacity)?,
            },
        })
    }
}

/// The borrowed immutable model binds the cache to one weight identity, not
/// merely matching dimensions. No history/sampling is owned by this session.
pub struct CachedSession<'a> {
    model: &'a Decoder,
    layers: Vec<LayerCache>,
    capacity: usize,
    processed: usize,
    valid: bool,
    logical_bytes: usize,
    workspace_bytes: usize,
    scratch: StepScratch,
}
impl<'a> CachedSession<'a> {
    pub fn new(model: &'a Decoder, capacity: usize) -> Result<Self, CacheError> {
        let c = model.config();
        if capacity == 0 || capacity > c.max_tokens {
            return Err(CacheError::InvalidCapacity);
        }
        let per = c
            .kv_heads
            .checked_mul(capacity)
            .and_then(|n| n.checked_mul(c.head_dim))
            .ok_or(CacheError::DimensionOverflow)?;
        let logical_bytes = per
            .checked_mul(2)
            .and_then(|n| n.checked_mul(c.layers))
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or(CacheError::DimensionOverflow)?;
        // Bound cache plus reusable workspace before allocating either.
        let workspace_bytes = workspace_bytes(c, capacity)?;
        if logical_bytes
            .checked_add(workspace_bytes)
            .ok_or(CacheError::DimensionOverflow)?
            > 128 * 1024 * 1024
        {
            return Err(CacheError::ResourceBudget);
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(c.layers)
            .map_err(|_| CacheError::Allocation)?;
        for _ in 0..c.layers {
            layers.push(LayerCache {
                key: zeros(per)?,
                value: zeros(per)?,
            });
        }
        Ok(Self {
            model,
            layers,
            capacity,
            processed: 0,
            valid: true,
            logical_bytes,
            workspace_bytes,
            scratch: StepScratch::new(c, capacity)?,
        })
    }
    pub fn processed(&self) -> usize {
        self.processed
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub(crate) fn model_config(&self) -> crate::model::ModelConfig {
        self.model.config()
    }
    pub fn is_valid(&self) -> bool {
        self.valid
    }
    pub fn logical_bytes(&self) -> usize {
        self.logical_bytes
    }
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
    /// Stale bytes are unreachable after logical reset. Storage is reused.
    pub fn reset(&mut self) {
        self.processed = 0;
        self.valid = true;
    }
    /// Initial serial prefill; requires an empty, valid session. This is not
    /// optimized matrix-matrix prefill. Returns only the last position's logits.
    pub fn prefill(&mut self, tokens: &[usize]) -> Result<Vec<f32>, CacheError> {
        if !self.valid {
            return Err(CacheError::InvalidState);
        }
        if self.processed != 0 {
            return Err(CacheError::NotEmpty);
        }
        self.append(tokens)
    }
    /// Appends a chunk through serial one-token steps. All predictable input
    /// errors are checked before any token write. Execution failures invalidate
    /// the entire session, including any already processed part of this chunk.
    pub fn append(&mut self, tokens: &[usize]) -> Result<Vec<f32>, CacheError> {
        let mut logits = zeros(self.model.config().vocab_size)?;
        self.append_into(tokens, &mut logits)?;
        Ok(logits)
    }
    /// Same admission and failure rules as append, with caller-owned logits.
    /// Admission errors preserve output; execution errors may modify it and
    /// invalidate the cache. Only the final position's logits remain on success.
    pub fn append_into(&mut self, tokens: &[usize], logits: &mut [f32]) -> Result<(), CacheError> {
        if !self.valid {
            return Err(CacheError::InvalidState);
        }
        if tokens.is_empty() {
            return Err(CacheError::EmptyInput);
        }
        if logits.len() != self.model.config().vocab_size {
            return Err(CacheError::InvalidOutputLength);
        }
        if tokens
            .iter()
            .any(|&id| id >= self.model.config().vocab_size)
        {
            return Err(CacheError::InvalidToken);
        }
        if self
            .processed
            .checked_add(tokens.len())
            .is_none_or(|n| n > self.capacity)
        {
            return Err(CacheError::Full);
        }
        for &token in tokens {
            self.step_into(token, logits)?;
        }
        Ok(())
    }
    /// Initial serial prefill into exact-width caller output; requires empty state.
    pub fn prefill_into(&mut self, tokens: &[usize], logits: &mut [f32]) -> Result<(), CacheError> {
        if !self.valid {
            return Err(CacheError::InvalidState);
        }
        if self.processed != 0 {
            return Err(CacheError::NotEmpty);
        }
        self.append_into(tokens, logits)
    }
    /// Appends exactly one evaluated token at the next absolute position.
    /// Predictable admission errors leave state unchanged. Any execution error
    /// invalidates all layer prefixes; reset is required before another step.
    pub fn step(&mut self, token: usize) -> Result<Vec<f32>, CacheError> {
        let mut logits = zeros(self.model.config().vocab_size)?;
        self.step_into(token, &mut logits)?;
        Ok(logits)
    }
    /// Evaluates one token without allocating: output must have vocabulary width.
    /// Admission errors preserve output; execution errors may modify output and
    /// invalidate every cached prefix. The caller must discard that output.
    pub fn step_into(&mut self, token: usize, logits: &mut [f32]) -> Result<(), CacheError> {
        self.step_inner(token, logits, None)
    }
    pub fn step_profiled(&mut self, token: usize) -> Result<(Vec<f32>, StepProfile), CacheError> {
        let start = Instant::now();
        let mut profile = StepProfile::default();
        let mut logits = zeros(self.model.config().vocab_size)?;
        self.step_inner(token, &mut logits, Some(&mut profile))?;
        profile.total_ns = start.elapsed().as_nanos();
        Ok((logits, profile))
    }
    fn step_inner(
        &mut self,
        token: usize,
        logits: &mut [f32],
        profile: Option<&mut StepProfile>,
    ) -> Result<(), CacheError> {
        if !self.valid {
            return Err(CacheError::InvalidState);
        }
        if logits.len() != self.model.config().vocab_size {
            return Err(CacheError::InvalidOutputLength);
        }
        if token >= self.model.config().vocab_size {
            return Err(CacheError::InvalidToken);
        }
        if self.processed == self.capacity {
            return Err(CacheError::Full);
        }
        match self.forward(token, logits, profile) {
            Ok(()) => {
                self.processed += 1;
                Ok(())
            }
            Err(error) => {
                self.valid = false;
                self.processed = 0;
                Err(error)
            }
        }
    }
    fn forward(
        &mut self,
        token: usize,
        logits: &mut [f32],
        mut profile: Option<&mut StepProfile>,
    ) -> Result<(), CacheError> {
        let diagnostic = profile.is_some();
        let c = self.model.config();
        let h = c.hidden_dim;
        let weights = &self.model.weights;
        let w = &mut self.scratch;
        let clock = begin(diagnostic);
        weights
            .embedding
            .view()
            .gather(&[token], &mut w.hidden, c.vocab_size, h)
            .map_err(CacheError::Operation)?;
        if let Some(p) = profile.as_deref_mut() {
            p.embedding_ns = elapsed(clock);
        }
        for (layer, cache) in weights.layers.iter().zip(&mut self.layers) {
            let total = begin(diagnostic);
            let mut detail = LayerProfile::default();
            let clock = begin(diagnostic);
            rmsnorm(
                &w.hidden,
                &layer.attention_norm,
                &mut w.norm,
                1,
                h,
                c.norm_epsilon,
            )
            .map_err(CacheError::Operation)?;
            detail.attention_norm_ns = elapsed(clock);
            attend_one(
                &w.norm,
                AttentionWeights {
                    query: (&layer.query).into(),
                    key: (&layer.key).into(),
                    value: (&layer.value).into(),
                    output: (&layer.attention_output).into(),
                    query_bias: layer.query_bias.as_deref(),
                    key_bias: layer.key_bias.as_deref(),
                    value_bias: layer.value_bias.as_deref(),
                },
                AttentionConfig {
                    hidden_dim: h,
                    query_heads: c.query_heads,
                    kv_heads: c.kv_heads,
                    head_dim: c.head_dim,
                    rope_base: c.rope_base,
                },
                cache,
                (self.processed, self.capacity),
                &mut w.attention,
                diagnostic.then_some(&mut detail),
            )?;
            let clock = begin(diagnostic);
            residual_add(&w.hidden, &w.attention.output, &mut w.residual)
                .map_err(CacheError::Operation)?;
            rmsnorm(
                &w.residual,
                &layer.ffn_norm,
                &mut w.norm,
                1,
                h,
                c.norm_epsilon,
            )
            .map_err(CacheError::Operation)?;
            detail.residual_norm_ns = elapsed(clock);
            let clock = begin(diagnostic);
            layer
                .gate
                .view()
                .project(&w.norm, &mut w.gate, 1, h, c.ff_dim, false)
                .map_err(CacheError::from)?;
            layer
                .up
                .view()
                .project(&w.norm, &mut w.up, 1, h, c.ff_dim, false)
                .map_err(CacheError::from)?;
            detail.gate_up_ns = elapsed(clock);
            let clock = begin(diagnostic);
            silu(&w.gate, &mut w.activated).map_err(CacheError::Operation)?;
            for (product, &u) in w.activated.iter_mut().zip(&w.up) {
                *product *= u;
            }
            finite(&w.activated)?;
            detail.silu_product_ns = elapsed(clock);
            let clock = begin(diagnostic);
            layer
                .down
                .view()
                .project(&w.activated, &mut w.ffn, 1, c.ff_dim, h, false)
                .map_err(CacheError::from)?;
            residual_add(&w.residual, &w.ffn, &mut w.hidden).map_err(CacheError::Operation)?;
            finite(&w.hidden)?;
            detail.down_residual_ns = elapsed(clock);
            detail.total_ns = elapsed(total);
            if let Some(p) = profile.as_deref_mut() {
                p.layers.push(detail);
            }
        }
        let clock = begin(diagnostic);
        rmsnorm(
            &w.hidden,
            &weights.final_norm,
            &mut w.norm,
            1,
            h,
            c.norm_epsilon,
        )
        .map_err(CacheError::Operation)?;
        if let Some(p) = profile.as_deref_mut() {
            p.final_norm_ns = elapsed(clock);
        }
        let clock = begin(diagnostic);
        weights
            .embedding
            .view()
            .project(&w.norm, logits, 1, h, c.vocab_size, true)
            .map_err(CacheError::from)?;
        finite(logits)?;
        if let Some(p) = profile {
            p.logits_ns = elapsed(clock);
        }
        Ok(())
    }
}

// All dimensions/weight buffers are validated by the bound Decoder; cache
// dimensions are validated at construction. This helper never exposes state.
fn attend_one(
    x: &[f32],
    weights: AttentionWeights<'_>,
    c: AttentionConfig,
    cache: &mut LayerCache,
    position_capacity: (usize, usize),
    workspace: &mut AttentionScratch,
    mut profile: Option<&mut LayerProfile>,
) -> Result<(), CacheError> {
    let (position, capacity) = position_capacity;
    let AttentionScratch {
        query,
        key,
        value,
        rotated_query,
        rotated_key,
        context,
        output,
        scores,
        probabilities,
    } = workspace;
    let diagnostic = profile.is_some();
    let clock = begin(diagnostic);
    let h = c.hidden_dim;
    let d = c.head_dim;
    let kv = c.kv_heads * d;
    weights
        .query
        .project(x, query, 1, h, h, false)
        .map_err(CacheError::from)?;
    weights
        .key
        .project(x, key, 1, h, kv, false)
        .map_err(CacheError::from)?;
    weights
        .value
        .project(x, value, 1, h, kv, false)
        .map_err(CacheError::from)?;
    for (values, bias) in [
        (&mut *query, weights.query_bias),
        (&mut *key, weights.key_bias),
        (&mut *value, weights.value_bias),
    ] {
        if let Some(bias) = bias {
            for (v, &b) in values.iter_mut().zip(bias) {
                *v += b;
            }
        }
        finite(values)?;
    }
    if let Some(p) = profile.as_deref_mut() {
        p.qkv_ns = elapsed(clock);
    }
    let clock = begin(diagnostic);
    rope(
        query,
        rotated_query,
        1,
        c.query_heads,
        d,
        position,
        c.rope_base,
    )
    .map_err(CacheError::Rotary)?;
    rope(key, rotated_key, 1, c.kv_heads, d, position, c.rope_base).map_err(CacheError::Rotary)?;
    finite(rotated_query)?;
    finite(rotated_key)?;
    if let Some(p) = profile.as_deref_mut() {
        p.rope_ns = elapsed(clock);
    }
    let clock = begin(diagnostic);
    for head in 0..c.kv_heads {
        let offset = (head * capacity + position) * d;
        cache.key[offset..offset + d].copy_from_slice(&rotated_key[head * d..(head + 1) * d]);
        cache.value[offset..offset + d].copy_from_slice(&value[head * d..(head + 1) * d]);
    }
    context.fill(0.0);
    let scores = &mut scores[..=position];
    let probabilities = &mut probabilities[..=position];
    for qh in 0..c.query_heads {
        let kvh = qh / (c.query_heads / c.kv_heads);
        for (p, score) in scores.iter_mut().enumerate() {
            let offset = (kvh * capacity + p) * d;
            let mut sum = 0.0_f32;
            for i in 0..d {
                sum += rotated_query[qh * d + i] * cache.key[offset + i];
            }
            *score = sum / (d as f32).sqrt();
        }
        softmax(scores, probabilities, 1, position + 1).map_err(CacheError::Operation)?;
        for (p, &probability) in probabilities.iter().enumerate() {
            let offset = (kvh * capacity + p) * d;
            for i in 0..d {
                context[qh * d + i] += probability * cache.value[offset + i];
            }
        }
    }
    finite(context)?;
    if let Some(p) = profile.as_deref_mut() {
        p.attention_mix_ns = elapsed(clock);
    }
    let clock = begin(diagnostic);
    weights
        .output
        .project(context, output, 1, h, h, false)
        .map_err(CacheError::from)?;
    finite(output)?;
    if let Some(p) = profile {
        p.attention_output_ns = elapsed(clock);
    }
    Ok(())
}
