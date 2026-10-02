//! Model-bound, fixed-capacity KV state; layout `[kv_head,position,head_dim]`.
use crate::attention::{AttentionConfig, AttentionWeights};
use crate::model::Decoder;
use crate::ops::{OpError, embedding, residual_add, rmsnorm, silu, softmax};
use crate::rope::{RopeError, rope};
use crate::{LinearError, linear};

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
    EmptyInput,
    NotEmpty,
    Full,
    InvalidState,
    NonFinite,
    Operation(OpError),
    Linear(LinearError),
    Rotary(RopeError),
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

/// The borrowed immutable model binds the cache to one weight identity, not
/// merely matching dimensions. No history/sampling is owned by this session.
pub struct CachedSession<'a> {
    model: &'a Decoder,
    layers: Vec<LayerCache>,
    capacity: usize,
    processed: usize,
    valid: bool,
    logical_bytes: usize,
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
        // Explicit initial cache-only admission; weights/scratch are separate.
        if logical_bytes > 128 * 1024 * 1024 {
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
        })
    }
    pub fn processed(&self) -> usize {
        self.processed
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn is_valid(&self) -> bool {
        self.valid
    }
    pub fn logical_bytes(&self) -> usize {
        self.logical_bytes
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
        if !self.valid {
            return Err(CacheError::InvalidState);
        }
        if tokens.is_empty() {
            return Err(CacheError::EmptyInput);
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
        let mut last = Vec::new();
        for &token in tokens {
            last = self.step(token)?;
        }
        Ok(last)
    }
    /// Appends exactly one evaluated token at the next absolute position.
    /// Predictable admission errors leave state unchanged. Any execution error
    /// invalidates all layer prefixes; reset is required before another step.
    pub fn step(&mut self, token: usize) -> Result<Vec<f32>, CacheError> {
        if !self.valid {
            return Err(CacheError::InvalidState);
        }
        if token >= self.model.config().vocab_size {
            return Err(CacheError::InvalidToken);
        }
        if self.processed == self.capacity {
            return Err(CacheError::Full);
        }
        match self.forward(token) {
            Ok(logits) => {
                self.processed += 1;
                Ok(logits)
            }
            Err(error) => {
                self.valid = false;
                self.processed = 0;
                Err(error)
            }
        }
    }
    fn forward(&mut self, token: usize) -> Result<Vec<f32>, CacheError> {
        let c = self.model.config();
        let h = c.hidden_dim;
        let weights = &self.model.weights;
        let mut hidden = vec![0.0; h];
        embedding(&[token], &weights.embedding, &mut hidden, c.vocab_size, h)
            .map_err(CacheError::Operation)?;
        for (layer, cache) in weights.layers.iter().zip(&mut self.layers) {
            let mut norm = vec![0.0; h];
            rmsnorm(
                &hidden,
                &layer.attention_norm,
                &mut norm,
                1,
                h,
                c.norm_epsilon,
            )
            .map_err(CacheError::Operation)?;
            let attention = attend_one(
                &norm,
                AttentionWeights {
                    query: &layer.query,
                    key: &layer.key,
                    value: &layer.value,
                    output: &layer.attention_output,
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
                self.processed,
                self.capacity,
            )?;
            let mut residual = vec![0.0; h];
            residual_add(&hidden, &attention, &mut residual).map_err(CacheError::Operation)?;
            rmsnorm(&residual, &layer.ffn_norm, &mut norm, 1, h, c.norm_epsilon)
                .map_err(CacheError::Operation)?;
            let mut gate = vec![0.0; c.ff_dim];
            let mut up = vec![0.0; c.ff_dim];
            linear(&norm, &layer.gate, &mut gate, 1, h, c.ff_dim).map_err(CacheError::Linear)?;
            linear(&norm, &layer.up, &mut up, 1, h, c.ff_dim).map_err(CacheError::Linear)?;
            let mut activated = vec![0.0; c.ff_dim];
            silu(&gate, &mut activated).map_err(CacheError::Operation)?;
            for (product, &u) in activated.iter_mut().zip(&up) {
                *product *= u;
            }
            finite(&activated)?;
            let mut ffn = vec![0.0; h];
            linear(&activated, &layer.down, &mut ffn, 1, c.ff_dim, h)
                .map_err(CacheError::Linear)?;
            residual_add(&residual, &ffn, &mut hidden).map_err(CacheError::Operation)?;
            finite(&hidden)?;
        }
        let mut norm = vec![0.0; h];
        rmsnorm(
            &hidden,
            &weights.final_norm,
            &mut norm,
            1,
            h,
            c.norm_epsilon,
        )
        .map_err(CacheError::Operation)?;
        let mut logits = vec![0.0; c.vocab_size];
        linear(&norm, &weights.embedding, &mut logits, 1, h, c.vocab_size)
            .map_err(CacheError::Linear)?;
        finite(&logits)?;
        Ok(logits)
    }
}

// All dimensions/weight buffers are validated by the bound Decoder; cache
// dimensions are validated at construction. This helper never exposes state.
fn attend_one(
    x: &[f32],
    weights: AttentionWeights<'_>,
    c: AttentionConfig,
    cache: &mut LayerCache,
    position: usize,
    capacity: usize,
) -> Result<Vec<f32>, CacheError> {
    let h = c.hidden_dim;
    let d = c.head_dim;
    let kv = c.kv_heads * d;
    let mut query = vec![0.0; h];
    let mut key = vec![0.0; kv];
    let mut value = vec![0.0; kv];
    linear(x, weights.query, &mut query, 1, h, h).map_err(CacheError::Linear)?;
    linear(x, weights.key, &mut key, 1, h, kv).map_err(CacheError::Linear)?;
    linear(x, weights.value, &mut value, 1, h, kv).map_err(CacheError::Linear)?;
    for (values, bias) in [
        (&mut query, weights.query_bias),
        (&mut key, weights.key_bias),
        (&mut value, weights.value_bias),
    ] {
        if let Some(bias) = bias {
            for (v, &b) in values.iter_mut().zip(bias) {
                *v += b;
            }
        }
        finite(values)?;
    }
    let mut rotated_query = vec![0.0; h];
    let mut rotated_key = vec![0.0; kv];
    rope(
        &query,
        &mut rotated_query,
        1,
        c.query_heads,
        d,
        position,
        c.rope_base,
    )
    .map_err(CacheError::Rotary)?;
    rope(
        &key,
        &mut rotated_key,
        1,
        c.kv_heads,
        d,
        position,
        c.rope_base,
    )
    .map_err(CacheError::Rotary)?;
    finite(&rotated_query)?;
    finite(&rotated_key)?;
    for head in 0..c.kv_heads {
        let offset = (head * capacity + position) * d;
        cache.key[offset..offset + d].copy_from_slice(&rotated_key[head * d..(head + 1) * d]);
        cache.value[offset..offset + d].copy_from_slice(&value[head * d..(head + 1) * d]);
    }
    let mut context = vec![0.0; h];
    let mut scores = vec![0.0; position + 1];
    let mut probabilities = vec![0.0; position + 1];
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
        softmax(&scores, &mut probabilities, 1, position + 1).map_err(CacheError::Operation)?;
        for (p, &probability) in probabilities.iter().enumerate() {
            let offset = (kvh * capacity + p) * d;
            for i in 0..d {
                context[qh * d + i] += probability * cache.value[offset + i];
            }
        }
    }
    finite(&context)?;
    let mut output = vec![0.0; h];
    linear(&context, weights.output, &mut output, 1, h, h).map_err(CacheError::Linear)?;
    finite(&output)?;
    Ok(output)
}
