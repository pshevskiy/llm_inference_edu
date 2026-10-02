//! Transparent causal grouped-query attention. One score row is reused across heads.

use crate::ops::{OpError, softmax};
use crate::rope::{RopeError, rope};
use crate::{LinearError, linear};

#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    pub hidden_dim: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
}

/// All projection matrices are `[output,input]` row-major; optional biases are `[output]`.
#[derive(Clone, Copy)]
pub struct AttentionWeights<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub output: &'a [f32],
    pub query_bias: Option<&'a [f32]>,
    pub key_bias: Option<&'a [f32]>,
    pub value_bias: Option<&'a [f32]>,
}

/// Owned intermediate buffers for numerical inspection, not a persistent cache.
/// Q/context have shape `[tokens,query_heads,head_dim]`; K/V use kv_heads.
/// Output has shape `[tokens,hidden_dim]`. All buffers are row-major.
pub struct AttentionTrace {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub rotated_query: Vec<f32>,
    pub rotated_key: Vec<f32>,
    pub context: Vec<f32>,
    pub output: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionError {
    ZeroDimension,
    DimensionOverflow,
    InvalidHeadGrouping,
    HiddenDimensionMismatch,
    InvalidLength {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        buffer: &'static str,
        index: usize,
    },
    Linear(LinearError),
    Rotary(RopeError),
    Probability(OpError),
}

fn finite(buffer: &'static str, x: &[f32]) -> Result<(), AttentionError> {
    for (index, value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(AttentionError::NonFinite { buffer, index });
        }
    }
    Ok(())
}

/// Projects Q/K/V, applies split-half RoPE to Q/K, attends only to positions
/// through the current token, and projects the concatenated query-head context.
///
/// Query head `h` uses KV head `h / (query_heads / kv_heads)`. K/V are not
/// duplicated. Scores are divided by sqrt(head_dim) before stable softmax.
/// One length-tokens score/probability row is reused, avoiding a full T*T matrix.
/// This teaching/reference path owns its scratch and returns named activations;
/// it is not allocation-free. On any error, no result/state is exposed as valid.
pub fn causal_attention(
    x: &[f32],
    weights: AttentionWeights<'_>,
    config: AttentionConfig,
    tokens: usize,
    start_position: usize,
) -> Result<AttentionTrace, AttentionError> {
    let h = config.hidden_dim;
    let d = config.head_dim;
    if tokens == 0 || h == 0 || d == 0 || config.query_heads == 0 || config.kv_heads == 0 {
        return Err(AttentionError::ZeroDimension);
    }
    if !config.query_heads.is_multiple_of(config.kv_heads) {
        return Err(AttentionError::InvalidHeadGrouping);
    }
    let q_width = config
        .query_heads
        .checked_mul(d)
        .ok_or(AttentionError::DimensionOverflow)?;
    if q_width != h {
        return Err(AttentionError::HiddenDimensionMismatch);
    }
    let kv_width = config
        .kv_heads
        .checked_mul(d)
        .ok_or(AttentionError::DimensionOverflow)?;
    let q_len = tokens
        .checked_mul(h)
        .ok_or(AttentionError::DimensionOverflow)?;
    let kv_len = tokens
        .checked_mul(kv_width)
        .ok_or(AttentionError::DimensionOverflow)?;
    let q_weights = h.checked_mul(h).ok_or(AttentionError::DimensionOverflow)?;
    let kv_weights = kv_width
        .checked_mul(h)
        .ok_or(AttentionError::DimensionOverflow)?;
    // Also validate aggregate logical workspace byte arithmetic before allocation.
    q_len
        .checked_mul(4)
        .and_then(|n| kv_len.checked_mul(3).and_then(|k| n.checked_add(k)))
        .and_then(|n| tokens.checked_mul(2).and_then(|s| n.checked_add(s)))
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or(AttentionError::DimensionOverflow)?;
    for (buffer, values, expected) in [
        ("x", x, q_len),
        ("query_weight", weights.query, q_weights),
        ("key_weight", weights.key, kv_weights),
        ("value_weight", weights.value, kv_weights),
        ("output_weight", weights.output, q_weights),
    ] {
        if values.len() != expected {
            return Err(AttentionError::InvalidLength {
                buffer,
                expected,
                actual: values.len(),
            });
        }
        finite(buffer, values)?;
    }

    for (buffer, bias, expected) in [
        ("query_bias", weights.query_bias, h),
        ("key_bias", weights.key_bias, kv_width),
        ("value_bias", weights.value_bias, kv_width),
    ] {
        if let Some(values) = bias {
            if values.len() != expected {
                return Err(AttentionError::InvalidLength {
                    buffer,
                    expected,
                    actual: values.len(),
                });
            }
            finite(buffer, values)?;
        }
    }

    let mut query = vec![0.0; q_len];
    let mut key = vec![0.0; kv_len];
    let mut value = vec![0.0; kv_len];
    linear(x, weights.query, &mut query, tokens, h, h).map_err(AttentionError::Linear)?;
    linear(x, weights.key, &mut key, tokens, h, kv_width).map_err(AttentionError::Linear)?;
    linear(x, weights.value, &mut value, tokens, h, kv_width).map_err(AttentionError::Linear)?;
    for (projected, bias, width) in [
        (&mut query, weights.query_bias, h),
        (&mut key, weights.key_bias, kv_width),
        (&mut value, weights.value_bias, kv_width),
    ] {
        if let Some(bias) = bias {
            for row in projected.chunks_exact_mut(width) {
                for (out, &b) in row.iter_mut().zip(bias) {
                    *out += b;
                }
            }
        }
    }
    finite("query", &query)?;
    finite("key", &key)?;
    finite("value", &value)?;
    let mut rotated_query = vec![0.0; q_len];
    let mut rotated_key = vec![0.0; kv_len];
    rope(
        &query,
        &mut rotated_query,
        tokens,
        config.query_heads,
        d,
        start_position,
        config.rope_base,
    )
    .map_err(AttentionError::Rotary)?;
    rope(
        &key,
        &mut rotated_key,
        tokens,
        config.kv_heads,
        d,
        start_position,
        config.rope_base,
    )
    .map_err(AttentionError::Rotary)?;
    finite("rotated_query", &rotated_query)?;
    finite("rotated_key", &rotated_key)?;

    let mut context = vec![0.0; q_len];
    let mut scores = vec![0.0; tokens];
    let mut probabilities = vec![0.0; tokens];
    let heads_per_group = config.query_heads / config.kv_heads;
    let scale = (d as f32).sqrt();
    for t in 0..tokens {
        for qh in 0..config.query_heads {
            let kvh = qh / heads_per_group;
            let q_offset = t * h + qh * d;
            for (p, score) in scores[..=t].iter_mut().enumerate() {
                let k_offset = p * kv_width + kvh * d;
                let mut dot = 0.0_f32;
                for i in 0..d {
                    dot += rotated_query[q_offset + i] * rotated_key[k_offset + i];
                }
                *score = dot / scale;
            }
            // Causality is a prefix slice: no future score is read or normalized.
            softmax(&scores[..=t], &mut probabilities[..=t], 1, t + 1)
                .map_err(AttentionError::Probability)?;
            for (p, &probability) in probabilities[..=t].iter().enumerate() {
                let v_offset = p * kv_width + kvh * d;
                for i in 0..d {
                    context[q_offset + i] += probability * value[v_offset + i];
                }
            }
        }
    }
    finite("context", &context)?;
    let mut output = vec![0.0; q_len];
    linear(&context, weights.output, &mut output, tokens, h, h).map_err(AttentionError::Linear)?;
    finite("output", &output)?;
    Ok(AttentionTrace {
        query,
        key,
        value,
        rotated_query,
        rotated_key,
        context,
        output,
    })
}
