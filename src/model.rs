//! One concrete pre-normalized decoder: GQA/RoPE, residuals, RMSNorm and SwiGLU.

use crate::attention::{
    AttentionConfig, AttentionError, AttentionTrace, AttentionWeights, causal_attention,
};
use crate::kernels::linear_lanes4;
use crate::ops::{OpError, embedding, residual_add, rmsnorm, silu};
use crate::{LinearError, linear};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub layers: usize,
    pub hidden_dim: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ff_dim: usize,
    pub max_tokens: usize,
    pub rope_base: f32,
    pub norm_epsilon: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerWeights {
    pub attention_norm: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention_output: Vec<f32>,
    #[serde(default)]
    pub query_bias: Option<Vec<f32>>,
    #[serde(default)]
    pub key_bias: Option<Vec<f32>>,
    #[serde(default)]
    pub value_bias: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelWeights {
    pub embedding: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelFile {
    config: ModelConfig,
    weights: ModelWeights,
}

/// Weights are immutable after construction and validation. Output weights are
/// tied to the embedding table, with no duplicate storage. Q/K/V biases are optional.
pub struct Decoder {
    config: ModelConfig,
    pub(crate) weights: ModelWeights,
}

pub struct LayerTrace {
    pub attention_norm: Vec<f32>,
    pub attention: AttentionTrace,
    pub after_attention: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub activated_gate: Vec<f32>,
    pub product: Vec<f32>,
    pub ffn_output: Vec<f32>,
    pub output: Vec<f32>,
}

pub struct DecoderTrace {
    pub embedding: Vec<f32>,
    pub layers: Vec<LayerTrace>,
    pub final_norm: Vec<f32>,
    /// Teacher-forced logits `[tokens,vocab_size]`; only present in inspection mode.
    pub logits: Vec<f32>,
}

pub struct ForwardOutput {
    /// Only the final input position's logits, `[vocab_size]`.
    pub logits: Vec<f32>,
    pub trace: Option<DecoderTrace>,
}

#[derive(Debug)]
pub enum ModelError {
    Json(String),
    Checkpoint(crate::weights::LoadError),
    InvalidConfig(&'static str),
    DimensionOverflow,
    InvalidLayerCount,
    ContextLimit,
    TraceLimit,
    InvalidLength {
        buffer: &'static str,
        layer: Option<usize>,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        buffer: &'static str,
        index: usize,
    },
    Operation(OpError),
    Attention(AttentionError),
    Linear(LinearError),
}

fn product(a: usize, b: usize) -> Result<usize, ModelError> {
    a.checked_mul(b).ok_or(ModelError::DimensionOverflow)
}

fn finite(buffer: &'static str, x: &[f32]) -> Result<(), ModelError> {
    for (index, value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(ModelError::NonFinite { buffer, index });
        }
    }
    Ok(())
}

fn weight(
    buffer: &'static str,
    layer: Option<usize>,
    x: &[f32],
    expected: usize,
) -> Result<(), ModelError> {
    if x.len() != expected {
        return Err(ModelError::InvalidLength {
            buffer,
            layer,
            expected,
            actual: x.len(),
        });
    }
    finite(buffer, x)
}

impl Decoder {
    /// Reads the tiny synthetic fixture format, not a pretrained-checkpoint format.
    pub fn from_json(json: &str) -> Result<Self, ModelError> {
        let file: ModelFile =
            serde_json::from_str(json).map_err(|e| ModelError::Json(e.to_string()))?;
        Self::new(file.config, file.weights)
    }

    pub fn new(config: ModelConfig, weights: ModelWeights) -> Result<Self, ModelError> {
        let c = config;
        if [
            c.vocab_size,
            c.layers,
            c.hidden_dim,
            c.query_heads,
            c.kv_heads,
            c.head_dim,
            c.ff_dim,
            c.max_tokens,
        ]
        .contains(&0)
        {
            return Err(ModelError::InvalidConfig("zero dimension"));
        }
        if !c.head_dim.is_multiple_of(2) || !c.query_heads.is_multiple_of(c.kv_heads) {
            return Err(ModelError::InvalidConfig(
                "rotary dimension or head grouping",
            ));
        }
        if product(c.query_heads, c.head_dim)? != c.hidden_dim {
            return Err(ModelError::InvalidConfig("hidden dimension"));
        }
        if !c.rope_base.is_finite()
            || c.rope_base <= 0.0
            || !c.norm_epsilon.is_finite()
            || c.norm_epsilon <= 0.0
        {
            return Err(ModelError::InvalidConfig("base or epsilon"));
        }
        if c.max_tokens > (1 << 24) {
            return Err(ModelError::InvalidConfig("FP32 position limit"));
        }
        let kv = product(c.kv_heads, c.head_dim)?;
        let square = product(c.hidden_dim, c.hidden_dim)?;
        let kv_weights = product(kv, c.hidden_dim)?;
        let ff_weights = product(c.ff_dim, c.hidden_dim)?;
        // Validate configured activation counts, not only weight dimensions.
        product(product(c.max_tokens, c.hidden_dim)?, size_of::<f32>())?;
        product(product(c.max_tokens, c.ff_dim)?, size_of::<f32>())?;
        if weights.layers.len() != c.layers {
            return Err(ModelError::InvalidLayerCount);
        }
        weight(
            "embedding",
            None,
            &weights.embedding,
            product(c.vocab_size, c.hidden_dim)?,
        )?;
        weight("final_norm", None, &weights.final_norm, c.hidden_dim)?;
        for (i, l) in weights.layers.iter().enumerate() {
            for (name, values, len) in [
                ("attention_norm", &l.attention_norm, c.hidden_dim),
                ("query", &l.query, square),
                ("key", &l.key, kv_weights),
                ("value", &l.value, kv_weights),
                ("attention_output", &l.attention_output, square),
                ("ffn_norm", &l.ffn_norm, c.hidden_dim),
                ("gate", &l.gate, ff_weights),
                ("up", &l.up, ff_weights),
                ("down", &l.down, ff_weights),
            ] {
                weight(name, Some(i), values, len)?;
            }
            for (name, values, len) in [
                ("query_bias", &l.query_bias, c.hidden_dim),
                ("key_bias", &l.key_bias, kv),
                ("value_bias", &l.value_bias, kv),
            ] {
                if let Some(values) = values {
                    weight(name, Some(i), values, len)?;
                }
            }
        }
        Ok(Self { config, weights })
    }

    /// Consumes validated Qwen buffers; moves allocations rather than copying
    /// weights. Capacity is a caller admission choice, never the tokenizer's
    /// advertised context. T06 starts with at most 32 tokens.
    pub fn from_qwen(
        loaded: crate::weights::LoadedQwen,
        capacity: usize,
    ) -> Result<Self, ModelError> {
        use crate::weights::{LoadError, tensor_shapes};
        let c = loaded.config;
        tensor_shapes(&c).map_err(ModelError::Checkpoint)?;
        if capacity == 0 || capacity > 32 || capacity > c.max_position_embeddings {
            return Err(ModelError::ContextLimit);
        }
        let mut tensors = loaded.tensors;
        let mut take = |name: &str| {
            tensors
                .remove(name)
                .ok_or_else(|| ModelError::Checkpoint(LoadError::MissingTensor(name.into())))
        };
        let embedding = take("model.embed_tokens.weight")?;
        let final_norm = take("model.norm.weight")?;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(LayerWeights {
                attention_norm: take(&format!("{p}.input_layernorm.weight"))?,
                query: take(&format!("{p}.self_attn.q_proj.weight"))?,
                key: take(&format!("{p}.self_attn.k_proj.weight"))?,
                value: take(&format!("{p}.self_attn.v_proj.weight"))?,
                query_bias: Some(take(&format!("{p}.self_attn.q_proj.bias"))?),
                key_bias: Some(take(&format!("{p}.self_attn.k_proj.bias"))?),
                value_bias: Some(take(&format!("{p}.self_attn.v_proj.bias"))?),
                attention_output: take(&format!("{p}.self_attn.o_proj.weight"))?,
                ffn_norm: take(&format!("{p}.post_attention_layernorm.weight"))?,
                gate: take(&format!("{p}.mlp.gate_proj.weight"))?,
                up: take(&format!("{p}.mlp.up_proj.weight"))?,
                down: take(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        if let Some(name) = tensors.keys().next() {
            return Err(ModelError::Checkpoint(LoadError::UnexpectedTensor(
                name.clone(),
            )));
        }
        Self::new(
            ModelConfig {
                vocab_size: c.vocab_size,
                layers: c.num_hidden_layers,
                hidden_dim: c.hidden_size,
                query_heads: c.num_attention_heads,
                kv_heads: c.num_key_value_heads,
                head_dim: c.hidden_size / c.num_attention_heads,
                ff_dim: c.intermediate_size,
                max_tokens: capacity,
                rope_base: c.rope_theta,
                norm_epsilon: c.rms_norm_eps,
            },
            ModelWeights {
                embedding,
                layers,
                final_norm,
            },
        )
    }

    pub fn config(&self) -> ModelConfig {
        self.config
    }

    /// Logical FP32 payload only: allocator overhead and resident pages differ.
    pub fn logical_weight_bytes(&self) -> Result<usize, ModelError> {
        let mut elements = self
            .weights
            .embedding
            .len()
            .checked_add(self.weights.final_norm.len())
            .ok_or(ModelError::DimensionOverflow)?;
        for layer in &self.weights.layers {
            for values in [
                &layer.attention_norm,
                &layer.query,
                &layer.key,
                &layer.value,
                &layer.attention_output,
                &layer.ffn_norm,
                &layer.gate,
                &layer.up,
                &layer.down,
            ] {
                elements = elements
                    .checked_add(values.len())
                    .ok_or(ModelError::DimensionOverflow)?;
            }
            for values in [&layer.query_bias, &layer.key_bias, &layer.value_bias]
                .into_iter()
                .flatten()
            {
                elements = elements
                    .checked_add(values.len())
                    .ok_or(ModelError::DimensionOverflow)?;
            }
        }
        product(elements, size_of::<f32>())
    }

    /// Full-prefix scalar forward. `inspect=true` retains named activations and
    /// all-position logits, limited to 32 tokens. Normal mode returns only last
    /// logits and drops each layer's temporary buffers before the next layer.
    /// No KV cache or model runtime is used. A failed call returns no valid state.
    pub fn forward(&self, token_ids: &[usize], inspect: bool) -> Result<ForwardOutput, ModelError> {
        let c = self.config;
        let t = token_ids.len();
        let h = c.hidden_dim;
        if t == 0 || t > c.max_tokens {
            return Err(ModelError::ContextLimit);
        }
        if inspect && t > 32 {
            return Err(ModelError::TraceLimit);
        }
        let len = product(t, h)?;
        let ff_len = product(t, c.ff_dim)?;
        let mut hidden = vec![0.0; len];
        embedding(
            token_ids,
            &self.weights.embedding,
            &mut hidden,
            c.vocab_size,
            h,
        )
        .map_err(ModelError::Operation)?;
        let traced_embedding = if inspect { Some(hidden.clone()) } else { None };
        let mut traces = Vec::new();
        for l in &self.weights.layers {
            let mut attention_norm = vec![0.0; len];
            rmsnorm(
                &hidden,
                &l.attention_norm,
                &mut attention_norm,
                t,
                h,
                c.norm_epsilon,
            )
            .map_err(ModelError::Operation)?;
            let attention = causal_attention(
                &attention_norm,
                AttentionWeights {
                    query: &l.query,
                    key: &l.key,
                    value: &l.value,
                    output: &l.attention_output,
                    query_bias: l.query_bias.as_deref(),
                    key_bias: l.key_bias.as_deref(),
                    value_bias: l.value_bias.as_deref(),
                },
                AttentionConfig {
                    hidden_dim: h,
                    query_heads: c.query_heads,
                    kv_heads: c.kv_heads,
                    head_dim: c.head_dim,
                    rope_base: c.rope_base,
                },
                t,
                0,
            )
            .map_err(ModelError::Attention)?;
            let mut after_attention = vec![0.0; len];
            residual_add(&hidden, &attention.output, &mut after_attention)
                .map_err(ModelError::Operation)?;
            let mut ffn_norm = vec![0.0; len];
            rmsnorm(
                &after_attention,
                &l.ffn_norm,
                &mut ffn_norm,
                t,
                h,
                c.norm_epsilon,
            )
            .map_err(ModelError::Operation)?;
            let mut gate = vec![0.0; ff_len];
            let mut up = vec![0.0; ff_len];
            linear(&ffn_norm, &l.gate, &mut gate, t, h, c.ff_dim).map_err(ModelError::Linear)?;
            linear(&ffn_norm, &l.up, &mut up, t, h, c.ff_dim).map_err(ModelError::Linear)?;
            let mut activated_gate = vec![0.0; ff_len];
            silu(&gate, &mut activated_gate).map_err(ModelError::Operation)?;
            let mut product_values = vec![0.0; ff_len];
            for ((&a, &b), out) in activated_gate.iter().zip(&up).zip(&mut product_values) {
                *out = a * b;
            }
            finite("SwiGLU product", &product_values)?;
            let mut ffn_output = vec![0.0; len];
            linear(&product_values, &l.down, &mut ffn_output, t, c.ff_dim, h)
                .map_err(ModelError::Linear)?;
            let mut next = vec![0.0; len];
            residual_add(&after_attention, &ffn_output, &mut next)
                .map_err(ModelError::Operation)?;
            finite("layer output", &next)?;
            if inspect {
                traces.push(LayerTrace {
                    attention_norm,
                    attention,
                    after_attention,
                    ffn_norm,
                    gate,
                    up,
                    activated_gate,
                    product: product_values,
                    ffn_output,
                    output: next.clone(),
                });
            }
            hidden = next;
        }
        let mut final_norm = vec![0.0; len];
        rmsnorm(
            &hidden,
            &self.weights.final_norm,
            &mut final_norm,
            t,
            h,
            c.norm_epsilon,
        )
        .map_err(ModelError::Operation)?;
        let mut logits = vec![0.0; c.vocab_size];
        let trace = if inspect {
            let mut all_logits = vec![0.0; product(t, c.vocab_size)?];
            linear_lanes4(
                &final_norm,
                &self.weights.embedding,
                &mut all_logits,
                t,
                h,
                c.vocab_size,
            )
            .map_err(ModelError::Linear)?;
            finite("logits", &all_logits)?;
            logits.copy_from_slice(&all_logits[(t - 1) * c.vocab_size..]);
            Some(DecoderTrace {
                embedding: traced_embedding.expect("inspection embedding"),
                layers: traces,
                final_norm,
                logits: all_logits,
            })
        } else {
            linear_lanes4(
                &final_norm[(t - 1) * h..],
                &self.weights.embedding,
                &mut logits,
                1,
                h,
                c.vocab_size,
            )
            .map_err(ModelError::Linear)?;
            finite("logits", &logits)?;
            None
        };
        Ok(ForwardOutput { logits, trace })
    }
}
