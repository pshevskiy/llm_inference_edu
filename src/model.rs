//! One concrete pre-normalized decoder: GQA/RoPE, residuals, RMSNorm and SwiGLU.

use crate::LinearError;
use crate::attention::{
    AttentionConfig, AttentionError, AttentionTrace, AttentionWeights, causal_attention,
};
use crate::matrix::{ProjectionError, WeightMatrix};
use crate::ops::{OpError, residual_add, rmsnorm, silu};
use crate::quantization::QuantError;
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
    pub query: WeightMatrix,
    pub key: WeightMatrix,
    pub value: WeightMatrix,
    pub attention_output: WeightMatrix,
    #[serde(default)]
    pub query_bias: Option<Vec<f32>>,
    #[serde(default)]
    pub key_bias: Option<Vec<f32>>,
    #[serde(default)]
    pub value_bias: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub gate: WeightMatrix,
    pub up: WeightMatrix,
    pub down: WeightMatrix,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelWeights {
    pub embedding: WeightMatrix,
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
    Quantized(QuantError),
}

impl From<ProjectionError> for ModelError {
    fn from(e: ProjectionError) -> Self {
        match e {
            ProjectionError::Float(e) => Self::Linear(e),
            ProjectionError::Quantized(e) => Self::Quantized(e),
        }
    }
}
fn matrix_weight(
    buffer: &'static str,
    layer: Option<usize>,
    m: &WeightMatrix,
    rows: usize,
    width: usize,
) -> Result<(), ModelError> {
    let expected = product(rows, width)?;
    if m.len() != expected {
        return Err(ModelError::InvalidLength {
            buffer,
            layer,
            expected,
            actual: m.len(),
        });
    }
    if !m.view().matches_shape(rows, width) {
        return Err(ModelError::InvalidConfig("matrix shape"));
    }
    if let Some(index) = m.view().nonfinite_index() {
        return Err(ModelError::NonFinite { buffer, index });
    }
    Ok(())
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
        product(c.hidden_dim, c.hidden_dim)?;
        product(kv, c.hidden_dim)?;
        product(c.ff_dim, c.hidden_dim)?;
        // Validate configured activation counts, not only weight dimensions.
        product(product(c.max_tokens, c.hidden_dim)?, size_of::<f32>())?;
        product(product(c.max_tokens, c.ff_dim)?, size_of::<f32>())?;
        if weights.layers.len() != c.layers {
            return Err(ModelError::InvalidLayerCount);
        }
        matrix_weight(
            "embedding",
            None,
            &weights.embedding,
            c.vocab_size,
            c.hidden_dim,
        )?;
        weight("final_norm", None, &weights.final_norm, c.hidden_dim)?;
        for (i, l) in weights.layers.iter().enumerate() {
            weight("attention_norm", Some(i), &l.attention_norm, c.hidden_dim)?;
            weight("ffn_norm", Some(i), &l.ffn_norm, c.hidden_dim)?;
            for (name, m, rows, width) in [
                ("query", &l.query, c.hidden_dim, c.hidden_dim),
                ("key", &l.key, kv, c.hidden_dim),
                ("value", &l.value, kv, c.hidden_dim),
                (
                    "attention_output",
                    &l.attention_output,
                    c.hidden_dim,
                    c.hidden_dim,
                ),
                ("gate", &l.gate, c.ff_dim, c.hidden_dim),
                ("up", &l.up, c.ff_dim, c.hidden_dim),
                ("down", &l.down, c.hidden_dim, c.ff_dim),
            ] {
                matrix_weight(name, Some(i), m, rows, width)?;
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
    /// advertised context. CPU experiments may explicitly request up to 2048
    /// positions; this construction limit is not numerical/device qualification.
    /// Existing CLI/native callers retain their separately qualified budgets.
    pub fn from_qwen(
        loaded: crate::weights::LoadedQwen,
        capacity: usize,
    ) -> Result<Self, ModelError> {
        use crate::weights::{LoadError, tensor_shapes};
        let c = loaded.config;
        tensor_shapes(&c).map_err(ModelError::Checkpoint)?;
        if capacity == 0 || capacity > 2048 || capacity > c.max_position_embeddings {
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
                query: take(&format!("{p}.self_attn.q_proj.weight"))?.into(),
                key: take(&format!("{p}.self_attn.k_proj.weight"))?.into(),
                value: take(&format!("{p}.self_attn.v_proj.weight"))?.into(),
                query_bias: Some(take(&format!("{p}.self_attn.q_proj.bias"))?),
                key_bias: Some(take(&format!("{p}.self_attn.k_proj.bias"))?),
                value_bias: Some(take(&format!("{p}.self_attn.v_proj.bias"))?),
                attention_output: take(&format!("{p}.self_attn.o_proj.weight"))?.into(),
                ffn_norm: take(&format!("{p}.post_attention_layernorm.weight"))?,
                gate: take(&format!("{p}.mlp.gate_proj.weight"))?.into(),
                up: take(&format!("{p}.mlp.up_proj.weight"))?.into(),
                down: take(&format!("{p}.mlp.down_proj.weight"))?.into(),
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
                embedding: embedding.into(),
                layers,
                final_norm,
            },
        )
    }

    pub fn config(&self) -> ModelConfig {
        self.config
    }

    /// Logical stored payload (FP32 or INT8 values+scales), not resident pages.
    pub fn logical_weight_bytes(&self) -> Result<usize, ModelError> {
        let mut bytes = self
            .weights
            .embedding
            .logical_bytes()
            .ok_or(ModelError::DimensionOverflow)?;
        bytes = bytes
            .checked_add(product(self.weights.final_norm.len(), 4)?)
            .ok_or(ModelError::DimensionOverflow)?;
        for l in &self.weights.layers {
            for m in [
                &l.query,
                &l.key,
                &l.value,
                &l.attention_output,
                &l.gate,
                &l.up,
                &l.down,
            ] {
                bytes = bytes
                    .checked_add(m.logical_bytes().ok_or(ModelError::DimensionOverflow)?)
                    .ok_or(ModelError::DimensionOverflow)?;
            }
            for v in [&l.attention_norm, &l.ffn_norm].into_iter().chain(
                [&l.query_bias, &l.key_bias, &l.value_bias]
                    .into_iter()
                    .flatten(),
            ) {
                bytes = bytes
                    .checked_add(product(v.len(), 4)?)
                    .ok_or(ModelError::DimensionOverflow)?;
            }
        }
        Ok(bytes)
    }
    /// Development-only reproduction of the rejected T16 INT8 model experiment.
    /// Consumes this model; cache borrows prevent conversion of a live session.
    /// Only matrix weights change; norms/biases stay FP32. Failure exposes no
    /// partially converted model. Default constructors always remain FP32.
    #[cfg(feature = "int8-experiment")]
    pub fn into_int8(self, group: usize) -> Result<Self, ModelError> {
        if !(1..=256).contains(&group) {
            return Err(ModelError::Quantized(QuantError::InvalidGroupSize));
        }
        self.convert_matrices(Some(group))
    }
    /// Once-dequantized independent execution reference; never called per token.
    #[cfg(feature = "int8-experiment")]
    pub fn into_dequantized_reference(self) -> Result<Self, ModelError> {
        self.convert_matrices(None)
    }
    #[cfg(feature = "int8-experiment")]
    fn convert_matrices(mut self, group: Option<usize>) -> Result<Self, ModelError> {
        let c = self.config;
        let kv = product(c.kv_heads, c.head_dim)?;
        fn convert(
            m: &mut WeightMatrix,
            rows: usize,
            width: usize,
            group: Option<usize>,
        ) -> Result<(), ModelError> {
            let previous = std::mem::replace(m, WeightMatrix::Float(Vec::new()));
            *m = match group {
                Some(g) => previous.into_int8(rows, width, g),
                None => previous.into_float(),
            }
            .map_err(ModelError::Quantized)?;
            Ok(())
        }
        convert(
            &mut self.weights.embedding,
            c.vocab_size,
            c.hidden_dim,
            group,
        )?;
        for l in &mut self.weights.layers {
            for (m, rows, width) in [
                (&mut l.query, c.hidden_dim, c.hidden_dim),
                (&mut l.key, kv, c.hidden_dim),
                (&mut l.value, kv, c.hidden_dim),
                (&mut l.attention_output, c.hidden_dim, c.hidden_dim),
                (&mut l.gate, c.ff_dim, c.hidden_dim),
                (&mut l.up, c.ff_dim, c.hidden_dim),
                (&mut l.down, c.hidden_dim, c.ff_dim),
            ] {
                convert(m, rows, width, group)?;
            }
        }
        Self::new(c, self.weights)
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
        self.weights
            .embedding
            .view()
            .gather(token_ids, &mut hidden, c.vocab_size, h)
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
                    query: (&l.query).into(),
                    key: (&l.key).into(),
                    value: (&l.value).into(),
                    output: (&l.attention_output).into(),
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
            l.gate
                .view()
                .project(&ffn_norm, &mut gate, t, h, c.ff_dim, false)
                .map_err(ModelError::from)?;
            l.up.view()
                .project(&ffn_norm, &mut up, t, h, c.ff_dim, false)
                .map_err(ModelError::from)?;
            let mut activated_gate = vec![0.0; ff_len];
            silu(&gate, &mut activated_gate).map_err(ModelError::Operation)?;
            let mut product_values = vec![0.0; ff_len];
            for ((&a, &b), out) in activated_gate.iter().zip(&up).zip(&mut product_values) {
                *out = a * b;
            }
            finite("SwiGLU product", &product_values)?;
            let mut ffn_output = vec![0.0; len];
            l.down
                .view()
                .project(&product_values, &mut ffn_output, t, c.ff_dim, h, false)
                .map_err(ModelError::from)?;
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
            self.weights
                .embedding
                .view()
                .project(&final_norm, &mut all_logits, t, h, c.vocab_size, true)
                .map_err(ModelError::from)?;
            finite("logits", &all_logits)?;
            logits.copy_from_slice(&all_logits[(t - 1) * c.vocab_size..]);
            Some(DecoderTrace {
                embedding: traced_embedding.expect("inspection embedding"),
                layers: traces,
                final_norm,
                logits: all_logits,
            })
        } else {
            self.weights
                .embedding
                .view()
                .project(
                    &final_norm[(t - 1) * h..],
                    &mut logits,
                    1,
                    h,
                    c.vocab_size,
                    true,
                )
                .map_err(ModelError::from)?;
            finite("logits", &logits)?;
            None
        };
        Ok(ForwardOutput { logits, trace })
    }
}
