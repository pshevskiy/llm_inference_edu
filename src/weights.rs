//! Validated loading of the selected Qwen2 BF16 checkpoint, without executing code.

use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct QwenConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_act: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub use_sliding_window: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Asset {
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Inventory {
    pub repository: String,
    pub revision: String,
    pub license: String,
    pub format: String,
    pub shards: Vec<String>,
    pub assets: BTreeMap<String, Asset>,
}

/// Only validated, named FP32 buffers. The tied output head has no separate copy.
/// Q/K/V biases are retained; execution compatibility is qualified in T06.
pub struct LoadedQwen {
    pub config: QwenConfig,
    pub tensors: BTreeMap<String, Vec<f32>>,
    pub logical_weight_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorRecord {
    pub name: String,
    pub shape: Vec<usize>,
    pub storage_dtype: String,
    pub elements: usize,
}

#[derive(Debug)]
pub enum LoadError {
    Io(String),
    Json(String),
    Format(String),
    Integrity(String),
    UnsupportedConfig,
    InvalidShardLayout,
    DimensionOverflow,
    MissingTensor(String),
    UnexpectedTensor(String),
    InvalidShape(String),
    InvalidDtype(String),
    NonFinite(String),
    Allocation,
    ResourceBudget,
}

fn count(shape: &[usize]) -> Result<usize, LoadError> {
    if shape.contains(&0) {
        return Err(LoadError::UnsupportedConfig);
    }
    shape.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim).ok_or(LoadError::DimensionOverflow)
    })
}

pub fn tensor_shapes(c: &QwenConfig) -> Result<BTreeMap<String, Vec<usize>>, LoadError> {
    if c.model_type != "qwen2"
        || c.architectures != ["Qwen2ForCausalLM"]
        || c.hidden_act != "silu"
        || !c.tie_word_embeddings
        || c.torch_dtype != "bfloat16"
        || c.use_sliding_window
        || c.rope_scaling.is_some()
        || [
            c.vocab_size,
            c.hidden_size,
            c.intermediate_size,
            c.num_hidden_layers,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.max_position_embeddings,
        ]
        .contains(&0)
        || !c.hidden_size.is_multiple_of(c.num_attention_heads)
        || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
        || !(c.hidden_size / c.num_attention_heads).is_multiple_of(2)
        || !c.rms_norm_eps.is_finite()
        || c.rms_norm_eps <= 0.0
        || !c.rope_theta.is_finite()
        || c.rope_theta <= 0.0
    {
        return Err(LoadError::UnsupportedConfig);
    }
    let h = c.hidden_size;
    let kv = c
        .num_key_value_heads
        .checked_mul(h / c.num_attention_heads)
        .ok_or(LoadError::DimensionOverflow)?;
    // Check aggregate dimensions before constructing names or reserving payloads.
    c.num_hidden_layers
        .checked_mul(12)
        .and_then(|n| n.checked_add(2))
        .ok_or(LoadError::DimensionOverflow)?;
    let square = count(&[h, h])?;
    let kv_weight = count(&[kv, h])?;
    let ff = count(&[c.intermediate_size, h])?;
    let per_layer = square
        .checked_mul(2)
        .and_then(|n| kv_weight.checked_mul(2).and_then(|v| n.checked_add(v)))
        .and_then(|n| ff.checked_mul(3).and_then(|v| n.checked_add(v)))
        .and_then(|n| h.checked_mul(3).and_then(|v| n.checked_add(v)))
        .and_then(|n| kv.checked_mul(2).and_then(|v| n.checked_add(v)))
        .ok_or(LoadError::DimensionOverflow)?;
    let elements = count(&[c.vocab_size, h])?
        .checked_add(h)
        .and_then(|n| {
            per_layer
                .checked_mul(c.num_hidden_layers)
                .and_then(|v| n.checked_add(v))
        })
        .ok_or(LoadError::DimensionOverflow)?;
    // Raw BF16 + expanded FP32 payload must fit the planning 6 GiB load budget.
    if elements
        .checked_mul(6)
        .ok_or(LoadError::DimensionOverflow)?
        > 6 * 1024 * 1024 * 1024
    {
        return Err(LoadError::ResourceBudget);
    }
    let mut result = BTreeMap::new();
    result.insert("model.embed_tokens.weight".into(), vec![c.vocab_size, h]);
    result.insert("model.norm.weight".into(), vec![h]);
    for layer in 0..c.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        for (name, shape) in [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![h, h]),
            ("self_attn.q_proj.bias", vec![h]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.k_proj.bias", vec![kv]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.v_proj.bias", vec![kv]),
            ("self_attn.o_proj.weight", vec![h, h]),
            ("mlp.gate_proj.weight", vec![c.intermediate_size, h]),
            ("mlp.up_proj.weight", vec![c.intermediate_size, h]),
            ("mlp.down_proj.weight", vec![h, c.intermediate_size]),
        ] {
            count(&shape)?;
            result.insert(format!("{prefix}.{name}"), shape);
        }
    }
    for shape in result.values() {
        count(shape)?
            .checked_mul(4)
            .ok_or(LoadError::DimensionOverflow)?;
    }
    Ok(result)
}

pub fn inspect_qwen(data: &[u8], config: &QwenConfig) -> Result<Vec<TensorRecord>, LoadError> {
    let expected = tensor_shapes(config)?;
    let file = SafeTensors::deserialize(data).map_err(|e| LoadError::Format(e.to_string()))?;
    for name in file.names() {
        if !expected.contains_key(name) {
            return Err(LoadError::UnexpectedTensor(name.to_owned()));
        }
    }
    let mut records = Vec::with_capacity(expected.len());
    for (name, shape) in expected {
        let tensor = file
            .tensor(&name)
            .map_err(|_| LoadError::MissingTensor(name.clone()))?;
        if tensor.shape() != shape {
            return Err(LoadError::InvalidShape(name));
        }
        if tensor.dtype() != Dtype::BF16 {
            return Err(LoadError::InvalidDtype(name));
        }
        let elements = count(&shape)?;
        if tensor.data().len()
            != elements
                .checked_mul(2)
                .ok_or(LoadError::DimensionOverflow)?
        {
            return Err(LoadError::InvalidShape(name));
        }
        records.push(TensorRecord {
            name,
            shape,
            storage_dtype: "BF16".into(),
            elements,
        });
    }
    Ok(records)
}

/// Checks the entire tensor inventory before decoding any tensor. BF16's 16
/// little-endian bits are the high 16 bits of FP32, so expansion is exact.
/// Non-finite weights are rejected; no partially loaded model is returned.
pub fn decode_qwen(data: &[u8], config: QwenConfig) -> Result<LoadedQwen, LoadError> {
    let records = inspect_qwen(data, &config)?;
    let elements = records.iter().try_fold(0usize, |sum, r| {
        sum.checked_add(r.elements)
            .ok_or(LoadError::DimensionOverflow)
    })?;
    let logical_weight_bytes = elements
        .checked_mul(4)
        .ok_or(LoadError::DimensionOverflow)?;
    if logical_weight_bytes
        .checked_add(data.len())
        .ok_or(LoadError::DimensionOverflow)?
        > 6 * 1024 * 1024 * 1024
    {
        return Err(LoadError::ResourceBudget);
    }
    let file = SafeTensors::deserialize(data).map_err(|e| LoadError::Format(e.to_string()))?;
    let mut tensors = BTreeMap::new();
    for record in records {
        let tensor = file
            .tensor(&record.name)
            .map_err(|e| LoadError::Format(e.to_string()))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(record.elements)
            .map_err(|_| LoadError::Allocation)?;
        for bits in tensor.data().chunks_exact(2) {
            let value = f32::from_bits(u32::from(u16::from_le_bytes([bits[0], bits[1]])) << 16);
            if !value.is_finite() {
                return Err(LoadError::NonFinite(record.name));
            }
            values.push(value);
        }
        tensors.insert(record.name, values);
    }
    Ok(LoadedQwen {
        config,
        tensors,
        logical_weight_bytes,
    })
}

pub(crate) fn verified_file(
    directory: &Path,
    name: &str,
    inventory: &Inventory,
) -> Result<Vec<u8>, LoadError> {
    // Only literal names selected here may be opened; inventory paths are not executed.
    let asset = inventory
        .assets
        .get(name)
        .ok_or_else(|| LoadError::Integrity(name.into()))?;
    let path = directory.join(name);
    let size = std::fs::metadata(&path)
        .map_err(|e| LoadError::Io(e.to_string()))?
        .len();
    if size != asset.bytes {
        return Err(LoadError::Integrity(name.into()));
    }
    let bytes = std::fs::read(path).map_err(|e| LoadError::Io(e.to_string()))?;
    if format!("{:x}", Sha256::digest(&bytes)) != asset.sha256 {
        return Err(LoadError::Integrity(name.into()));
    }
    Ok(bytes)
}

/// Loads the selected single-shard artifact after hashes, parser bounds, complete
/// names/shapes and BF16 storage checks. Raw BF16 bytes are freed on return;
/// peak loading payload includes raw bytes plus decoded FP32 weights.
pub fn load_qwen(directory: &Path, inventory: &Inventory) -> Result<LoadedQwen, LoadError> {
    if inventory.repository != "Qwen/Qwen2.5-0.5B-Instruct"
        || inventory.revision != "7ae557604adf67be50417f59c2c2f167def9a775"
        || inventory.format != "safetensors"
        || inventory.shards != ["model.safetensors"]
    {
        return Err(LoadError::InvalidShardLayout);
    }
    let config: QwenConfig =
        serde_json::from_slice(&verified_file(directory, "config.json", inventory)?)
            .map_err(|e| LoadError::Json(e.to_string()))?;
    let shapes = tensor_shapes(&config)?;
    let fp32_bytes = shapes.values().try_fold(0usize, |sum, shape| {
        sum.checked_add(
            count(shape)?
                .checked_mul(4)
                .ok_or(LoadError::DimensionOverflow)?,
        )
        .ok_or(LoadError::DimensionOverflow)
    })?;
    let raw_bytes = inventory
        .assets
        .get("model.safetensors")
        .ok_or_else(|| LoadError::Integrity("model.safetensors".into()))?
        .bytes;
    if u64::try_from(fp32_bytes)
        .map_err(|_| LoadError::DimensionOverflow)?
        .checked_add(raw_bytes)
        .ok_or(LoadError::DimensionOverflow)?
        > 6 * 1024 * 1024 * 1024
    {
        return Err(LoadError::ResourceBudget);
    }
    let bytes = verified_file(directory, "model.safetensors", inventory)?;
    decode_qwen(&bytes, config)
}
