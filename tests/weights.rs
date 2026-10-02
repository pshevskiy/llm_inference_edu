use llm_inference::weights::{
    Asset, Inventory, LoadError, QwenConfig, decode_qwen, inspect_qwen, load_qwen, tensor_shapes,
};
use safetensors::tensor::{Dtype, TensorView, serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn config() -> QwenConfig {
    serde_json::from_str(r#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","hidden_act":"silu","vocab_size":4,"hidden_size":4,"intermediate_size":6,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":32,"rms_norm_eps":0.000001,"rope_theta":1000000,"tie_word_embeddings":true,"torch_dtype":"bfloat16","use_sliding_window":false}"#).unwrap()
}

type Shapes = BTreeMap<String, Vec<usize>>;
type Buffers = BTreeMap<String, Vec<u8>>;

fn contents(c: &QwenConfig, dtype: Dtype) -> (Shapes, Buffers) {
    let shapes = tensor_shapes(c).unwrap();
    let mut data = BTreeMap::new();
    for (name, shape) in &shapes {
        let count: usize = shape.iter().product();
        let mut bytes = Vec::new();
        for i in 0..count {
            let value = [1.0_f32, -2.0, 0.5][i % 3];
            if dtype == Dtype::BF16 {
                bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            } else {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        data.insert(name.clone(), bytes);
    }
    (shapes, data)
}

fn bytes(shapes: &Shapes, data: &Buffers, dtype: Dtype) -> Vec<u8> {
    let views: BTreeMap<_, _> = shapes
        .iter()
        .map(|(name, shape)| {
            (
                name.clone(),
                TensorView::new(dtype, shape.clone(), &data[name]).unwrap(),
            )
        })
        .collect();
    serialize(views, None).unwrap()
}

#[test]
fn bf16_expansion_is_exact_and_tied_output_is_not_duplicated() {
    let c = config();
    let (shapes, data) = contents(&c, Dtype::BF16);
    let file = bytes(&shapes, &data, Dtype::BF16);
    let records = inspect_qwen(&file, &c).unwrap();
    assert_eq!(records.len(), 14);
    let loaded = decode_qwen(&file, c).unwrap();
    assert_eq!(loaded.tensors.len(), 14);
    assert!(!loaded.tensors.contains_key("lm_head.weight"));
    let mut elements = 0;
    for values in loaded.tensors.values() {
        elements += values.len();
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(v, [1.0, -2.0, 0.5][i % 3]);
        }
    }
    assert_eq!(loaded.logical_weight_bytes, elements * 4);
}

#[test]
fn missing_extra_wrong_shape_and_wrong_dtype_are_rejected() {
    let c = config();
    let (mut shapes, mut data) = contents(&c, Dtype::BF16);
    shapes.remove("model.norm.weight");
    data.remove("model.norm.weight");
    assert!(matches!(
        decode_qwen(&bytes(&shapes, &data, Dtype::BF16), c.clone()),
        Err(LoadError::MissingTensor(_))
    ));
    let (mut shapes, mut data) = contents(&c, Dtype::BF16);
    shapes.insert("lm_head.weight".into(), vec![4, 4]);
    data.insert(
        "lm_head.weight".into(),
        data["model.embed_tokens.weight"].clone(),
    );
    assert!(matches!(
        decode_qwen(&bytes(&shapes, &data, Dtype::BF16), c.clone()),
        Err(LoadError::UnexpectedTensor(_))
    ));
    let (mut shapes, data) = contents(&c, Dtype::BF16);
    shapes.insert("model.layers.0.self_attn.q_proj.weight".into(), vec![2, 8]);
    assert!(matches!(
        decode_qwen(&bytes(&shapes, &data, Dtype::BF16), c.clone()),
        Err(LoadError::InvalidShape(_))
    ));
    let (shapes, data) = contents(&c, Dtype::F32);
    assert!(matches!(
        decode_qwen(&bytes(&shapes, &data, Dtype::F32), c),
        Err(LoadError::InvalidDtype(_))
    ));
}

#[test]
fn parser_bounds_nonfinite_data_and_dimension_overflow_are_rejected() {
    let c = config();
    let (shapes, mut data) = contents(&c, Dtype::BF16);
    let file = bytes(&shapes, &data, Dtype::BF16);
    assert!(matches!(
        decode_qwen(&file[..file.len() - 1], c.clone()),
        Err(LoadError::Format(_))
    ));
    let mut extra = file.clone();
    extra.push(0);
    assert!(matches!(
        decode_qwen(&extra, c.clone()),
        Err(LoadError::Format(_))
    ));
    data.get_mut("model.norm.weight").unwrap()[..2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
    assert!(matches!(
        decode_qwen(&bytes(&shapes, &data, Dtype::BF16), c.clone()),
        Err(LoadError::NonFinite(_))
    ));
    let mut huge = c.clone();
    huge.vocab_size = usize::MAX;
    assert!(matches!(
        tensor_shapes(&huge),
        Err(LoadError::DimensionOverflow)
    ));
    let mut huge = c.clone();
    huge.num_hidden_layers = usize::MAX;
    assert!(matches!(
        tensor_shapes(&huge),
        Err(LoadError::DimensionOverflow)
    ));
    let mut huge = c.clone();
    huge.vocab_size = 1 << 30;
    assert!(matches!(
        tensor_shapes(&huge),
        Err(LoadError::ResourceBudget)
    ));
    let mut bad = c.clone();
    bad.num_attention_heads = 3;
    assert!(matches!(
        tensor_shapes(&bad),
        Err(LoadError::UnsupportedConfig)
    ));
    let mut bad = c;
    bad.tie_word_embeddings = false;
    assert!(matches!(
        tensor_shapes(&bad),
        Err(LoadError::UnsupportedConfig)
    ));
}

#[test]
fn local_hashes_and_single_shard_identity_are_checked_without_network() {
    let c = config();
    let (shapes, data) = contents(&c, Dtype::BF16);
    let file = bytes(&shapes, &data, Dtype::BF16);
    let cfg = serde_json::to_vec(&c).unwrap();
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "llm-inference-weights-{}-{suffix}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(dir.join("config.json"), &cfg).unwrap();
    std::fs::write(dir.join("model.safetensors"), &file).unwrap();
    let assets = BTreeMap::from([
        (
            "config.json".into(),
            Asset {
                bytes: cfg.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&cfg)),
            },
        ),
        (
            "model.safetensors".into(),
            Asset {
                bytes: file.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&file)),
            },
        ),
    ]);
    let mut inventory = Inventory {
        repository: "Qwen/Qwen2.5-0.5B-Instruct".into(),
        revision: "7ae557604adf67be50417f59c2c2f167def9a775".into(),
        license: "apache-2.0".into(),
        format: "safetensors".into(),
        shards: vec!["model.safetensors".into()],
        assets,
    };
    assert!(load_qwen(&dir, &inventory).is_ok());
    inventory.shards.push("other.safetensors".into());
    assert!(matches!(
        load_qwen(&dir, &inventory),
        Err(LoadError::InvalidShardLayout)
    ));
    inventory.shards.pop();
    let mut changed = file;
    changed[0] ^= 1;
    std::fs::write(dir.join("model.safetensors"), changed).unwrap();
    assert!(matches!(
        load_qwen(&dir, &inventory),
        Err(LoadError::Integrity(_))
    ));
    std::fs::remove_dir_all(dir).unwrap();
}
