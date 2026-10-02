use llm_inference::weights::{Inventory, TensorRecord, load_qwen, tensor_shapes};
use std::path::Path;

fn main() {
    let directory = std::env::args()
        .nth(1)
        .expect("pass the downloaded checkpoint directory");
    let inventory: Inventory =
        serde_json::from_str(include_str!("../fixtures/qwen-inventory.json")).unwrap();
    let loaded = load_qwen(Path::new(&directory), &inventory).unwrap();
    // Emit metadata only after the exact file, complete inventory and payload
    // have passed validation. No second full checkpoint read is needed.
    let records: Vec<_> = tensor_shapes(&loaded.config)
        .unwrap()
        .into_iter()
        .map(|(name, shape)| TensorRecord {
            elements: loaded.tensors[&name].len(),
            name,
            shape,
            storage_dtype: "BF16".into(),
        })
        .collect();
    std::fs::write(
        "fixtures/qwen-tensors.json",
        serde_json::to_string_pretty(&records).unwrap() + "\n",
    )
    .unwrap();
    println!(
        "Decoded {} tensors, logical FP32 payload {} bytes",
        loaded.tensors.len(),
        loaded.logical_weight_bytes
    );
    println!("Tied output shares model.embed_tokens.weight; no duplicate lm_head allocation");
}
