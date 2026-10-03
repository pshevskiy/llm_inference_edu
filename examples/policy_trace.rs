//! One short CPU correctness trace for T19, not a repeated performance benchmark.
use llm_inference::kv_cache::CachedSession;
use llm_inference::model::Decoder;
use llm_inference::weights::{Inventory, load_qwen};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn main() {
    assert!(!cfg!(debug_assertions), "release build required");
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        3,
        "pass MODEL_DIR REFERENCE_MANIFEST NEW_OUTPUT_DIR"
    );
    let directory = Path::new(&args[0]);
    let manifest_bytes = std::fs::read(&args[1]).unwrap();
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let inventory: Inventory =
        serde_json::from_str(include_str!("../fixtures/qwen-inventory.json")).unwrap();
    assert_eq!(manifest["revision"], inventory.revision);
    let output = Path::new(&args[2]);
    assert!(!output.exists(), "fresh output required");
    std::fs::create_dir_all(output).unwrap();
    let mut files = std::fs::read_dir("src")
        .unwrap()
        .map(|f| f.unwrap().path())
        .collect::<Vec<_>>();
    files.extend([
        "Cargo.toml".into(),
        "Cargo.lock".into(),
        "examples/policy_trace.rs".into(),
    ]);
    files.sort();
    let mut hash = Sha256::new();
    for file in files {
        let name = file.to_string_lossy();
        let bytes = std::fs::read(&file).unwrap();
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    let identity = json!({"model_revision":inventory.revision,"model_sha256":inventory.assets["model.safetensors"].sha256,"manifest_sha256":digest(&manifest_bytes),"precision":"FP32 weights/compute/KV","capacity":32,"runtime_source_sha256":format!("{:x}",hash.finalize()),"executable_sha256":digest(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),"scope":"one buffered step per teacher-forced position; logits only; no sampling/text/repeated benchmark","provenance":"runtime source snapshot and actual binary hash; rebuild first"});
    std::fs::write(
        output.join("identity.json"),
        serde_json::to_vec_pretty(&identity).unwrap(),
    )
    .unwrap();
    let before = Instant::now();
    let model = Decoder::from_qwen(load_qwen(directory, &inventory).unwrap(), 32).unwrap();
    let load_ns = before.elapsed().as_nanos();
    let mut session = CachedSession::new(&model, 32).unwrap();
    let mut logits = vec![0.; model.config().vocab_size];
    let mut records = Vec::new();
    for (case_index, case) in manifest["cases"].as_array().unwrap().iter().enumerate() {
        let ids = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert!(
            !ids.is_empty()
                && ids.len() <= 32
                && ids.iter().all(|&id| id < model.config().vocab_size)
        );
        session.reset();
        for (position, &token) in ids.iter().enumerate() {
            let before = Instant::now();
            session.step_into(token, &mut logits).unwrap();
            let step_ns = before.elapsed().as_nanos();
            assert!(logits.iter().all(|v| v.is_finite()));
            let bytes: Vec<_> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            let name = format!("{case_index}-{position}.bin");
            std::fs::write(output.join(&name), &bytes).unwrap();
            records.push(json!({"case":case_index,"ids":ids,"position":position,"file":name,"sha256":digest(&bytes),"step_ns":step_ns,"processed":session.processed()}));
        }
    }
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(
            &json!({"load_verify_expand_ns":load_ns,"records":records,"passed":true}),
        )
        .unwrap(),
    )
    .unwrap();
    println!("Buffered Rust policy correctness trace saved");
}
