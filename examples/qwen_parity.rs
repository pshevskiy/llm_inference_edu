use llm_inference::kv_cache::CachedSession;
use llm_inference::model::Decoder;
use llm_inference::tokenizer::{ChatTokenizer, Message, Role, render_chat};
use llm_inference::weights::{Inventory, load_qwen};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;

fn ids(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}
fn main() {
    let directory = std::env::args().nth(1).expect("pass checkpoint directory");
    let inventory: Inventory =
        serde_json::from_str(include_str!("../fixtures/qwen-inventory.json")).unwrap();
    let tokenizer = ChatTokenizer::load(Path::new(&directory), &inventory).unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/t06-tokenizer.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let expected = ids(&case["ids"]);
        assert_eq!(
            tokenizer.encode(case["text"].as_str().unwrap()).unwrap(),
            expected
        );
        let mut stream = tokenizer.text_stream();
        let mut streamed = String::new();
        for &token in &expected {
            if let Some(text) = stream.push(token).unwrap() {
                streamed.push_str(&text);
            }
        }
        assert_eq!(streamed, tokenizer.decode(&expected, true).unwrap());
        assert_eq!(stream.pending_tokens(), 0);
        assert_eq!(
            tokenizer.decode(&expected, false).unwrap(),
            case["decoded"].as_str().unwrap()
        );
    }
    for case in fixture["chats"].as_array().unwrap() {
        let messages: Vec<_> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Message {
                role: match m["role"].as_str().unwrap() {
                    "system" => Role::System,
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => panic!("unsupported fixture role"),
                },
                content: m["content"].as_str().unwrap(),
            })
            .collect();
        let generation = case["generation_prompt"].as_bool().unwrap();
        assert_eq!(
            render_chat(&messages, generation).unwrap(),
            case["text"].as_str().unwrap()
        );
        assert_eq!(
            tokenizer.chat(&messages, generation).unwrap(),
            ids(&case["ids"])
        );
    }
    println!("Tokenizer: 7 plain cases, 6 chat cases match pinned Transformers");
    let model =
        Decoder::from_qwen(load_qwen(Path::new(&directory), &inventory).unwrap(), 32).unwrap();
    let reference = Path::new("docs/evidence/t06-reference");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(reference.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["revision"], inventory.revision);
    let logit_atol = manifest["logit_atol"].as_f64().unwrap() as f32;
    let intermediate_atol = manifest["intermediate_atol"].as_f64().unwrap() as f32;
    let rtol = manifest["rtol"].as_f64().unwrap() as f32;
    assert_eq!(logit_atol, 1e-3);
    assert_eq!(intermediate_atol, 5e-3);
    assert_eq!(rtol, 1e-4);
    let mut accepted = true;
    for case in manifest["cases"].as_array().unwrap() {
        let tokens = ids(&case["ids"]);
        let full_start = Instant::now();
        let output = model.forward(&tokens, true).unwrap();
        let full_time = full_start.elapsed();
        let trace = output.trace.unwrap();
        let mut session = CachedSession::new(&model, 32).unwrap();
        for (position, &token) in tokens.iter().enumerate() {
            let cached = session.step(token).unwrap();
            assert_eq!(
                cached,
                trace.logits[position * model.config().vocab_size
                    ..(position + 1) * model.config().vocab_size],
                "cached position {position}"
            );
            assert_eq!(session.processed(), position + 1);
        }
        println!(
            "{} cached/uncached: {} positions exactly equal; logical KV {} bytes",
            case["text"],
            tokens.len(),
            session.logical_bytes()
        );
        session.reset();
        let serial_start = Instant::now();
        assert_eq!(session.prefill(&tokens).unwrap(), output.logits);
        let serial_time = serial_start.elapsed();
        session.reset();
        let chunk_start = Instant::now();
        session.prefill(&tokens[..1]).unwrap();
        assert_eq!(session.append(&tokens[1..]).unwrap(), output.logits);
        println!(
            "{} single-sample prefill seconds: uncached_with_inspection={:.6}, serial={:.6}, serial_chunk_calls={:.6}",
            case["text"],
            full_time.as_secs_f64(),
            serial_time.as_secs_f64(),
            chunk_start.elapsed().as_secs_f64()
        );
        for record in case["arrays"].as_array().unwrap() {
            let name = record["name"].as_str().unwrap();
            let atol = if name == "logits" {
                logit_atol
            } else {
                intermediate_atol
            };
            let actual: &[f32] = match name {
                "embedding" => &trace.embedding,
                "final_norm" => &trace.final_norm,
                "logits" => &trace.logits,
                _ => {
                    let (layer, field) = name.split_once('.').unwrap();
                    let index = layer
                        .strip_prefix("layer_")
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();
                    let l = &trace.layers[index];
                    match field {
                        "attention_norm" => &l.attention_norm,
                        "query" => &l.attention.query,
                        "key" => &l.attention.key,
                        "value" => &l.attention.value,
                        "context" => &l.attention.context,
                        "attention_output" => &l.attention.output,
                        "ffn_norm" => &l.ffn_norm,
                        "gate" => &l.gate,
                        "up" => &l.up,
                        "product" => &l.product,
                        "ffn_output" => &l.ffn_output,
                        "output" => &l.output,
                        _ => panic!("unknown reference array"),
                    }
                }
            };
            let bytes = std::fs::read(reference.join(record["file"].as_str().unwrap())).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                record["sha256"].as_str().unwrap()
            );
            let expected: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(bytes.len() % 4, 0);
            assert_eq!(
                expected.len(),
                record["elements"].as_u64().unwrap() as usize
            );
            assert_eq!(actual.len(), expected.len());
            let mut max_abs = 0.0_f32;
            let mut max_relative = 0.0_f32;
            let mut max_budget = 0.0_f32;
            for (&a, &b) in actual.iter().zip(&expected) {
                assert!(a.is_finite() && b.is_finite(), "{name}: nonfinite");
                let error = (a - b).abs();
                max_abs = max_abs.max(error);
                max_relative = max_relative.max(error / b.abs().max(atol));
                max_budget = max_budget.max(error / (atol + rtol * b.abs()));
            }
            println!(
                "{} {name}: n={} max_abs={max_abs:.8e} max_relative={max_relative:.8e} budget={max_budget:.6}",
                case["text"],
                actual.len()
            );
            accepted &= max_budget <= 1.0;
            if name == "logits" {
                for (a, b) in actual
                    .chunks_exact(model.config().vocab_size)
                    .zip(expected.chunks_exact(model.config().vocab_size))
                {
                    let argmax = |v: &[f32]| {
                        v.iter()
                            .enumerate()
                            .max_by(|(i, x), (j, y)| x.total_cmp(y).then_with(|| j.cmp(i)))
                            .unwrap()
                            .0
                    };
                    assert_eq!(argmax(a), argmax(b));
                }
            }
        }
    }
    assert!(
        accepted,
        "one or more arrays exceed the predetermined parity budget"
    );
    println!("Real-model teacher-forced parity passed");
}
