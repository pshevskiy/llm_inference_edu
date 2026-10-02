use llm_inference::model::{Decoder, ModelError};
use serde_json::Value;

const MODEL: &str = include_str!("../fixtures/tiny-model.json");

fn close(name: &str, actual: &[f32], gold: &Value) {
    let expected: Vec<_> = gold
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0_f32;
    let mut max_relative = 0.0_f32;
    for (&a, &b) in actual.iter().zip(&expected) {
        assert!(a.is_finite());
        let error = (a - b).abs();
        assert!(
            error <= 1e-5 + 1e-4 * b.abs(),
            "{name}: actual {a}, expected {b}, error {error}"
        );
        max_abs = max_abs.max(error);
        max_relative = max_relative.max(error / b.abs().max(1e-5));
    }
    println!("{name}: max_abs={max_abs:.8e}, max_relative={max_relative:.8e}");
}

#[test]
fn every_block_and_teacher_forced_logit_matches_numpy() {
    let decoder = Decoder::from_json(MODEL).unwrap();
    let inputs: Value = serde_json::from_str(include_str!("../fixtures/t04-inputs.json")).unwrap();
    let ids: Vec<_> = inputs["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let expected: Value =
        serde_json::from_str(include_str!("../fixtures/t04-expected.json")).unwrap();
    let out = decoder.forward(&ids, true).unwrap();
    let trace = out.trace.unwrap();
    let gold = &expected["outputs"];
    close("embedding", &trace.embedding, &gold["embedding"]);
    for (i, l) in trace.layers.iter().enumerate() {
        for (name, actual) in [
            ("attention_norm", &l.attention_norm),
            ("query", &l.attention.query),
            ("key", &l.attention.key),
            ("value", &l.attention.value),
            ("rotated_query", &l.attention.rotated_query),
            ("rotated_key", &l.attention.rotated_key),
            ("context", &l.attention.context),
            ("attention_output", &l.attention.output),
            ("after_attention", &l.after_attention),
            ("ffn_norm", &l.ffn_norm),
            ("gate", &l.gate),
            ("up", &l.up),
            ("activated_gate", &l.activated_gate),
            ("product", &l.product),
            ("ffn_output", &l.ffn_output),
            ("output", &l.output),
        ] {
            let key = format!("layer_{i}.{name}");
            close(&key, actual, &gold[&key]);
        }
    }
    close("final_norm", &trace.final_norm, &gold["final_norm"]);
    close("logits", &trace.logits, &gold["logits"]);
    assert_eq!(out.logits, trace.logits[2 * 64..]);
    let plain = decoder.forward(&ids, false).unwrap();
    assert!(plain.trace.is_none());
    assert_eq!(plain.logits, out.logits);
}

#[test]
fn prefixes_match_teacher_forced_rows_and_future_tokens_cannot_change_them() {
    let model = Decoder::from_json(MODEL).unwrap();
    let original = model.forward(&[1, 5, 7], true).unwrap().trace.unwrap();
    let changed = model.forward(&[1, 5, 22], true).unwrap().trace.unwrap();
    assert_eq!(&original.logits[..128], &changed.logits[..128]);
    for t in 1..=3 {
        let ids = &[1, 5, 7][..t];
        let out = model.forward(ids, false).unwrap();
        assert_eq!(out.logits, original.logits[(t - 1) * 64..t * 64]);
    }
}

#[test]
fn invalid_tokens_context_config_and_weights_are_rejected() {
    let model = Decoder::from_json(MODEL).unwrap();
    assert!(matches!(
        model.forward(&[], false),
        Err(ModelError::ContextLimit)
    ));
    assert!(matches!(
        model.forward(&[1; 33], false),
        Err(ModelError::ContextLimit)
    ));
    assert!(matches!(
        model.forward(&[1, 64], false),
        Err(ModelError::Operation(_))
    ));
    let mut json: Value = serde_json::from_str(MODEL).unwrap();
    json["weights"]["layers"][0]["gate"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(matches!(
        Decoder::from_json(&json.to_string()),
        Err(ModelError::InvalidLength {
            buffer: "gate",
            layer: Some(0),
            ..
        })
    ));
    let mut json: Value = serde_json::from_str(MODEL).unwrap();
    json["config"]["kv_heads"] = Value::from(3);
    assert!(matches!(
        Decoder::from_json(&json.to_string()),
        Err(ModelError::InvalidConfig(_))
    ));
    let mut json: Value = serde_json::from_str(MODEL).unwrap();
    json["config"]["norm_epsilon"] = Value::from(0);
    assert!(matches!(
        Decoder::from_json(&json.to_string()),
        Err(ModelError::InvalidConfig(_))
    ));
    let mut json: Value = serde_json::from_str(MODEL).unwrap();
    json["weights"]["layers"].as_array_mut().unwrap().pop();
    assert!(matches!(
        Decoder::from_json(&json.to_string()),
        Err(ModelError::InvalidLayerCount)
    ));
    assert!(matches!(Decoder::from_json("{}"), Err(ModelError::Json(_))));
}
