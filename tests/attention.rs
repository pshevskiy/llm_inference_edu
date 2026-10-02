use llm_inference::attention::{
    AttentionConfig, AttentionError, AttentionWeights, causal_attention,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn config() -> AttentionConfig {
    AttentionConfig {
        hidden_dim: 4,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        rope_base: 10000.0,
    }
}

#[test]
fn every_named_activation_matches_numpy() {
    let case: Value = serde_json::from_str(include_str!("../fixtures/t03-inputs.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("../fixtures/t03-expected.json")).unwrap();
    let x = floats(&case["x"]);
    let q = floats(&case["query"]);
    let k = floats(&case["key"]);
    let v = floats(&case["value"]);
    let o = floats(&case["output"]);
    let trace = causal_attention(
        &x,
        AttentionWeights {
            query_bias: None,
            key_bias: None,
            value_bias: None,
            query: &q,
            key: &k,
            value: &v,
            output: &o,
        },
        config(),
        3,
        2,
    )
    .unwrap();
    for (name, actual) in [
        ("query", &trace.query),
        ("key", &trace.key),
        ("value", &trace.value),
        ("rotated_query", &trace.rotated_query),
        ("rotated_key", &trace.rotated_key),
        ("context", &trace.context),
        ("output", &trace.output),
    ] {
        let expected = floats(&gold["outputs"][name]);
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0_f32;
        let mut max_relative = 0.0_f32;
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!(a.is_finite());
            let error = (a - b).abs();
            assert!(error <= 1e-5 + 1e-4 * b.abs(), "{name}: {a} != {b}");
            max_abs = max_abs.max(error);
            max_relative = max_relative.max(error / b.abs().max(1e-5));
        }
        println!("{name}: max_abs={max_abs:.8e}, max_relative={max_relative:.8e}");
    }
}

#[test]
fn future_tokens_cannot_change_earlier_outputs() {
    let case: Value = serde_json::from_str(include_str!("../fixtures/t03-inputs.json")).unwrap();
    let mut x = floats(&case["x"]);
    let q = floats(&case["query"]);
    let k = floats(&case["key"]);
    let v = floats(&case["value"]);
    let o = floats(&case["output"]);
    let weights = AttentionWeights {
        query_bias: None,
        key_bias: None,
        value_bias: None,
        query: &q,
        key: &k,
        value: &v,
        output: &o,
    };
    let before = causal_attention(&x, weights, config(), 3, 2).unwrap();
    x[8..].copy_from_slice(&[10.0, -7.0, 3.0, 8.0]);
    let after = causal_attention(&x, weights, config(), 3, 2).unwrap();
    assert_eq!(&before.context[..8], &after.context[..8]);
    assert_eq!(&before.output[..8], &after.output[..8]);
    let prefix = causal_attention(&x[..8], weights, config(), 2, 2).unwrap();
    assert_eq!(prefix.output, after.output[..8]);
}

#[test]
fn every_query_head_uses_its_kv_group_without_duplication() {
    // At one token the sole probability is 1. Distinct KV rows expose mapping.
    let cfg = AttentionConfig {
        hidden_dim: 8,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 2,
        rope_base: 10000.0,
    };
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let q = vec![0.0; 64];
    let k = vec![0.0; 32];
    let mut v = vec![0.0; 32];
    for i in 0..4 {
        v[i * 8 + i] = 1.0;
    }
    let mut o = vec![0.0; 64];
    for i in 0..8 {
        o[i * 8 + i] = 1.0;
    }
    let trace = causal_attention(
        &x,
        AttentionWeights {
            query_bias: None,
            key_bias: None,
            value_bias: None,
            query: &q,
            key: &k,
            value: &v,
            output: &o,
        },
        cfg,
        1,
        0,
    )
    .unwrap();
    assert_eq!(trace.value, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(trace.context, [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    assert_eq!(trace.output, trace.context);
}

#[test]
fn invalid_config_lengths_and_numerics_return_errors() {
    let q = vec![0.0; 16];
    let kv = vec![0.0; 8];
    let weights = AttentionWeights {
        query_bias: None,
        key_bias: None,
        value_bias: None,
        query: &q,
        key: &kv,
        value: &kv,
        output: &q,
    };
    for (cfg, tokens, error) in [
        (config(), 0, AttentionError::ZeroDimension),
        (
            AttentionConfig {
                kv_heads: 3,
                ..config()
            },
            1,
            AttentionError::InvalidHeadGrouping,
        ),
        (
            AttentionConfig {
                hidden_dim: 3,
                ..config()
            },
            1,
            AttentionError::HiddenDimensionMismatch,
        ),
        (
            AttentionConfig {
                query_heads: usize::MAX,
                kv_heads: 1,
                ..config()
            },
            1,
            AttentionError::DimensionOverflow,
        ),
        (config(), usize::MAX, AttentionError::DimensionOverflow),
    ] {
        assert!(matches!(causal_attention(&[1.0;4],weights,cfg,tokens,0),Err(e) if e==error));
    }
    assert!(matches!(
        causal_attention(&[1.0; 3], weights, config(), 1, 0),
        Err(AttentionError::InvalidLength { buffer: "x", .. })
    ));
    for bad in [f32::NAN, f32::INFINITY] {
        assert!(matches!(
            causal_attention(&[1.0, 2.0, 3.0, bad], weights, config(), 1, 0),
            Err(AttentionError::NonFinite {
                buffer: "x",
                index: 3
            })
        ));
    }
    let bad = AttentionWeights {
        query_bias: None,
        key_bias: None,
        value_bias: None,
        query: &q[..15],
        ..weights
    };
    assert!(matches!(
        causal_attention(&[1.0; 4], bad, config(), 1, 0),
        Err(AttentionError::InvalidLength {
            buffer: "query_weight",
            ..
        })
    ));
    let huge = vec![f32::MAX; 16];
    let weights = AttentionWeights {
        query_bias: None,
        key_bias: None,
        value_bias: None,
        query: &huge,
        ..weights
    };
    assert!(matches!(
        causal_attention(&[2.0; 4], weights, config(), 1, 0),
        Err(AttentionError::NonFinite {
            buffer: "query",
            ..
        })
    ));
}

#[test]
fn qkv_biases_are_added_before_rotary_and_values_are_shared_by_group() {
    let q = [0.0; 16];
    let kv = [0.0; 8];
    let output = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    ];
    let weights = AttentionWeights {
        query: &q,
        key: &kv,
        value: &kv,
        output: &output,
        query_bias: Some(&[1., 2., 3., 4.]),
        key_bias: Some(&[5., 6.]),
        value_bias: Some(&[-2., 3.]),
    };
    let result = causal_attention(&[0.; 4], weights, config(), 1, 0).unwrap();
    assert_eq!(result.query, [1., 2., 3., 4.]);
    assert_eq!(result.key, [5., 6.]);
    assert_eq!(result.value, [-2., 3.]);
    assert_eq!(result.rotated_query, result.query);
    assert_eq!(result.output, [-2., 3., -2., 3.]);
    assert!(matches!(
        causal_attention(
            &[0.; 4],
            AttentionWeights {
                key_bias: Some(&[1.]),
                ..weights
            },
            config(),
            1,
            0
        ),
        Err(AttentionError::InvalidLength {
            buffer: "key_bias",
            ..
        })
    ));
    assert!(matches!(
        causal_attention(
            &[0.; 4],
            AttentionWeights {
                value_bias: Some(&[f32::NAN, 0.]),
                ..weights
            },
            config(),
            1,
            0
        ),
        Err(AttentionError::NonFinite {
            buffer: "value_bias",
            ..
        })
    ));
}
