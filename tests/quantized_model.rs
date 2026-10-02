#![cfg(feature = "int8-experiment")]
use llm_inference::kernels::linear_lanes4;
use llm_inference::kv_cache::CachedSession;
use llm_inference::matrix::{MatrixRef, ProjectionError, WeightMatrix};
use llm_inference::model::{Decoder, ModelConfig, ModelError, ModelWeights};
use llm_inference::ops::OpError;
use llm_inference::quantization::{GroupwiseInt8, QuantError, linear_int8_lanes4};
fn tiny() -> Decoder {
    Decoder::from_json(include_str!("../fixtures/tiny-model.json")).unwrap()
}
#[test]
fn fused_four_lanes_match_reconstructed_reference_across_group_and_row_tails() {
    for width in [1, 3, 4, 5, 7, 31, 32, 33, 65, 127, 896, 4864] {
        let x: Vec<_> = (0..3 * width)
            .map(|i| ((i * 17 % 101) as f32 - 50.) / 53.)
            .collect();
        let w: Vec<_> = (0..7 * width)
            .map(|i| ((i * 23 % 97) as f32 - 48.) / 61.)
            .collect();
        for group in [1, 3, 4, 7, 32, 64, 256] {
            let q = GroupwiseInt8::quantize(&w, 7, width, group).unwrap();
            let mut deq = vec![0.; w.len()];
            q.dequantize(&mut deq).unwrap();
            let mut actual = vec![0.; 21];
            let mut expected = vec![0.; 21];
            linear_int8_lanes4(&x, &q, &mut actual, 3).unwrap();
            linear_lanes4(&x, &deq, &mut expected, 3, width, 7).unwrap();
            assert_eq!(actual, expected, "width={width}, group={group}");
        }
    }
}
#[test]
fn one_quantized_decoder_matches_every_reference_activation_and_cached_position() {
    for group in [3, 4, 32] {
        let q = tiny().into_int8(group).unwrap();
        let reference = tiny()
            .into_int8(group)
            .unwrap()
            .into_dequantized_reference()
            .unwrap();
        let ids = [1, 5, 7, 7, 41, 53, 2, 63];
        let trace = q.forward(&ids, true).unwrap().trace.unwrap();
        let expected = reference.forward(&ids, true).unwrap().trace.unwrap();
        assert_eq!(trace.embedding, expected.embedding);
        assert_eq!(trace.final_norm, expected.final_norm);
        assert_eq!(trace.logits, expected.logits);
        for (a, b) in trace.layers.iter().zip(&expected.layers) {
            for (a, b) in [
                (&a.attention_norm, &b.attention_norm),
                (&a.attention.query, &b.attention.query),
                (&a.attention.key, &b.attention.key),
                (&a.attention.value, &b.attention.value),
                (&a.attention.rotated_query, &b.attention.rotated_query),
                (&a.attention.rotated_key, &b.attention.rotated_key),
                (&a.attention.context, &b.attention.context),
                (&a.attention.output, &b.attention.output),
                (&a.after_attention, &b.after_attention),
                (&a.ffn_norm, &b.ffn_norm),
                (&a.gate, &b.gate),
                (&a.up, &b.up),
                (&a.activated_gate, &b.activated_gate),
                (&a.product, &b.product),
                (&a.ffn_output, &b.ffn_output),
                (&a.output, &b.output),
            ] {
                assert_eq!(a, b);
            }
        }
        let mut cache = CachedSession::new(&q, 8).unwrap();
        let mut logits = vec![0.; 64];
        for _ in 0..3 {
            cache.reset();
            for (p, &id) in ids.iter().enumerate() {
                cache.step_into(id, &mut logits).unwrap();
                assert_eq!(logits, q.forward(&ids[..=p], false).unwrap().logits);
                assert_eq!(cache.processed(), p + 1);
            }
        }
    }
    assert_eq!(tiny().logical_weight_bytes().unwrap(), 82560);
    // 20,480 matrix elements +640 scales*4 +160 retained FP32 norm values*4.
    assert_eq!(
        tiny()
            .into_int8(32)
            .unwrap()
            .logical_weight_bytes()
            .unwrap(),
        23680
    );
    assert!(matches!(
        tiny().into_int8(0),
        Err(ModelError::Quantized(QuantError::InvalidGroupSize))
    ));
}
#[test]
fn same_element_count_with_wrong_quantized_shape_is_rejected_at_construction() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/tiny-model.json")).unwrap();
    let config: ModelConfig = serde_json::from_value(fixture["config"].clone()).unwrap();
    let mut weights: ModelWeights = serde_json::from_value(fixture["weights"].clone()).unwrap();
    weights.embedding =
        WeightMatrix::Int8(GroupwiseInt8::quantize(&vec![1.; 2048], 32, 64, 32).unwrap());
    assert!(matches!(
        Decoder::new(config, weights),
        Err(ModelError::InvalidConfig("matrix shape"))
    ));
}
#[test]
fn quantized_lookup_and_projection_admission_preserve_output() {
    let q = GroupwiseInt8::quantize(&[1., 2., 3., 4., 5., 6.], 2, 3, 4).unwrap();
    let m = MatrixRef::Int8(&q);
    let mut out = [37.; 6];
    assert!(matches!(
        m.gather(&[0, 2], &mut out, 2, 3),
        Err(OpError::InvalidToken { position: 1, .. })
    ));
    assert_eq!(out, [37.; 6]);
    assert!(matches!(
        m.gather(&[0], &mut out, 3, 2),
        Err(OpError::InvalidShape {
            actual_rows: 2,
            actual_width: 3,
            ..
        })
    ));
    assert_eq!(out, [37.; 6]);
    assert!(matches!(
        m.project(&[1.; 2], &mut out[..2], 1, 2, 3, true),
        Err(ProjectionError::Quantized(QuantError::InvalidShape { .. }))
    ));
    assert_eq!(out, [37.; 6]);
    for (xlen, ylen, tokens) in [
        (5, 4, 2),
        (7, 4, 2),
        (6, 3, 2),
        (6, 5, 2),
        (0, 4, 0),
        (0, 4, usize::MAX),
    ] {
        let mut y = vec![37.; ylen];
        assert!(linear_int8_lanes4(&vec![1.; xlen], &q, &mut y, tokens).is_err());
        assert!(y.iter().all(|&v| v == 37.));
    }
    let mut valid = [0.; 6];
    m.gather(&[1, 0], &mut valid, 2, 3).unwrap();
    let mut deq = [0.; 6];
    q.dequantize(&mut deq).unwrap();
    assert_eq!(valid, [deq[3], deq[4], deq[5], deq[0], deq[1], deq[2]]);
}
