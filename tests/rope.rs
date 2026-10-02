use llm_inference::rope::{RopeError, rope};
use serde_json::Value;

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn size(value: &Value) -> usize {
    usize::try_from(value.as_u64().unwrap()).unwrap()
}

#[test]
fn matches_independent_rotary_fixtures_and_preserves_head_norm() {
    let inputs: Value = serde_json::from_str(include_str!("../fixtures/t02-inputs.json")).unwrap();
    let expected: Value =
        serde_json::from_str(include_str!("../fixtures/t02-expected.json")).unwrap();
    for case in inputs["cases"].as_array().unwrap() {
        let x = floats(&case["x"]);
        let name = case["name"].as_str().unwrap();
        let gold = floats(&expected["outputs"][name]);
        let dim = size(&case["head_dim"]);
        let mut y = vec![99.0; x.len()];
        rope(
            &x,
            &mut y,
            size(&case["tokens"]),
            size(&case["heads"]),
            dim,
            size(&case["start_position"]),
            case["base"].as_f64().unwrap() as f32,
        )
        .unwrap();
        let mut max_error = 0.0_f32;
        let mut max_relative = 0.0_f32;
        for (&actual, &expected) in y.iter().zip(&gold) {
            assert!(actual.is_finite());
            let error = (actual - expected).abs();
            assert!(error <= 1e-5 + 1e-4 * expected.abs());
            max_error = max_error.max(error);
            max_relative = max_relative.max(error / expected.abs().max(1e-5));
        }
        println!("{name}: max_abs={max_error:.8e}, max_relative={max_relative:.8e}");
        for (before, after) in x.chunks_exact(dim).zip(y.chunks_exact(dim)) {
            let norm_before: f64 = before.iter().map(|&v| f64::from(v).powi(2)).sum();
            let norm_after: f64 = after.iter().map(|&v| f64::from(v).powi(2)).sum();
            assert!((norm_before - norm_after).abs() <= 1e-5 + 1e-4 * norm_before);
        }
    }
}

#[test]
fn zero_position_is_identity_for_every_head() {
    let x = [1.0, 2.0, 3.0, 4.0, -1.0, 5.0, -3.0, 2.0];
    let mut y = [99.0; 8];
    rope(&x, &mut y, 1, 2, 4, 0, 10000.0).unwrap();
    assert_eq!(y, x);
}

#[test]
fn absolute_offsets_match_the_same_rows_in_a_longer_call() {
    let x: Vec<_> = (1..=12).map(|i| i as f32).collect();
    let mut whole = [0.0; 12];
    let mut tail = [0.0; 8];
    rope(&x, &mut whole, 3, 1, 4, 5, 10000.0).unwrap();
    rope(&x[4..], &mut tail, 2, 1, 4, 6, 10000.0).unwrap();
    assert_eq!(tail, whole[4..]);
}

#[test]
fn split_half_pairing_matches_a_hand_calculated_one_radian_rotation() {
    // Base 1 gives both pairs the same angle. At position 1 it is one radian.
    let mut y = [0.0; 4];
    rope(&[1.0, 0.0, 0.0, 1.0], &mut y, 1, 1, 4, 1, 1.0).unwrap();
    let gold = [0.5403023, -0.84147096, 0.84147096, 0.5403023];
    for (&a, &b) in y.iter().zip(&gold) {
        assert!((a - b).abs() <= 1e-5);
    }
}

#[test]
fn invalid_shapes_positions_and_base_leave_output_unchanged() {
    for (tokens, heads, dim, start, base, error) in [
        (0, 1, 4, 0, 10000.0, RopeError::ZeroDimension),
        (1, 0, 4, 0, 10000.0, RopeError::ZeroDimension),
        (1, 1, 0, 0, 10000.0, RopeError::ZeroDimension),
        (1, 1, 3, 0, 10000.0, RopeError::OddHeadDimension),
        (1, usize::MAX, 2, 0, 10000.0, RopeError::DimensionOverflow),
        (usize::MAX, 1, 2, 0, 10000.0, RopeError::DimensionOverflow),
        (2, 1, 2, usize::MAX, 10000.0, RopeError::PositionOverflow),
        (
            1,
            1,
            4,
            (1 << 24) + 1,
            10000.0,
            RopeError::PositionOutOfRange,
        ),
        (1, 1, 4, 0, 0.0, RopeError::InvalidBase),
        (1, 1, 4, 0, -1.0, RopeError::InvalidBase),
        (1, 1, 4, 0, f32::NAN, RopeError::InvalidBase),
        (1, 1, 4, 0, f32::INFINITY, RopeError::InvalidBase),
    ] {
        let mut y = [37.0; 4];
        assert_eq!(
            rope(&[1.0; 4], &mut y, tokens, heads, dim, start, base),
            Err(error)
        );
        assert_eq!(y, [37.0; 4]);
    }
}

#[test]
fn short_and_long_lengths_and_late_nonfinite_input_leave_output_unchanged() {
    for (x_len, y_len, buffer, actual) in [
        (3, 4, "x", 3),
        (5, 4, "x", 5),
        (4, 3, "y", 3),
        (4, 5, "y", 5),
    ] {
        let mut y = vec![37.0; y_len];
        assert_eq!(
            rope(&vec![1.0; x_len], &mut y, 1, 1, 4, 0, 10000.0),
            Err(RopeError::InvalidLength {
                buffer,
                expected: 4,
                actual
            })
        );
        assert_eq!(y, vec![37.0; y_len]);
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut y = [37.0; 4];
        assert_eq!(
            rope(&[1.0, 2.0, 3.0, bad], &mut y, 1, 1, 4, 0, 10000.0),
            Err(RopeError::NonFinite { index: 3 })
        );
        assert_eq!(y, [37.0; 4]);
    }
}

#[test]
fn exact_position_limit_and_invalid_angle_are_checked() {
    let mut y = [37.0; 2];
    rope(&[1.0, 0.0], &mut y, 1, 1, 2, 1 << 24, 10000.0).unwrap();
    assert!(y.iter().all(|v| v.is_finite()));
    let mut y = [37.0; 8];
    assert_eq!(
        rope(&[1.0; 8], &mut y, 1, 1, 8, 1 << 24, f32::from_bits(1)),
        Err(RopeError::InvalidAngle)
    );
    assert_eq!(y, [37.0; 8]);
}
