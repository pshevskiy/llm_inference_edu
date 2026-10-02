use llm_inference::ops::{
    OpError, embedding, masked_softmax, residual_add, rmsnorm, silu, softmax,
};
use serde_json::Value;

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;

fn fixtures() -> (Value, Value) {
    (
        serde_json::from_str(include_str!("../fixtures/t01-inputs.json")).unwrap(),
        serde_json::from_str(include_str!("../fixtures/t01-expected.json")).unwrap(),
    )
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect()
}

fn size(value: &Value) -> usize {
    usize::try_from(value.as_u64().unwrap()).unwrap()
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let mut max_absolute = 0.0_f32;
    let mut max_relative = 0.0_f32;
    let mut max_budget_fraction = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{name}[{index}]"
        );
        let error = (actual - expected).abs();
        let budget = ATOL + RTOL * expected.abs();
        max_absolute = max_absolute.max(error);
        max_relative = max_relative.max(error / expected.abs().max(ATOL));
        max_budget_fraction = max_budget_fraction.max(error / budget);
        assert!(
            error <= budget,
            "{name}[{index}]: actual={actual}, expected={expected}, error={error}, budget={budget}",
        );
    }
    println!(
        "{name}: max_abs={max_absolute:.8e}, max_relative={max_relative:.8e}, max_budget_fraction={max_budget_fraction:.8e}"
    );
}

#[test]
fn embedding_matches_numpy_fixture() {
    let (inputs, expected) = fixtures();
    let case = &inputs["embedding"];
    let ids: Vec<_> = case["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(size)
        .collect();
    let table = floats(&case["table"]);
    let gold = floats(&expected["outputs"]["embedding"]);
    let mut y = vec![99.0; gold.len()];
    embedding(
        &ids,
        &table,
        &mut y,
        size(&case["vocab_size"]),
        size(&case["width"]),
    )
    .unwrap();
    assert_eq!(y, gold); // Row copies need exact equality, including repeated IDs.
    assert_close("embedding", &y, &gold);
}

#[test]
fn residual_matches_numpy_fixture() {
    let (inputs, expected) = fixtures();
    let case = &inputs["residual_add"];
    let x = floats(&case["x"]);
    let residual = floats(&case["residual"]);
    let gold = floats(&expected["outputs"]["residual_add"]);
    let mut y = vec![99.0; gold.len()];
    residual_add(&x, &residual, &mut y).unwrap();
    assert_eq!(y, gold); // These dyadic inputs give exactly representable sums.
    assert_close("residual_add", &y, &gold);
}

#[test]
fn rmsnorm_matches_numpy_fixture() {
    let (inputs, expected) = fixtures();
    let case = &inputs["rmsnorm"];
    let x = floats(&case["x"]);
    let weight = floats(&case["weight"]);
    let gold = floats(&expected["outputs"]["rmsnorm"]);
    let mut y = vec![99.0; gold.len()];
    rmsnorm(
        &x,
        &weight,
        &mut y,
        size(&case["rows"]),
        size(&case["width"]),
        case["epsilon"].as_f64().unwrap() as f32,
    )
    .unwrap();
    assert_close("rmsnorm", &y, &gold);
    assert_eq!(&y[3..], &[0.0; 3]);
}

#[test]
fn silu_matches_numpy_fixture() {
    let (inputs, expected) = fixtures();
    let x = floats(&inputs["silu"]["x"]);
    let gold = floats(&expected["outputs"]["silu"]);
    let mut y = vec![99.0; gold.len()];
    silu(&x, &mut y).unwrap();
    assert_close("silu", &y, &gold);
}

#[test]
fn softmax_matches_numpy_fixture() {
    let (inputs, expected) = fixtures();
    let case = &inputs["softmax"];
    let x = floats(&case["logits"]);
    let gold = floats(&expected["outputs"]["softmax"]);
    let mut y = vec![99.0; gold.len()];
    softmax(&x, &mut y, size(&case["rows"]), size(&case["width"])).unwrap();
    assert_close("softmax", &y, &gold);
    for row in y.chunks_exact(size(&case["width"])) {
        assert_close("probability_sum", &[row.iter().sum()], &[1.0]);
        assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }
}

#[test]
fn masked_softmax_matches_numpy_fixture() {
    let (inputs, expected) = fixtures();
    let case = &inputs["masked_softmax"];
    let x = floats(&case["logits"]);
    let mask: Vec<_> = case["mask"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let gold = floats(&expected["outputs"]["masked_softmax"]);
    let mut y = vec![99.0; gold.len()];
    masked_softmax(&x, &mask, &mut y, size(&case["rows"]), size(&case["width"])).unwrap();
    assert_close("masked_softmax", &y, &gold);
    for (&allowed, &value) in mask.iter().zip(&y) {
        if !allowed {
            assert_eq!(value, 0.0);
        }
    }
    for row in y.chunks_exact(size(&case["width"])) {
        assert_close("masked_probability_sum", &[row.iter().sum()], &[1.0]);
    }
}

#[test]
fn embedding_rejects_late_invalid_id_before_copying() {
    for id in [2, usize::MAX] {
        let mut y = [37.0; 4];
        assert_eq!(
            embedding(&[0, id], &[1.0, 2.0, 3.0, 4.0], &mut y, 2, 2),
            Err(OpError::InvalidToken {
                position: 1,
                id,
                vocab_size: 2
            }),
        );
        assert_eq!(y, [37.0; 4]);
    }
}

#[test]
fn embedding_rejects_shapes_and_lengths_without_writes() {
    for (ids, vocab, width, error) in [
        (&[][..], 2, 2, OpError::ZeroDimension),
        (&[0][..], 0, 2, OpError::ZeroDimension),
        (&[0][..], 2, 0, OpError::ZeroDimension),
        (&[0][..], usize::MAX, 2, OpError::DimensionOverflow),
        (&[0, 0][..], 1, usize::MAX, OpError::DimensionOverflow),
    ] {
        let mut y = [37.0; 4];
        assert_eq!(embedding(ids, &[], &mut y, vocab, width), Err(error));
        assert_eq!(y, [37.0; 4]);
    }
    for (table_len, y_len, buffer, actual) in [
        (3, 4, "table", 3),
        (5, 4, "table", 5),
        (4, 3, "y", 3),
        (4, 5, "y", 5),
    ] {
        let mut y = vec![37.0; y_len];
        assert_eq!(
            embedding(&[0, 1], &vec![1.0; table_len], &mut y, 2, 2),
            Err(OpError::InvalidLength {
                buffer,
                expected: 4,
                actual
            })
        );
        assert_eq!(y, vec![37.0; y_len]);
    }
}

#[test]
fn residual_rejects_lengths_without_writes() {
    for (residual_len, y_len, buffer, actual) in [
        (1, 2, "residual", 1),
        (3, 2, "residual", 3),
        (2, 1, "y", 1),
        (2, 3, "y", 3),
    ] {
        let mut y = vec![37.0; y_len];
        assert_eq!(
            residual_add(&[1.0, 2.0], &vec![1.0; residual_len], &mut y),
            Err(OpError::InvalidLength {
                buffer,
                expected: 2,
                actual
            })
        );
        assert_eq!(y, vec![37.0; y_len]);
    }
    let mut y = [37.0];
    assert_eq!(residual_add(&[], &[], &mut y), Err(OpError::ZeroDimension));
    assert_eq!(y, [37.0]);
}

#[test]
fn rmsnorm_rejects_shapes_and_lengths_without_writes() {
    for (rows, width, error) in [
        (0, 2, OpError::ZeroDimension),
        (2, 0, OpError::ZeroDimension),
        (usize::MAX, 2, OpError::DimensionOverflow),
    ] {
        let mut y = [37.0; 4];
        assert_eq!(rmsnorm(&[], &[], &mut y, rows, width, 1e-5), Err(error));
        assert_eq!(y, [37.0; 4]);
    }
    for (x_len, weight_len, y_len, buffer, expected, actual) in [
        (3, 2, 4, "x", 4, 3),
        (5, 2, 4, "x", 4, 5),
        (4, 1, 4, "weight", 2, 1),
        (4, 3, 4, "weight", 2, 3),
        (4, 2, 3, "y", 4, 3),
        (4, 2, 5, "y", 4, 5),
    ] {
        let mut y = vec![37.0; y_len];
        assert_eq!(
            rmsnorm(
                &vec![1.0; x_len],
                &vec![1.0; weight_len],
                &mut y,
                2,
                2,
                1e-5
            ),
            Err(OpError::InvalidLength {
                buffer,
                expected,
                actual
            })
        );
        assert_eq!(y, vec![37.0; y_len]);
    }
}

#[test]
fn rmsnorm_rejects_bad_epsilon_without_writes() {
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut y = [37.0; 2];
        assert_eq!(
            rmsnorm(&[0.0; 2], &[1.0; 2], &mut y, 1, 2, epsilon),
            Err(OpError::InvalidEpsilon)
        );
        assert_eq!(y, [37.0; 2]);
    }
}

#[test]
fn rmsnorm_handles_zero_and_large_finite_rows() {
    let mut y = [37.0; 4];
    rmsnorm(&[0.0, 0.0, 1e30, -1e30], &[1.0; 2], &mut y, 2, 2, 1e-5).unwrap();
    // Equal-magnitude values normalize to +/-1; epsilon is negligible here.
    assert_close("large_rmsnorm", &y, &[0.0, 0.0, 1.0, -1.0]);
    rmsnorm(&[0.0; 4], &[1.0; 2], &mut y, 2, 2, f32::from_bits(1)).unwrap();
    assert_eq!(y, [0.0; 4]);
}

#[test]
fn rmsnorm_rejects_nonfinite_input_and_weight_before_writes() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut y = [37.0; 2];
        assert_eq!(
            rmsnorm(&[1.0, value], &[1.0; 2], &mut y, 1, 2, 1e-5),
            Err(OpError::NonFinite {
                buffer: "x",
                index: 1
            })
        );
        assert_eq!(y, [37.0; 2]);
        assert_eq!(
            rmsnorm(&[1.0; 2], &[1.0, value], &mut y, 1, 2, 1e-5),
            Err(OpError::NonFinite {
                buffer: "weight",
                index: 1
            })
        );
        assert_eq!(y, [37.0; 2]);
    }
}

#[test]
fn silu_handles_extreme_finite_values() {
    let mut y = [37.0; 3];
    silu(&[-1000.0, 1000.0, f32::MAX], &mut y).unwrap();
    assert_eq!(y, [0.0, 1000.0, f32::MAX]);
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn silu_rejects_invalid_input_without_writes() {
    let mut y = [37.0];
    assert_eq!(silu(&[], &mut y), Err(OpError::ZeroDimension));
    assert_eq!(y, [37.0]);
    for y_len in [1, 3] {
        let mut y = vec![37.0; y_len];
        assert_eq!(
            silu(&[1.0; 2], &mut y),
            Err(OpError::InvalidLength {
                buffer: "y",
                expected: 2,
                actual: y_len
            })
        );
        assert_eq!(y, vec![37.0; y_len]);
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut y = [37.0; 2];
        assert_eq!(
            silu(&[1.0, value], &mut y),
            Err(OpError::NonFinite {
                buffer: "x",
                index: 1
            })
        );
        assert_eq!(y, [37.0; 2]);
    }
}

#[test]
fn softmax_handles_extremes_shifts_and_single_entry() {
    let mut y = [37.0; 2];
    softmax(&[f32::MAX, -f32::MAX], &mut y, 1, 2).unwrap();
    assert_eq!(y, [1.0, 0.0]);
    let mut base = [0.0; 3];
    let mut shifted = [0.0; 3];
    softmax(&[2.0, 3.0, 4.0], &mut base, 1, 3).unwrap();
    softmax(&[1002.0, 1003.0, 1004.0], &mut shifted, 1, 3).unwrap();
    assert_close("shift_invariance", &shifted, &base);
    let mut single = [37.0];
    softmax(&[-1000.0], &mut single, 1, 1).unwrap();
    assert_eq!(single, [1.0]);
}

#[test]
fn softmax_rejects_shapes_lengths_and_nonfinite_logits_without_writes() {
    for (rows, width, error) in [
        (0, 2, OpError::ZeroDimension),
        (2, 0, OpError::ZeroDimension),
        (usize::MAX, 2, OpError::DimensionOverflow),
    ] {
        let mut y = [37.0; 4];
        assert_eq!(softmax(&[], &mut y, rows, width), Err(error));
        assert_eq!(y, [37.0; 4]);
    }
    for (x_len, y_len, buffer, actual) in [
        (3, 4, "logits", 3),
        (5, 4, "logits", 5),
        (4, 3, "y", 3),
        (4, 5, "y", 5),
    ] {
        let mut y = vec![37.0; y_len];
        assert_eq!(
            softmax(&vec![1.0; x_len], &mut y, 2, 2),
            Err(OpError::InvalidLength {
                buffer,
                expected: 4,
                actual
            })
        );
        assert_eq!(y, vec![37.0; y_len]);
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut y = [37.0; 4];
        assert_eq!(
            softmax(&[1.0, 2.0, value, 4.0], &mut y, 2, 2),
            Err(OpError::NonFinite {
                buffer: "logits",
                index: 2
            })
        );
        assert_eq!(y, [37.0; 4]);
    }
}

#[test]
fn masked_softmax_rejects_late_all_masked_row_without_writes() {
    let mut y = [37.0; 4];
    assert_eq!(
        masked_softmax(&[1.0; 4], &[true, false, false, false], &mut y, 2, 2),
        Err(OpError::AllMasked { row: 1 })
    );
    assert_eq!(y, [37.0; 4]);
}

#[test]
fn masked_softmax_ignores_masked_nonfinite_logits() {
    let mut y = [37.0; 4];
    masked_softmax(
        &[f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 5.0],
        &[false, false, false, true],
        &mut y,
        1,
        4,
    )
    .unwrap();
    assert_eq!(y, [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn masked_softmax_rejects_wrong_mask_lengths_and_active_nonfinite_values() {
    for mask_len in [3, 5] {
        let mut y = [37.0; 4];
        assert_eq!(
            masked_softmax(&[1.0; 4], &vec![true; mask_len], &mut y, 2, 2),
            Err(OpError::InvalidLength {
                buffer: "mask",
                expected: 4,
                actual: mask_len
            })
        );
        assert_eq!(y, [37.0; 4]);
    }
    let mut y = [37.0; 4];
    assert_eq!(
        masked_softmax(
            &[1.0, 2.0, f32::NEG_INFINITY, 4.0],
            &[true; 4],
            &mut y,
            2,
            2
        ),
        Err(OpError::NonFinite {
            buffer: "logits",
            index: 2
        })
    );
    assert_eq!(y, [37.0; 4]);
}
