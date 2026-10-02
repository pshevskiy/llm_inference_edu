use llm_inference::linear;
use llm_inference::quantization::{GroupwiseInt8, QuantError, linear_int8};
#[test]
fn rounding_zero_groups_row_boundaries_and_tail_have_independent_values() {
    let weights = [127., 0.5, -0.5, 0., 254., 0., 0., 0., 0., 0.];
    let q = GroupwiseInt8::quantize(&weights, 2, 5, 4).unwrap();
    assert_eq!(q.values(), [127, 1, -1, 0, 127, 0, 0, 0, 0, 0]);
    assert_eq!(q.scales(), [1., 2., 1., 1.]);
    assert_eq!(q.groups_per_row(), 2);
    assert_eq!(q.logical_bytes(), 10 + 4 * 4);
    let mut dequantized = [99.; 10];
    q.dequantize(&mut dequantized).unwrap();
    assert_eq!(dequantized, [127., 1., -1., 0., 254., 0., 0., 0., 0., 0.]);
    let mut y = [99.; 2];
    linear_int8(&[0.5, 2., -1., 4., 0.25], &q, &mut y, 1).unwrap();
    assert_eq!(y, [130., 0.]);
    let mut bad = [37.; 9];
    assert!(matches!(
        q.dequantize(&mut bad),
        Err(QuantError::InvalidLength { .. })
    ));
    assert_eq!(bad, [37.; 9]);
}
#[test]
fn reconstructed_error_bound_and_fused_dot_match_independent_reference() {
    for width in [1, 3, 7, 32, 33, 65, 127, 896] {
        for group in [1, 3, 4, 32, 64, 256] {
            let rows = 7;
            let w: Vec<_> = (0..rows * width)
                .map(|i| ((i * 23 % 97) as f32 - 48.) / 61.)
                .collect();
            let x: Vec<_> = (0..3 * width)
                .map(|i| ((i * 17 % 101) as f32 - 50.) / 53.)
                .collect();
            let q = GroupwiseInt8::quantize(&w, rows, width, group).unwrap();
            let mut deq = vec![0.; rows * width];
            q.dequantize(&mut deq).unwrap();
            for o in 0..rows {
                for g in 0..width.div_ceil(group) {
                    let start = g * group;
                    let end = (start + group).min(width);
                    let maximum = w[o * width + start..o * width + end]
                        .iter()
                        .map(|&v| f64::from(v).abs())
                        .fold(0., f64::max);
                    let expected_scale = if maximum == 0. {
                        1.
                    } else {
                        (maximum / 127.) as f32
                    };
                    assert_eq!(q.scales()[o * q.groups_per_row() + g], expected_scale);
                    for i in start..end {
                        let ratio = f64::from(w[o * width + i]) / f64::from(expected_scale);
                        // FP64 equation independently rounded once to the specified FP32 quotient,
                        // then nearest-away-from-zero quantization. Near halves differ from a FP64-only quotient.
                        let expected = f64::from(ratio as f32).round().clamp(-127., 127.) as i8;
                        assert_eq!(
                            q.values()[o * width + i],
                            expected,
                            "width={width}, group={group}, o={o}, i={i}, w_bits={}, scale_bits={}, ratio={ratio}",
                            w[o * width + i].to_bits(),
                            expected_scale.to_bits()
                        );
                        let error =
                            (f64::from(w[o * width + i]) - f64::from(deq[o * width + i])).abs();
                        // Half-step quantization plus independently bounded FP32 division/reconstruction rounding.
                        let bound = f64::from(expected_scale) * 0.5
                            + 4. * f64::from(f32::EPSILON) * maximum;
                        assert!(
                            error <= bound,
                            "width={width},group={group},error={error},bound={bound}"
                        );
                    }
                }
            }
            let mut fused = vec![99.; 3 * rows];
            let mut reference = vec![99.; 3 * rows];
            linear_int8(&x, &q, &mut fused, 3).unwrap();
            linear(&x, &deq, &mut reference, 3, width, rows).unwrap();
            assert_eq!(fused, reference);
            for t in 0..3 {
                for o in 0..rows {
                    let expected: f64 = (0..width)
                        .map(|i| f64::from(x[t * width + i]) * f64::from(deq[o * width + i]))
                        .sum();
                    assert!(
                        (f64::from(fused[t * rows + o]) - expected).abs()
                            <= 1e-5 + 1e-4 * expected.abs()
                    );
                }
            }
        }
    }
}
#[test]
fn invalid_quantization_inputs_and_scale_underflow_are_rejected() {
    for group in [0, 257, usize::MAX] {
        assert!(matches!(
            GroupwiseInt8::quantize(&[], 1, 1, group),
            Err(QuantError::InvalidGroupSize)
        ));
    }
    for (rows, width) in [(0, 1), (1, 0)] {
        assert!(matches!(
            GroupwiseInt8::quantize(&[], rows, width, 32),
            Err(QuantError::ZeroDimension)
        ));
    }
    for (rows, width, group) in [(usize::MAX, 2, 32), (1, usize::MAX, 1)] {
        assert!(matches!(
            GroupwiseInt8::quantize(&[], rows, width, group),
            Err(QuantError::DimensionOverflow)
        ));
    }
    for len in [3, 5] {
        assert!(matches!(
            GroupwiseInt8::quantize(&vec![1.; len], 2, 2, 4),
            Err(QuantError::InvalidLength { .. })
        ));
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            GroupwiseInt8::quantize(&[1., value], 1, 2, 32),
            Err(QuantError::NonFiniteWeight { index: 1 })
        ));
    }
    assert!(matches!(
        GroupwiseInt8::quantize(&[f32::from_bits(1)], 1, 1, 32),
        Err(QuantError::UnrepresentableScale)
    ));
}
#[test]
fn fused_admission_errors_leave_output_unchanged() {
    let q = GroupwiseInt8::quantize(&[1.; 6], 2, 3, 4).unwrap();
    for (x, y, tokens, error) in [
        (0, 4, 0, QuantError::ZeroDimension),
        (0, 4, usize::MAX, QuantError::DimensionOverflow),
    ] {
        let mut out = vec![37.; y];
        assert_eq!(linear_int8(&vec![1.; x], &q, &mut out, tokens), Err(error));
        assert!(out.iter().all(|&v| v == 37.));
    }
    for (x, y) in [(5, 4), (7, 4), (6, 3), (6, 5)] {
        let mut out = vec![37.; y];
        assert!(matches!(
            linear_int8(&vec![1.; x], &q, &mut out, 2),
            Err(QuantError::InvalidLength { .. })
        ));
        assert!(out.iter().all(|&v| v == 37.));
    }
}

#[test]
fn fp32_quotient_can_round_to_half_when_fp64_is_below_it() {
    let max = f32::from_bits(1061777744);
    let half = f32::from_bits(1053389136);
    let q = GroupwiseInt8::quantize(&[max, half], 1, 2, 2).unwrap();
    assert_eq!(q.scales()[0].to_bits(), 1003161439);
    assert_eq!(q.values(), [127, 64]);
    assert!(f64::from(half) / f64::from(q.scales()[0]) < 63.5);
    assert_eq!(half / q.scales()[0], 63.5);
}

#[test]
fn positive_subnormal_scale_has_coarse_rounding_and_explicit_clamping() {
    let weight = f32::from_bits(190);
    let q = GroupwiseInt8::quantize(&[weight], 1, 1, 32).unwrap();
    assert_eq!(q.scales()[0].to_bits(), 1);
    assert_eq!(q.values(), [127]);
    let mut reconstructed = [0.];
    q.dequantize(&mut reconstructed).unwrap();
    assert_eq!(reconstructed[0].to_bits(), 127);
    let units = (f64::from(weight) - f64::from(reconstructed[0])) / f64::from(f32::from_bits(1));
    assert_eq!(units, 63.);
    let mut output = [99.];
    linear_int8(&[1.], &q, &mut output, 1).unwrap();
    assert_eq!(output[0].to_bits(), 127);
}
