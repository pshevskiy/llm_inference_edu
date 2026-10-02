use llm_inference::{LinearError, kernels::linear_lanes4, linear};
#[test]
fn lane_kernel_matches_hand_computed_nonsquare_and_tail_values() {
    let mut y = [99.; 4];
    linear_lanes4(
        &[1., 2., 3., 4., 5., 6.],
        &[1., 0., -1., 2., 1., 0.],
        &mut y,
        2,
        3,
        2,
    )
    .unwrap();
    assert_eq!(y, [-2., 4., -2., 13.]);
    let mut tail = [99.];
    linear_lanes4(
        &[1., 2., 3., 4., 5.],
        &[1., -1., 2., 0., -2.],
        &mut tail,
        1,
        5,
        1,
    )
    .unwrap();
    assert_eq!(tail, [-5.]);
}
#[test]
fn lane_and_scalar_match_independent_f64_equations_at_frozen_operator_gate() {
    for input in [
        1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 896, 4864,
    ] {
        let x: Vec<f32> = (0..3 * input)
            .map(|i| ((i * 17 % 101) as f32 - 50.) / 53.)
            .collect();
        let w: Vec<f32> = (0..7 * input)
            .map(|i| ((i * 23 % 97) as f32 - 48.) / 61.)
            .collect();
        let mut lane = vec![0.; 21];
        let mut scalar = vec![0.; 21];
        linear_lanes4(&x, &w, &mut lane, 3, input, 7).unwrap();
        linear(&x, &w, &mut scalar, 3, input, 7).unwrap();
        for t in 0..3 {
            for o in 0..7 {
                let expected: f64 = (0..input)
                    .map(|i| f64::from(x[t * input + i]) * f64::from(w[o * input + i]))
                    .sum();
                for actual in [lane[t * 7 + o], scalar[t * 7 + o]] {
                    assert!(actual.is_finite());
                    assert!(
                        (f64::from(actual) - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                        "I={input}, t={t}, o={o}, actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }
}
#[test]
fn admission_errors_preserve_output_for_all_products_and_buffers() {
    for (t, i, o) in [(0, 3, 2), (2, 0, 2), (2, 3, 0)] {
        let mut out = [37.; 4];
        assert_eq!(
            linear_lanes4(&[], &[], &mut out, t, i, o),
            Err(LinearError::ZeroDimension)
        );
        assert_eq!(out, [37.; 4]);
    }
    for (t, i, o) in [(usize::MAX, 2, 1), (1, usize::MAX, 2), (usize::MAX, 1, 2)] {
        let mut out = [37.; 4];
        assert_eq!(
            linear_lanes4(&[], &[], &mut out, t, i, o),
            Err(LinearError::DimensionOverflow)
        );
        assert_eq!(out, [37.; 4]);
    }
    for (x, w, y) in [
        (5, 6, 4),
        (7, 6, 4),
        (6, 5, 4),
        (6, 7, 4),
        (6, 6, 3),
        (6, 6, 5),
    ] {
        let mut out = vec![37.; y];
        assert!(matches!(
            linear_lanes4(&vec![1.; x], &vec![1.; w], &mut out, 2, 3, 2),
            Err(LinearError::InvalidLength { .. })
        ));
        assert!(out.iter().all(|&v| v == 37.));
    }
}
