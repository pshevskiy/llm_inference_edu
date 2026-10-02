use llm_inference::LinearError;
#[path = "../examples/support/prefill_kernel.rs"]
mod prefill_kernel;
use llm_inference::kernels::linear_lanes4;
use prefill_kernel::{ChunkError, linear_chunked};
#[test]
fn all_chunks_and_tails_match_lane_order_and_independent_f64() {
    for tokens in [1, 3, 7, 8, 9, 16, 32] {
        for width in [1, 3, 4, 5, 7, 8, 9, 32, 127, 896] {
            let output = 11;
            let x: Vec<_> = (0..tokens * width)
                .map(|i| ((i * 17 % 101) as f32 - 50.) / 53.)
                .collect();
            let w: Vec<_> = (0..output * width)
                .map(|i| ((i % 251) as f32 - 125.) / 257.)
                .collect();
            let mut reference = vec![0.; tokens * output];
            linear_lanes4(&x, &w, &mut reference, tokens, width, output).unwrap();
            for chunk in 1..=8 {
                let mut actual = vec![99.; tokens * output];
                linear_chunked(&x, &w, &mut actual, tokens, width, output, chunk).unwrap();
                assert_eq!(actual, reference, "T={tokens}, I={width}, chunk={chunk}");
                for t in 0..tokens {
                    for o in 0..output {
                        let expected: f64 = (0..width)
                            .map(|i| f64::from(x[t * width + i]) * f64::from(w[o * width + i]))
                            .sum();
                        assert!(
                            (f64::from(actual[t * output + o]) - expected).abs()
                                <= 1e-5 + 1e-4 * expected.abs()
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn chunk_shape_overflow_and_length_admission_preserve_output() {
    for chunk in [0, 9, usize::MAX] {
        let mut out = [37.; 4];
        assert_eq!(
            linear_chunked(&[], &[], &mut out, 2, 3, 2, chunk),
            Err(ChunkError::InvalidChunkSize)
        );
        assert_eq!(out, [37.; 4]);
    }
    for (t, i, o, error) in [
        (0, 3, 2, LinearError::ZeroDimension),
        (2, 0, 2, LinearError::ZeroDimension),
        (2, 3, 0, LinearError::ZeroDimension),
        (usize::MAX, 2, 1, LinearError::DimensionOverflow),
        (1, usize::MAX, 2, LinearError::DimensionOverflow),
        (usize::MAX, 1, 2, LinearError::DimensionOverflow),
    ] {
        let mut out = [37.; 4];
        assert_eq!(
            linear_chunked(&[], &[], &mut out, t, i, o, 4),
            Err(ChunkError::Linear(error))
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
            linear_chunked(&vec![1.; x], &vec![1.; w], &mut out, 2, 3, 2, 4),
            Err(ChunkError::Linear(LinearError::InvalidLength { .. }))
        ));
        assert!(out.iter().all(|&v| v == 37.));
    }
}
