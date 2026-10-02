//! Measured CPU candidates; the original `linear` remains the scalar reference.
use crate::{LinearError, validate_linear};

/// Same row-major shapes/error-before-write contract as `linear`, with four
/// independent FP32 accumulation lanes. Reduction order differs from scalar;
/// NaN/infinity/rounding/overflow follow this explicit arithmetic order.
/// Safe Rust permits compiler SIMD vectorization without target intrinsics,
/// alignment assumptions, unsafe code, extra threads or scratch allocation.
/// The tail handles widths not divisible by four. No bias or finite-value check.
pub fn linear_lanes4(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    tokens: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<(), LinearError> {
    validate_linear(x, w, y, tokens, input_dim, output_dim)?;
    for (xrow, yrow) in x
        .chunks_exact(input_dim)
        .zip(y.chunks_exact_mut(output_dim))
    {
        for (row, output) in w.chunks_exact(input_dim).zip(yrow) {
            *output = dot_lanes4(xrow, row);
        }
    }
    Ok(())
}
// Kept out of line so generated assembly has an inspectable dot-product symbol.
#[inline(never)]
fn dot_lanes4(x: &[f32], w: &[f32]) -> f32 {
    let mut sums = [0.0_f32; 4];
    let mut xchunks = x.chunks_exact(4);
    let mut wchunks = w.chunks_exact(4);
    for (a, b) in xchunks.by_ref().zip(wchunks.by_ref()) {
        sums[0] += a[0] * b[0];
        sums[1] += a[1] * b[1];
        sums[2] += a[2] * b[2];
        sums[3] += a[3] * b[3];
    }
    let mut sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
    for (&a, &b) in xchunks.remainder().iter().zip(wchunks.remainder()) {
        sum += a * b;
    }
    sum
}
