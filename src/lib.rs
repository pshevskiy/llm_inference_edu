//! Small scalar FP32 operations over explicit, contiguous buffers.

#![forbid(unsafe_code)]

pub mod attention;
pub mod model;
pub mod ops;
pub mod rope;
pub mod tokenizer;
pub mod weights;

/// Invalid dimensions or storage for a linear operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearError {
    ZeroDimension,
    DimensionOverflow,
    InvalidLength {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
}

/// Computes `Y[t, o] = sum_i X[t, i] * W[o, i]` without allocating.
///
/// All buffers are row-major: `X[tokens, input_dim]`,
/// `W[output_dim, input_dim]`, and `Y[tokens, output_dim]`.
/// The weights store one row per output; equivalently, `Y = X * W^T`.
/// There is no bias. The caller owns `y`, whose contents are overwritten.
///
/// Offsets are `t * input_dim + i`, `o * input_dim + i`, and
/// `t * output_dim + o`, respectively. Each dot product accumulates in
/// increasing input-index order using ordinary FP32 multiplication and
/// addition. NaN, infinity, rounding, and arithmetic overflow follow ordinary
/// floating-point semantics; values are not checked for finiteness.
///
/// # Errors
///
/// Rejects zero dimensions, dimension-product overflow, and buffer lengths
/// that do not exactly match their shapes, in that order. All validation
/// completes before the first write; on error, `y` is unchanged.
pub fn linear(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    tokens: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<(), LinearError> {
    if tokens == 0 || input_dim == 0 || output_dim == 0 {
        return Err(LinearError::ZeroDimension);
    }

    let x_len = tokens
        .checked_mul(input_dim)
        .ok_or(LinearError::DimensionOverflow)?;
    let w_len = output_dim
        .checked_mul(input_dim)
        .ok_or(LinearError::DimensionOverflow)?;
    let y_len = tokens
        .checked_mul(output_dim)
        .ok_or(LinearError::DimensionOverflow)?;

    for (buffer, expected, actual) in [
        ("x", x_len, x.len()),
        ("w", w_len, w.len()),
        ("y", y_len, y.len()),
    ] {
        if actual != expected {
            return Err(LinearError::InvalidLength {
                buffer,
                expected,
                actual,
            });
        }
    }

    for t in 0..tokens {
        for o in 0..output_dim {
            let mut sum = 0.0_f32;
            for i in 0..input_dim {
                sum += x[t * input_dim + i] * w[o * input_dim + i];
            }
            y[t * output_dim + o] = sum;
        }
    }
    Ok(())
}
