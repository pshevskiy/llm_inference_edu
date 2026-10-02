//! Scalar operations on contiguous FP32 buffers. Every output belongs to the caller.
//! All validation finishes before writes, so an error leaves output unchanged.

/// Shape, token, or numerical input errors for the basic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpError {
    ZeroDimension,
    DimensionOverflow,
    InvalidLength {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidToken {
        position: usize,
        id: usize,
        vocab_size: usize,
    },
    InvalidEpsilon,
    NonFinite {
        buffer: &'static str,
        index: usize,
    },
    AllMasked {
        row: usize,
    },
}

fn matrix_len(rows: usize, width: usize) -> Result<usize, OpError> {
    if rows == 0 || width == 0 {
        return Err(OpError::ZeroDimension);
    }
    rows.checked_mul(width).ok_or(OpError::DimensionOverflow)
}

fn check_len(buffer: &'static str, actual: usize, expected: usize) -> Result<(), OpError> {
    if actual != expected {
        return Err(OpError::InvalidLength {
            buffer,
            expected,
            actual,
        });
    }
    Ok(())
}

fn check_finite(buffer: &'static str, values: &[f32]) -> Result<(), OpError> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(OpError::NonFinite { buffer, index });
        }
    }
    Ok(())
}

/// Gathers `table[token_ids[t], i]` into `y[t, i]`.
///
/// `table[vocab_size, width]` and `y[token_ids.len(), width]` are row-major.
/// Rejects zero dimensions, product overflow, inexact lengths, and invalid IDs.
/// All IDs are validated before copying any row. Values are copied unchanged.
pub fn embedding(
    token_ids: &[usize],
    table: &[f32],
    y: &mut [f32],
    vocab_size: usize,
    width: usize,
) -> Result<(), OpError> {
    let table_len = matrix_len(vocab_size, width)?;
    let output_len = matrix_len(token_ids.len(), width)?;
    check_len("table", table.len(), table_len)?;
    check_len("y", y.len(), output_len)?;
    for (position, &id) in token_ids.iter().enumerate() {
        if id >= vocab_size {
            return Err(OpError::InvalidToken {
                position,
                id,
                vocab_size,
            });
        }
    }
    for (&id, output) in token_ids.iter().zip(y.chunks_exact_mut(width)) {
        output.copy_from_slice(&table[id * width..(id + 1) * width]);
    }
    Ok(())
}

/// Adds matching elements: `y[i] = x[i] + residual[i]`.
///
/// Nonempty slices must have equal lengths and use the same logical layout;
/// there is no broadcasting. Values follow ordinary FP32 arithmetic semantics.
pub fn residual_add(x: &[f32], residual: &[f32], y: &mut [f32]) -> Result<(), OpError> {
    if x.is_empty() {
        return Err(OpError::ZeroDimension);
    }
    check_len("residual", residual.len(), x.len())?;
    check_len("y", y.len(), x.len())?;
    for ((&a, &b), output) in x.iter().zip(residual).zip(y) {
        *output = a + b;
    }
    Ok(())
}

/// Applies RMSNorm independently to each row of `x[rows, width]`.
///
/// `y[t, i] = x[t, i] / sqrt(mean_j(x[t, j]^2) + epsilon) * weight[i]`.
/// `weight[width]` scales features; there is no bias or mean subtraction.
/// Requires positive finite epsilon and finite input/weights. Shapes must be
/// nonzero, products checked, and lengths exact. Output is row-major.
///
/// A row scale keeps squared values bounded, avoiding intermediate overflow
/// for large finite inputs. All arithmetic remains FP32; the final weight
/// multiplication can still overflow under ordinary arithmetic semantics.
pub fn rmsnorm(
    x: &[f32],
    weight: &[f32],
    y: &mut [f32],
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<(), OpError> {
    let len = matrix_len(rows, width)?;
    check_len("x", x.len(), len)?;
    check_len("weight", weight.len(), width)?;
    check_len("y", y.len(), len)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OpError::InvalidEpsilon);
    }
    check_finite("x", x)?;
    check_finite("weight", weight)?;

    let sqrt_epsilon = epsilon.sqrt();
    for (row, output) in x.chunks_exact(width).zip(y.chunks_exact_mut(width)) {
        let scale = row.iter().fold(sqrt_epsilon, |max, x| max.max(x.abs()));
        let mut sum_squares = 0.0_f32;
        for &value in row {
            let scaled = value / scale;
            sum_squares += scaled * scaled;
        }
        let scaled_epsilon = sqrt_epsilon / scale;
        let denominator = (sum_squares / width as f32 + scaled_epsilon * scaled_epsilon).sqrt();
        for ((&value, &gain), out) in row.iter().zip(weight).zip(output) {
            *out = (value / scale) / denominator * gain;
        }
    }
    Ok(())
}

/// Applies `SiLU(x) = x * sigmoid(x)` elementwise to nonempty finite input.
///
/// Exact input/output lengths are required. The negative branch uses `exp(x)`
/// rather than `exp(-x)` so large negative values do not overflow the exponential.
pub fn silu(x: &[f32], y: &mut [f32]) -> Result<(), OpError> {
    if x.is_empty() {
        return Err(OpError::ZeroDimension);
    }
    check_len("y", y.len(), x.len())?;
    check_finite("x", x)?;
    for (&value, output) in x.iter().zip(y) {
        *output = if value >= 0.0 {
            value / (1.0 + (-value).exp())
        } else {
            let exponential = value.exp();
            value * exponential / (1.0 + exponential)
        };
    }
    Ok(())
}

/// Row-wise softmax of `logits[rows, width]`, with row-major output.
///
/// Requires nonzero dimensions, checked products, exact lengths, and finite
/// logits. Subtracts each row's maximum before exponentiation.
pub fn softmax(logits: &[f32], y: &mut [f32], rows: usize, width: usize) -> Result<(), OpError> {
    softmax_impl(logits, None, y, rows, width)
}

/// Row-wise softmax over entries where the matching row-major mask is `true`.
///
/// `mask` has the same shape and exact length as logits. Masked output entries
/// are zero; masked logit values are ignored, even when non-finite. Allowed
/// logits must be finite. Rejects an all-masked row before any output writes.
pub fn masked_softmax(
    logits: &[f32],
    mask: &[bool],
    y: &mut [f32],
    rows: usize,
    width: usize,
) -> Result<(), OpError> {
    softmax_impl(logits, Some(mask), y, rows, width)
}

fn softmax_impl(
    logits: &[f32],
    mask: Option<&[bool]>,
    y: &mut [f32],
    rows: usize,
    width: usize,
) -> Result<(), OpError> {
    let len = matrix_len(rows, width)?;
    check_len("logits", logits.len(), len)?;
    check_len("y", y.len(), len)?;
    if let Some(mask) = mask {
        check_len("mask", mask.len(), len)?;
    }

    // Check every row first: a bad later row must not leave earlier output written.
    for (row_index, row) in logits.chunks_exact(width).enumerate() {
        let mut has_allowed_entry = false;
        for (i, value) in row.iter().enumerate() {
            let index = row_index * width + i;
            if mask.is_none_or(|mask| mask[index]) {
                has_allowed_entry = true;
                if !value.is_finite() {
                    return Err(OpError::NonFinite {
                        buffer: "logits",
                        index,
                    });
                }
            }
        }
        if !has_allowed_entry {
            return Err(OpError::AllMasked { row: row_index });
        }
    }

    for (row_index, (row, output)) in logits
        .chunks_exact(width)
        .zip(y.chunks_exact_mut(width))
        .enumerate()
    {
        let mut max = f32::NEG_INFINITY;
        for (i, &value) in row.iter().enumerate() {
            if mask.is_none_or(|mask| mask[row_index * width + i]) {
                max = max.max(value);
            }
        }
        let mut sum = 0.0_f32;
        for (i, (&value, out)) in row.iter().zip(output.iter_mut()).enumerate() {
            *out = if mask.is_none_or(|mask| mask[row_index * width + i]) {
                (value - max).exp()
            } else {
                0.0
            };
            sum += *out;
        }
        for out in output {
            *out /= sum;
        }
    }
    Ok(())
}
