//! Rejected development-only prototype; never called by the production decoder.
use llm_inference::LinearError;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkError {
    InvalidChunkSize,
    Linear(LinearError),
}
/// Projection-only prefill experiment. Reuses each contiguous weight group
/// across up to eight token rows before advancing; does not populate a KV cache
/// or replace the serial model prefill. Identical lane/tail reduction order to
/// `linear_lanes4`. Chunk size 1..=8 bounds stack scratch to 128 bytes.
/// All shape/chunk validation precedes output writes; no allocation occurs.
#[inline(never)]
pub fn linear_chunked(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    tokens: usize,
    input_dim: usize,
    output_dim: usize,
    chunk_size: usize,
) -> Result<(), ChunkError> {
    if !(1..=8).contains(&chunk_size) {
        return Err(ChunkError::InvalidChunkSize);
    }
    validate(x, w, y, tokens, input_dim, output_dim).map_err(ChunkError::Linear)?;
    for start in (0..tokens).step_by(chunk_size) {
        let count = (tokens - start).min(chunk_size);
        for (o, row) in w.chunks_exact(input_dim).enumerate() {
            let mut sums = [[0.0_f32; 4]; 8];
            let bulk = input_dim - input_dim % 4;
            for i in (0..bulk).step_by(4) {
                let weights = &row[i..i + 4];
                for (t, lanes) in sums[..count].iter_mut().enumerate() {
                    let input = &x[(start + t) * input_dim + i..(start + t) * input_dim + i + 4];
                    lanes[0] += input[0] * weights[0];
                    lanes[1] += input[1] * weights[1];
                    lanes[2] += input[2] * weights[2];
                    lanes[3] += input[3] * weights[3];
                }
            }
            for (t, lanes) in sums[..count].iter().enumerate() {
                let mut sum = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3];
                for i in bulk..input_dim {
                    sum += x[(start + t) * input_dim + i] * row[i];
                }
                y[(start + t) * output_dim + o] = sum;
            }
        }
    }
    Ok(())
}

fn validate(
    x: &[f32],
    w: &[f32],
    y: &[f32],
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
    Ok(())
}
