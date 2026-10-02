//! Rotary positions with split-half pairing and explicit absolute positions.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeError {
    ZeroDimension,
    OddHeadDimension,
    DimensionOverflow,
    PositionOverflow,
    PositionOutOfRange,
    InvalidBase,
    InvalidAngle,
    InvalidLength {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        index: usize,
    },
}

/// Rotates `x[tokens, heads, head_dim]` into the identically shaped `y`.
///
/// Storage is row-major, offset `(t * heads + h) * head_dim + i`.
/// Split-half pairing couples feature `j` with `j + head_dim/2`:
/// `theta = (start_position + t) * base^(-2*j/head_dim)`;
/// `(a,b)` becomes `(a*cos(theta)-b*sin(theta), b*cos(theta)+a*sin(theta))`.
/// This convention is explicit; interleaved pairs are not accepted implicitly.
///
/// Requires nonzero dimensions, an even head dimension, exact lengths, finite
/// inputs, and a positive finite base. Position arithmetic is checked, and the
/// last position must be at most 2^24 so every integer position is exact in FP32.
/// Unrepresentable angles are rejected. All checks precede output writes.
/// Ordinary FP32 rounding and final arithmetic overflow still apply.
///
/// No buffers are allocated. Sine/cosine values are shared across heads at a
/// position/frequency; the scalar implementation remains a correctness reference.
pub fn rope(
    x: &[f32],
    y: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    start_position: usize,
    base: f32,
) -> Result<(), RopeError> {
    if tokens == 0 || heads == 0 || head_dim == 0 {
        return Err(RopeError::ZeroDimension);
    }
    if !head_dim.is_multiple_of(2) {
        return Err(RopeError::OddHeadDimension);
    }
    let row_len = heads
        .checked_mul(head_dim)
        .ok_or(RopeError::DimensionOverflow)?;
    let len = tokens
        .checked_mul(row_len)
        .ok_or(RopeError::DimensionOverflow)?;
    let last_position = start_position
        .checked_add(tokens - 1)
        .ok_or(RopeError::PositionOverflow)?;
    if last_position > (1 << 24) {
        return Err(RopeError::PositionOutOfRange);
    }
    if !base.is_finite() || base <= 0.0 {
        return Err(RopeError::InvalidBase);
    }
    for (buffer, actual) in [("x", x.len()), ("y", y.len())] {
        if actual != len {
            return Err(RopeError::InvalidLength {
                buffer,
                expected: len,
                actual,
            });
        }
    }
    for (index, value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(RopeError::NonFinite { index });
        }
    }
    let half = head_dim / 2;
    for j in 0..half {
        let frequency = base.powf(-((2 * j) as f32) / head_dim as f32);
        if !frequency.is_finite() || !(last_position as f32 * frequency).is_finite() {
            return Err(RopeError::InvalidAngle);
        }
    }

    for t in 0..tokens {
        let position = (start_position + t) as f32;
        for j in 0..half {
            let frequency = base.powf(-((2 * j) as f32) / head_dim as f32);
            let (sine, cosine) = (position * frequency).sin_cos();
            for h in 0..heads {
                let offset = t * row_len + h * head_dim;
                let a = x[offset + j];
                let b = x[offset + j + half];
                y[offset + j] = a * cosine - b * sine;
                y[offset + j + half] = b * cosine + a * sine;
            }
        }
    }
    Ok(())
}
