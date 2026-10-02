//! Groupwise symmetric weight-only INT8 teaching representation, not yet a model path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantError {
    InvalidGroupSize,
    InvalidShape {
        expected_rows: usize,
        expected_width: usize,
        actual_rows: usize,
        actual_width: usize,
    },
    ZeroDimension,
    DimensionOverflow,
    InvalidLength {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteWeight {
        index: usize,
    },
    UnrepresentableScale,
    Allocation,
}
/// Row-major weights `[output_dim,input_dim]`. Groups never cross a row.
/// Scale layout `[output_dim,ceil(input_dim/group_size)]` is also row-major.
/// Private fields keep dimension/scale/value invariants construction-bound.
pub struct GroupwiseInt8 {
    output_dim: usize,
    input_dim: usize,
    group_size: usize,
    groups_per_row: usize,
    values: Vec<i8>,
    scales: Vec<f32>,
    logical_bytes: usize,
}
fn encode(weight: f32, scale: f32) -> i8 {
    // Construction guarantees finite weight and positive finite scale.
    // round() uses nearest with ties away from zero; -128 is never encoded.
    (weight / scale).round().clamp(-127.0, 127.0) as i8
}
fn reserve<T>(len: usize) -> Result<Vec<T>, QuantError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| QuantError::Allocation)?;
    Ok(v)
}
impl GroupwiseInt8 {
    /// Validates dimensions, checked payload arithmetic, exact input and all
    /// finite weights before allocation. Group size1..256; default experiment32.
    /// scale=max(abs(group))/127; zero groups use scale1 and encoded zeros.
    /// A nonzero group whose FP32 scale underflows, or whose reconstruction
    /// overflows, is rejected explicitly rather than silently changing precision.
    /// Positive subnormal scales are admitted, with coarse FP32 rounding that
    /// can activate clamping; normal-scale relative error bounds do not apply.
    pub fn quantize(
        weights: &[f32],
        output_dim: usize,
        input_dim: usize,
        group_size: usize,
    ) -> Result<Self, QuantError> {
        if !(1..=256).contains(&group_size) {
            return Err(QuantError::InvalidGroupSize);
        }
        if output_dim == 0 || input_dim == 0 {
            return Err(QuantError::ZeroDimension);
        }
        let len = output_dim
            .checked_mul(input_dim)
            .ok_or(QuantError::DimensionOverflow)?;
        let groups_per_row = input_dim.div_ceil(group_size);
        let scale_count = output_dim
            .checked_mul(groups_per_row)
            .ok_or(QuantError::DimensionOverflow)?;
        let logical_bytes = scale_count
            .checked_mul(size_of::<f32>())
            .and_then(|n| n.checked_add(len))
            .ok_or(QuantError::DimensionOverflow)?;
        if weights.len() != len {
            return Err(QuantError::InvalidLength {
                buffer: "weights",
                expected: len,
                actual: weights.len(),
            });
        }
        if let Some(index) = weights.iter().position(|w| !w.is_finite()) {
            return Err(QuantError::NonFiniteWeight { index });
        }
        let mut values = reserve(len)?;
        let mut scales = reserve(scale_count)?;
        for row in weights.chunks_exact(input_dim) {
            for group in row.chunks(group_size) {
                let maximum = group.iter().fold(0.0_f32, |m, w| m.max(w.abs()));
                let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(QuantError::UnrepresentableScale);
                }
                scales.push(scale);
                for &w in group {
                    let q = encode(w, scale);
                    if !(f32::from(q) * scale).is_finite() {
                        return Err(QuantError::UnrepresentableScale);
                    }
                    values.push(q);
                }
            }
        }
        Ok(Self {
            output_dim,
            input_dim,
            group_size,
            groups_per_row,
            values,
            scales,
            logical_bytes,
        })
    }
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    pub fn group_size(&self) -> usize {
        self.group_size
    }
    pub fn groups_per_row(&self) -> usize {
        self.groups_per_row
    }
    pub fn values(&self) -> &[i8] {
        &self.values
    }
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }
    pub fn logical_bytes(&self) -> usize {
        self.logical_bytes
    }
    /// Explicit reconstruction reference. Caller owns full FP32 output; the
    /// fused candidate below never invokes this per token. Invalid length leaves
    /// output unchanged. Construction already qualified all reconstructed values.
    pub fn dequantize(&self, output: &mut [f32]) -> Result<(), QuantError> {
        if output.len() != self.values.len() {
            return Err(QuantError::InvalidLength {
                buffer: "output",
                expected: self.values.len(),
                actual: output.len(),
            });
        }
        for (o, (row, out)) in self
            .values
            .chunks_exact(self.input_dim)
            .zip(output.chunks_exact_mut(self.input_dim))
            .enumerate()
        {
            for (i, (&q, value)) in row.iter().zip(out).enumerate() {
                *value = f32::from(q) * self.scales[o * self.groups_per_row + i / self.group_size];
            }
        }
        Ok(())
    }
}
/// FP32 activations and accumulation. Each encoded weight is reconstructed in
/// a register just before multiplication, not in a full FP32 matrix. Ordinary
/// FP32 numerical semantics match scalar linear on the dequantized reference:
/// multiply q*scale first, then activation, add in increasing input-index order.
/// Exact shape admission precedes writes; no allocation, finite-input validation,
/// parallelism, SIMD intrinsics or bias. Overflow/NaN handling belongs to callers.
pub fn linear_int8(
    x: &[f32],
    weights: &GroupwiseInt8,
    y: &mut [f32],
    tokens: usize,
) -> Result<(), QuantError> {
    if tokens == 0 {
        return Err(QuantError::ZeroDimension);
    }
    let x_len = tokens
        .checked_mul(weights.input_dim)
        .ok_or(QuantError::DimensionOverflow)?;
    let y_len = tokens
        .checked_mul(weights.output_dim)
        .ok_or(QuantError::DimensionOverflow)?;
    for (buffer, expected, actual) in [("x", x_len, x.len()), ("y", y_len, y.len())] {
        if actual != expected {
            return Err(QuantError::InvalidLength {
                buffer,
                expected,
                actual,
            });
        }
    }
    for (xrow, yrow) in x
        .chunks_exact(weights.input_dim)
        .zip(y.chunks_exact_mut(weights.output_dim))
    {
        for (o, (row, out)) in weights
            .values
            .chunks_exact(weights.input_dim)
            .zip(yrow)
            .enumerate()
        {
            let mut sum = 0.0_f32;
            for (group, values) in row.chunks(weights.group_size).enumerate() {
                let scale = weights.scales[o * weights.groups_per_row + group];
                let start = group * weights.group_size;
                for (i, &q) in values.iter().enumerate() {
                    sum += xrow[start + i] * (f32::from(q) * scale);
                }
            }
            *out = sum;
        }
    }
    Ok(())
}
/// Four-lane fused candidate for tied vocabulary projection. Same reduction
/// order as linear_lanes4 on once-dequantized weights, including arbitrary
/// group boundaries and row tails. Original scalar linear_int8 remains callable.
pub fn linear_int8_lanes4(
    x: &[f32],
    weights: &GroupwiseInt8,
    y: &mut [f32],
    tokens: usize,
) -> Result<(), QuantError> {
    if tokens == 0 {
        return Err(QuantError::ZeroDimension);
    }
    let x_len = tokens
        .checked_mul(weights.input_dim)
        .ok_or(QuantError::DimensionOverflow)?;
    let y_len = tokens
        .checked_mul(weights.output_dim)
        .ok_or(QuantError::DimensionOverflow)?;
    for (buffer, expected, actual) in [("x", x_len, x.len()), ("y", y_len, y.len())] {
        if actual != expected {
            return Err(QuantError::InvalidLength {
                buffer,
                expected,
                actual,
            });
        }
    }
    for (xrow, yrow) in x
        .chunks_exact(weights.input_dim)
        .zip(y.chunks_exact_mut(weights.output_dim))
    {
        for (o, (row, out)) in weights
            .values
            .chunks_exact(weights.input_dim)
            .zip(yrow)
            .enumerate()
        {
            let mut sums = [0.0_f32; 4];
            let scales =
                &weights.scales[o * weights.groups_per_row..(o + 1) * weights.groups_per_row];
            let bulk = weights.input_dim - weights.input_dim % 4;
            if weights.group_size.is_multiple_of(4) {
                for (g, values) in row.chunks(weights.group_size).enumerate() {
                    let scale = scales[g];
                    let start = g * weights.group_size;
                    for (chunk, q) in values.chunks_exact(4).enumerate() {
                        let a = &xrow[start + chunk * 4..start + chunk * 4 + 4];
                        sums[0] += a[0] * (f32::from(q[0]) * scale);
                        sums[1] += a[1] * (f32::from(q[1]) * scale);
                        sums[2] += a[2] * (f32::from(q[2]) * scale);
                        sums[3] += a[3] * (f32::from(q[3]) * scale);
                    }
                }
            } else {
                for i in 0..bulk {
                    sums[i % 4] += xrow[i] * (f32::from(row[i]) * scales[i / weights.group_size]);
                }
            }
            let mut sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
            for i in bulk..weights.input_dim {
                sum += xrow[i] * (f32::from(row[i]) * scales[i / weights.group_size]);
            }
            *out = sum;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::encode;
    #[test]
    fn nearest_away_from_zero_and_symmetric_clamping_are_independent_constants() {
        for (w, q) in [
            (-400., -127),
            (-127.5, -127),
            (-0.5, -1),
            (0., 0),
            (0.5, 1),
            (127.5, 127),
            (400., 127),
        ] {
            assert_eq!(encode(w, 1.), q);
        }
    }
}
