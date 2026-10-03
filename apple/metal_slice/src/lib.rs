//! T27: one explicit Metal operation, outside the scalar engine.
//! Finite FP32 inputs/output; no decoder or backend selection is implied.
#![deny(unsafe_op_in_unsafe_fn)]

pub const MAX_ELEMENTS: usize = 1 << 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    ZeroLength,
    DimensionOverflow,
    TooLarge,
    InvalidLength {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        buffer: &'static str,
        index: usize,
    },
    Unavailable(&'static str),
    Gpu(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

/// Validate storage arithmetic before resource creation. This small experiment
/// bounds three FP32 buffers to 12 MiB total; that is not measured process RAM.
pub fn buffer_bytes(capacity: usize) -> Result<usize, Error> {
    if capacity == 0 {
        return Err(Error::ZeroLength);
    }
    let bytes = capacity
        .checked_mul(size_of::<f32>())
        .ok_or(Error::DimensionOverflow)?;
    if capacity > MAX_ELEMENTS || u32::try_from(capacity).is_err() {
        return Err(Error::TooLarge);
    }
    Ok(bytes)
}

#[cfg(any(target_os = "macos", test))]
fn validate(x: &[f32], residual: &[f32], output_len: usize, capacity: usize) -> Result<(), Error> {
    buffer_bytes(x.len())?;
    if x.len() > capacity {
        return Err(Error::TooLarge);
    }
    for (buffer, actual) in [("residual", residual.len()), ("y", output_len)] {
        if actual != x.len() {
            return Err(Error::InvalidLength {
                buffer,
                expected: x.len(),
                actual,
            });
        }
    }
    for (buffer, values) in [("x", x), ("residual", residual)] {
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFinite { buffer, index });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::{GpuTiming, ResidualAdd};

#[cfg(not(target_os = "macos"))]
pub struct ResidualAdd;
#[cfg(not(target_os = "macos"))]
impl ResidualAdd {
    pub fn new(capacity: usize) -> Result<Self, Error> {
        buffer_bytes(capacity)?;
        Err(Error::Unavailable("Metal slice requires macOS"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_storage_and_bounds() {
        assert_eq!(buffer_bytes(0), Err(Error::ZeroLength));
        assert_eq!(buffer_bytes(usize::MAX), Err(Error::DimensionOverflow));
        assert_eq!(buffer_bytes(MAX_ELEMENTS + 1), Err(Error::TooLarge));
        assert_eq!(buffer_bytes(MAX_ELEMENTS), Ok(4 * MAX_ELEMENTS));
    }
    #[test]
    fn preflight_is_portable() {
        assert_eq!(validate(&[], &[], 0, 32), Err(Error::ZeroLength));
        assert_eq!(
            validate(&[1.0; 33], &[2.0; 33], 33, 32),
            Err(Error::TooLarge)
        );
        assert_eq!(
            validate(&[1.0; 2], &[2.0], 2, 32),
            Err(Error::InvalidLength {
                buffer: "residual",
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            validate(&[1.0], &[2.0], 2, 32),
            Err(Error::InvalidLength {
                buffer: "y",
                expected: 1,
                actual: 2
            })
        );
        assert_eq!(
            validate(&[f32::NAN], &[1.0], 1, 32),
            Err(Error::NonFinite {
                buffer: "x",
                index: 0
            })
        );
        assert_eq!(
            validate(&[1.0], &[f32::INFINITY], 1, 32),
            Err(Error::NonFinite {
                buffer: "residual",
                index: 0
            })
        );
        assert_eq!(validate(&[1.0, -2.0], &[3.0, 4.0], 2, 32), Ok(()));
    }
}
