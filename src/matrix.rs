//! FP32 decoder weights and a feature-gated rejected INT8 development experiment.
use crate::kernels::linear_lanes4;
use crate::ops::{OpError, embedding};
use crate::quantization::QuantError;
#[cfg(feature = "int8-experiment")]
use crate::quantization::{GroupwiseInt8, linear_int8};
use crate::{LinearError, linear};
use serde::{Deserialize, Deserializer};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionError {
    Float(LinearError),
    Quantized(QuantError),
}
pub enum WeightMatrix {
    Float(Vec<f32>),
    #[cfg(feature = "int8-experiment")]
    Int8(GroupwiseInt8),
}
#[derive(Clone, Copy)]
pub enum MatrixRef<'a> {
    Float(&'a [f32]),
    #[cfg(feature = "int8-experiment")]
    Int8(&'a GroupwiseInt8),
}
impl<'de> Deserialize<'de> for WeightMatrix {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<f32>::deserialize(deserializer).map(Self::Float)
    }
}
impl From<Vec<f32>> for WeightMatrix {
    fn from(values: Vec<f32>) -> Self {
        Self::Float(values)
    }
}
impl<'a> From<&'a WeightMatrix> for MatrixRef<'a> {
    fn from(w: &'a WeightMatrix) -> Self {
        w.view()
    }
}
impl<'a> From<&'a Vec<f32>> for MatrixRef<'a> {
    fn from(w: &'a Vec<f32>) -> Self {
        Self::Float(w)
    }
}
impl<'a> From<&'a [f32]> for MatrixRef<'a> {
    fn from(w: &'a [f32]) -> Self {
        Self::Float(w)
    }
}
impl<'a, const N: usize> From<&'a [f32; N]> for MatrixRef<'a> {
    fn from(w: &'a [f32; N]) -> Self {
        Self::Float(w)
    }
}
impl WeightMatrix {
    pub fn view(&self) -> MatrixRef<'_> {
        match self {
            Self::Float(w) => MatrixRef::Float(w),
            #[cfg(feature = "int8-experiment")]
            Self::Int8(w) => MatrixRef::Int8(w),
        }
    }
    pub fn len(&self) -> usize {
        self.view().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn logical_bytes(&self) -> Option<usize> {
        match self {
            Self::Float(w) => w.len().checked_mul(4),
            #[cfg(feature = "int8-experiment")]
            Self::Int8(w) => Some(w.logical_bytes()),
        }
    }
    #[cfg(feature = "int8-experiment")]
    pub fn into_int8(self, rows: usize, cols: usize, group: usize) -> Result<Self, QuantError> {
        match self {
            Self::Float(w) => GroupwiseInt8::quantize(&w, rows, cols, group).map(Self::Int8),
            #[cfg(feature = "int8-experiment")]
            Self::Int8(w) => {
                if group != w.group_size() {
                    return Err(QuantError::InvalidGroupSize);
                }
                if rows != w.output_dim() || cols != w.input_dim() {
                    return Err(QuantError::InvalidShape {
                        expected_rows: rows,
                        expected_width: cols,
                        actual_rows: w.output_dim(),
                        actual_width: w.input_dim(),
                    });
                }
                Ok(Self::Int8(w))
            }
        }
    }
    #[cfg(feature = "int8-experiment")]
    pub fn into_float(self) -> Result<Self, QuantError> {
        match self {
            Self::Float(w) => Ok(Self::Float(w)),
            #[cfg(feature = "int8-experiment")]
            Self::Int8(w) => {
                let mut v = Vec::new();
                v.try_reserve_exact(w.values().len())
                    .map_err(|_| QuantError::Allocation)?;
                v.resize(w.values().len(), 0.);
                w.dequantize(&mut v)?;
                Ok(Self::Float(v))
            }
        }
    }
}
impl MatrixRef<'_> {
    pub fn len(self) -> usize {
        match self {
            Self::Float(w) => w.len(),
            #[cfg(feature = "int8-experiment")]
            Self::Int8(w) => w.values().len(),
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub(crate) fn check_shape(self, rows: usize, width: usize) -> Result<(), QuantError> {
        let _ = (rows, width);
        match self {
            Self::Float(_) => Ok(()),
            #[cfg(feature = "int8-experiment")]
            Self::Int8(w) => {
                if w.output_dim() == rows && w.input_dim() == width {
                    Ok(())
                } else {
                    Err(QuantError::InvalidShape {
                        expected_rows: rows,
                        expected_width: width,
                        actual_rows: w.output_dim(),
                        actual_width: w.input_dim(),
                    })
                }
            }
        }
    }
    pub(crate) fn matches_shape(self, rows: usize, width: usize) -> bool {
        self.check_shape(rows, width).is_ok()
    }
    pub(crate) fn nonfinite_index(self) -> Option<usize> {
        match self {
            Self::Float(w) => w.iter().position(|v| !v.is_finite()),
            #[cfg(feature = "int8-experiment")]
            Self::Int8(_) => None,
        }
    }
    pub fn project(
        self,
        x: &[f32],
        y: &mut [f32],
        tokens: usize,
        input: usize,
        output: usize,
        lanes: bool,
    ) -> Result<(), ProjectionError> {
        match self {
            Self::Float(w) => if lanes {
                linear_lanes4(x, w, y, tokens, input, output)
            } else {
                linear(x, w, y, tokens, input, output)
            }
            .map_err(ProjectionError::Float),
            #[cfg(feature = "int8-experiment")]
            Self::Int8(w) => {
                if !self.matches_shape(output, input) {
                    return Err(ProjectionError::Quantized(QuantError::InvalidShape {
                        expected_rows: output,
                        expected_width: input,
                        actual_rows: w.output_dim(),
                        actual_width: w.input_dim(),
                    }));
                }
                if lanes {
                    crate::quantization::linear_int8_lanes4(x, w, y, tokens)
                } else {
                    linear_int8(x, w, y, tokens)
                }
                .map_err(ProjectionError::Quantized)
            }
        }
    }
    pub fn gather(
        self,
        ids: &[usize],
        y: &mut [f32],
        rows: usize,
        width: usize,
    ) -> Result<(), OpError> {
        match self {
            Self::Float(w) => embedding(ids, w, y, rows, width),
            #[cfg(feature = "int8-experiment")]
            Self::Int8(w) => {
                if ids.is_empty() || rows == 0 || width == 0 {
                    return Err(OpError::ZeroDimension);
                }
                let len = ids
                    .len()
                    .checked_mul(width)
                    .ok_or(OpError::DimensionOverflow)?;
                rows.checked_mul(width).ok_or(OpError::DimensionOverflow)?;
                if !self.matches_shape(rows, width) {
                    return Err(OpError::InvalidShape {
                        expected_rows: rows,
                        expected_width: width,
                        actual_rows: w.output_dim(),
                        actual_width: w.input_dim(),
                    });
                }
                if y.len() != len {
                    return Err(OpError::InvalidLength {
                        buffer: "y",
                        expected: len,
                        actual: y.len(),
                    });
                }
                for (position, &id) in ids.iter().enumerate() {
                    if id >= rows {
                        return Err(OpError::InvalidToken {
                            position,
                            id,
                            vocab_size: rows,
                        });
                    }
                }
                for (&id, out) in ids.iter().zip(y.chunks_exact_mut(width)) {
                    for (i, v) in out.iter_mut().enumerate() {
                        *v = f32::from(w.values()[id * width + i])
                            * w.scales()[id * w.groups_per_row() + i / w.group_size()];
                    }
                }
                Ok(())
            }
        }
    }
}
