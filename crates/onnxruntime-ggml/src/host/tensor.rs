//! A dense row-major tensor in host memory.

use std::borrow::Cow;

use crate::error::{Error, Result};
use crate::ir::DType;

#[derive(Clone, Debug, PartialEq)]
pub enum Data {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    I8(Vec<i8>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Data,
}

pub fn numel_of(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl HostTensor {
    pub fn new(shape: Vec<usize>, data: Data) -> Result<HostTensor> {
        let t = HostTensor { shape, data };
        if t.data_len() != numel_of(&t.shape) {
            return Err(Error::shape(format!(
                "tensor data has {} elements but shape {:?} wants {}",
                t.data_len(),
                t.shape,
                numel_of(&t.shape)
            )));
        }
        Ok(t)
    }

    pub fn f32(shape: Vec<usize>, data: Vec<f32>) -> HostTensor {
        debug_assert_eq!(numel_of(&shape), data.len());
        HostTensor { shape, data: Data::F32(data) }
    }
    pub fn i64(shape: Vec<usize>, data: Vec<i64>) -> HostTensor {
        debug_assert_eq!(numel_of(&shape), data.len());
        HostTensor { shape, data: Data::I64(data) }
    }
    pub fn i32(shape: Vec<usize>, data: Vec<i32>) -> HostTensor {
        HostTensor { shape, data: Data::I32(data) }
    }
    pub fn bool(shape: Vec<usize>, data: Vec<bool>) -> HostTensor {
        HostTensor { shape, data: Data::Bool(data) }
    }
    pub fn scalar_f32(v: f32) -> HostTensor {
        HostTensor::f32(vec![], vec![v])
    }
    pub fn const_i64(v: i64) -> HostTensor {
        HostTensor::i64(vec![], vec![v])
    }
    pub fn zeros(dtype: DType, shape: Vec<usize>) -> HostTensor {
        let n = numel_of(&shape);
        let data = match dtype {
            DType::F32 | DType::F16 => Data::F32(vec![0.0; n]),
            DType::F64 => Data::F64(vec![0.0; n]),
            DType::I64 => Data::I64(vec![0; n]),
            DType::I32 => Data::I32(vec![0; n]),
            DType::I8 => Data::I8(vec![0; n]),
            DType::U8 => Data::U8(vec![0; n]),
            DType::Bool => Data::Bool(vec![false; n]),
        };
        HostTensor { shape, data }
    }

    /// Build from raw bytes in onnxruntime's layout. f16 is widened to f32 on
    /// the way in; nothing downstream computes in half precision.
    pub fn from_bytes(dtype: DType, shape: Vec<usize>, bytes: &[u8]) -> Result<HostTensor> {
        let n = numel_of(&shape);
        let need = n * dtype.size();
        if bytes.len() < need {
            return Err(Error::shape(format!("{} bytes for {dtype} {shape:?}, need {need}", bytes.len())));
        }
        let bytes = &bytes[..need];
        let data = match dtype {
            DType::F32 => Data::F32(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            DType::F16 => Data::F32(bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()),
            DType::F64 => Data::F64(
                bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect(),
            ),
            DType::I64 => Data::I64(
                bytes.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect(),
            ),
            DType::I32 => Data::I32(bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            DType::I8 => Data::I8(bytes.iter().map(|&b| b as i8).collect()),
            DType::U8 => Data::U8(bytes.to_vec()),
            DType::Bool => Data::Bool(bytes.iter().map(|&b| b != 0).collect()),
        };
        Ok(HostTensor { shape, data })
    }

    /// Raw bytes in the given element type, for handing back to onnxruntime.
    pub fn to_bytes(&self, dtype: DType) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.numel() * dtype.size());
        match dtype {
            DType::F32 => {
                // one memcpy on little-endian targets, which is every target we build for
                let v = self.as_f32();
                out.extend_from_slice(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) });
            }
            DType::F16 => {
                for v in self.as_f32().iter() {
                    out.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                }
            }
            DType::F64 => {
                for v in self.as_f64().iter() {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            DType::I64 => {
                for v in self.as_i64().iter() {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            DType::I32 => {
                for v in self.as_i64().iter() {
                    out.extend_from_slice(&(*v as i32).to_le_bytes());
                }
            }
            DType::I8 => out.extend(self.as_i64().iter().map(|v| *v as i8 as u8)),
            DType::U8 => out.extend(self.as_i64().iter().map(|v| *v as u8)),
            DType::Bool => out.extend(self.as_bool().iter().map(|v| *v as u8)),
        }
        Ok(out)
    }

    pub fn dtype(&self) -> DType {
        match &self.data {
            Data::F32(_) => DType::F32,
            Data::F64(_) => DType::F64,
            Data::I64(_) => DType::I64,
            Data::I32(_) => DType::I32,
            Data::I8(_) => DType::I8,
            Data::U8(_) => DType::U8,
            Data::Bool(_) => DType::Bool,
        }
    }

    pub fn data_len(&self) -> usize {
        match &self.data {
            Data::F32(v) => v.len(),
            Data::F64(v) => v.len(),
            Data::I64(v) => v.len(),
            Data::I32(v) => v.len(),
            Data::I8(v) => v.len(),
            Data::U8(v) => v.len(),
            Data::Bool(v) => v.len(),
        }
    }

    pub fn numel(&self) -> usize {
        numel_of(&self.shape)
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn strides(&self) -> Vec<usize> {
        strides_of(&self.shape)
    }

    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype().size()
    }

    /// Every element as f32 (borrowed when already f32).
    pub fn as_f32(&self) -> Cow<'_, [f32]> {
        match &self.data {
            Data::F32(v) => Cow::Borrowed(v.as_slice()),
            Data::F64(v) => Cow::Owned(v.iter().map(|&x| x as f32).collect()),
            Data::I64(v) => Cow::Owned(v.iter().map(|&x| x as f32).collect()),
            Data::I32(v) => Cow::Owned(v.iter().map(|&x| x as f32).collect()),
            Data::I8(v) => Cow::Owned(v.iter().map(|&x| x as f32).collect()),
            Data::U8(v) => Cow::Owned(v.iter().map(|&x| x as f32).collect()),
            Data::Bool(v) => Cow::Owned(v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect()),
        }
    }

    pub fn as_f64(&self) -> Cow<'_, [f64]> {
        match &self.data {
            Data::F64(v) => Cow::Borrowed(v.as_slice()),
            Data::F32(v) => Cow::Owned(v.iter().map(|&x| x as f64).collect()),
            Data::I64(v) => Cow::Owned(v.iter().map(|&x| x as f64).collect()),
            Data::I32(v) => Cow::Owned(v.iter().map(|&x| x as f64).collect()),
            Data::I8(v) => Cow::Owned(v.iter().map(|&x| x as f64).collect()),
            Data::U8(v) => Cow::Owned(v.iter().map(|&x| x as f64).collect()),
            Data::Bool(v) => Cow::Owned(v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect()),
        }
    }

    /// Every element as i64 (borrowed when already i64). Floats truncate.
    pub fn as_i64(&self) -> Cow<'_, [i64]> {
        match &self.data {
            Data::I64(v) => Cow::Borrowed(v.as_slice()),
            Data::F32(v) => Cow::Owned(v.iter().map(|&x| x as i64).collect()),
            Data::F64(v) => Cow::Owned(v.iter().map(|&x| x as i64).collect()),
            Data::I32(v) => Cow::Owned(v.iter().map(|&x| x as i64).collect()),
            Data::I8(v) => Cow::Owned(v.iter().map(|&x| x as i64).collect()),
            Data::U8(v) => Cow::Owned(v.iter().map(|&x| x as i64).collect()),
            Data::Bool(v) => Cow::Owned(v.iter().map(|&x| x as i64).collect()),
        }
    }

    pub fn as_bool(&self) -> Cow<'_, [bool]> {
        match &self.data {
            Data::Bool(v) => Cow::Borrowed(v.as_slice()),
            Data::F32(v) => Cow::Owned(v.iter().map(|&x| x != 0.0).collect()),
            Data::F64(v) => Cow::Owned(v.iter().map(|&x| x != 0.0).collect()),
            Data::I64(v) => Cow::Owned(v.iter().map(|&x| x != 0).collect()),
            Data::I32(v) => Cow::Owned(v.iter().map(|&x| x != 0).collect()),
            Data::I8(v) => Cow::Owned(v.iter().map(|&x| x != 0).collect()),
            Data::U8(v) => Cow::Owned(v.iter().map(|&x| x != 0).collect()),
        }
    }

    pub fn scalar_f64(&self) -> Result<f64> {
        if self.numel() != 1 {
            return Err(Error::shape(format!("expected a scalar, got shape {:?}", self.shape)));
        }
        Ok(self.as_f64()[0])
    }

    pub fn scalar_i64(&self) -> Result<i64> {
        if self.numel() != 1 {
            return Err(Error::shape(format!("expected a scalar, got shape {:?}", self.shape)));
        }
        Ok(self.as_i64()[0])
    }

    /// Reinterpret the same elements under a new shape (no data movement).
    pub fn reshaped(&self, shape: Vec<usize>) -> Result<HostTensor> {
        if numel_of(&shape) != self.numel() {
            return Err(Error::shape(format!("cannot reshape {:?} into {:?}", self.shape, shape)));
        }
        Ok(HostTensor { shape, data: self.data.clone() })
    }

    pub fn cast(&self, dtype: DType) -> HostTensor {
        if dtype == self.dtype() {
            return self.clone();
        }
        let data = match dtype {
            DType::F32 | DType::F16 => Data::F32(self.as_f32().into_owned()),
            DType::F64 => Data::F64(self.as_f64().into_owned()),
            DType::I64 => Data::I64(self.as_i64().into_owned()),
            DType::I32 => Data::I32(self.as_i64().iter().map(|&x| x as i32).collect()),
            DType::I8 => Data::I8(self.as_i64().iter().map(|&x| x as i8).collect()),
            DType::U8 => Data::U8(self.as_i64().iter().map(|&x| x as u8).collect()),
            DType::Bool => Data::Bool(self.as_bool().into_owned()),
        };
        HostTensor { shape: self.shape.clone(), data }
    }

    /// Gather elements by flat source index into a new tensor of `shape`.
    pub fn gather_flat(&self, shape: Vec<usize>, index: &[usize]) -> HostTensor {
        let data = match &self.data {
            Data::F32(v) => Data::F32(index.iter().map(|&i| v[i]).collect()),
            Data::F64(v) => Data::F64(index.iter().map(|&i| v[i]).collect()),
            Data::I64(v) => Data::I64(index.iter().map(|&i| v[i]).collect()),
            Data::I32(v) => Data::I32(index.iter().map(|&i| v[i]).collect()),
            Data::I8(v) => Data::I8(index.iter().map(|&i| v[i]).collect()),
            Data::U8(v) => Data::U8(index.iter().map(|&i| v[i]).collect()),
            Data::Bool(v) => Data::Bool(index.iter().map(|&i| v[i]).collect()),
        };
        HostTensor { shape, data }
    }

    /// Short description for log lines: dtype, shape, and the first few values.
    pub fn brief(&self) -> String {
        let n = self.numel();
        let head: Vec<String> = match &self.data {
            Data::F32(v) => v.iter().take(4).map(|x| format!("{x:.4}")).collect(),
            Data::F64(v) => v.iter().take(4).map(|x| format!("{x:.4}")).collect(),
            Data::I64(v) => v.iter().take(4).map(|x| x.to_string()).collect(),
            Data::I32(v) => v.iter().take(4).map(|x| x.to_string()).collect(),
            Data::I8(v) => v.iter().take(4).map(|x| x.to_string()).collect(),
            Data::U8(v) => v.iter().take(4).map(|x| x.to_string()).collect(),
            Data::Bool(v) => v.iter().take(4).map(|x| x.to_string()).collect(),
        };
        let more = if n > 4 { ", .." } else { "" };
        format!("{}{:?}[{}{}]", self.dtype(), self.shape, head.join(", "), more)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_round_trip() {
        let t = HostTensor::f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = t.to_bytes(DType::F32).unwrap();
        let back = HostTensor::from_bytes(DType::F32, vec![2, 2], &b).unwrap();
        assert_eq!(t, back);
        let h = t.to_bytes(DType::F16).unwrap();
        let back = HostTensor::from_bytes(DType::F16, vec![2, 2], &h).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn strides_and_casts() {
        assert_eq!(strides_of(&[2, 3, 4]), vec![12, 4, 1]);
        let t = HostTensor::i64(vec![3], vec![1, 0, 2]);
        assert_eq!(t.cast(DType::Bool).as_bool().to_vec(), vec![true, false, true]);
        assert_eq!(t.as_f32().to_vec(), vec![1.0, 0.0, 2.0]);
    }
}
