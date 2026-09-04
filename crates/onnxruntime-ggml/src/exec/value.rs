//! Runtime values. The executor never guesses where a tensor lives: it is one
//! of these three, and every op emitter says which it needs.

use std::sync::Arc;

use ggml_sys::ggml_tensor;

use crate::host::tensor::HostTensor;
use crate::ir::DType;

/// A tensor that exists inside the current ggml graph. `shape` is the ONNX
/// (row-major) shape; ggml's `ne` is its reverse padded with 1s, so the
/// original rank has to be carried separately.
#[derive(Clone, Copy, Debug)]
pub struct DeviceTensor {
    pub t: *mut ggml_tensor,
    pub rank: usize,
    pub shape: [usize; 4],
}

impl DeviceTensor {
    pub fn shape(&self) -> Vec<usize> {
        self.shape[..self.rank].to_vec()
    }
    pub fn numel(&self) -> usize {
        self.shape[..self.rank].iter().product()
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    /// Host data that belongs on the host: shapes, indices, masks, scalars.
    Host(Arc<HostTensor>),
    /// Host data that wants to be on the device (float activations, weights not
    /// yet uploaded, readbacks). It is uploaded the first time a device op
    /// consumes it; a host op can read it directly.
    Staged(Arc<HostTensor>),
    /// Lives in the current ggml graph; valid until the next flush.
    Device(DeviceTensor),
}

impl Value {
    pub fn host_of(t: HostTensor) -> Value {
        Value::Host(Arc::new(t))
    }

    pub fn staged_of(t: HostTensor) -> Value {
        Value::Staged(Arc::new(t))
    }

    pub fn is_device(&self) -> bool {
        matches!(self, Value::Device(_))
    }

    pub fn is_host(&self) -> bool {
        matches!(self, Value::Host(_))
    }

    pub fn is_staged(&self) -> bool {
        matches!(self, Value::Staged(_))
    }

    pub fn host(&self) -> Option<&HostTensor> {
        match self {
            Value::Host(t) | Value::Staged(t) => Some(t),
            Value::Device(_) => None,
        }
    }

    pub fn device(&self) -> Option<DeviceTensor> {
        match self {
            Value::Device(d) => Some(*d),
            _ => None,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Value::Host(t) | Value::Staged(t) => t.shape.clone(),
            Value::Device(d) => d.shape(),
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            Value::Host(t) | Value::Staged(t) => t.rank(),
            Value::Device(d) => d.rank,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Value::Host(t) | Value::Staged(t) => t.dtype(),
            Value::Device(_) => DType::F32,
        }
    }

    pub fn place(&self) -> &'static str {
        match self {
            Value::Host(_) => "host",
            Value::Staged(_) => "staged",
            Value::Device(_) => "device",
        }
    }

    pub fn brief(&self) -> String {
        match self {
            Value::Host(t) => format!("host:{}", t.brief()),
            Value::Staged(t) => format!("staged:{}", t.brief()),
            Value::Device(d) => format!("device:f32{:?}", d.shape()),
        }
    }
}
