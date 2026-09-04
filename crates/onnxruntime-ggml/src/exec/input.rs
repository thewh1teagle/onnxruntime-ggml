//! Graph inputs as the runtime receives them from onnxruntime.
//!
//! Large float inputs are *borrowed*: a pointer into onnxruntime's own buffer,
//! valid for the duration of the `Compute` call that owns the run. The sticky
//! cache fingerprints them in place and a hit costs nothing; a miss uploads
//! straight from that pointer. Copying 61 MiB of KV cache into owned tensors
//! every decode step used to cost more than the step's compute.

use crate::host::tensor::HostTensor;
use crate::ir::DType;

pub enum InputRef {
    Owned(HostTensor),
    /// # Safety
    /// `ptr` must stay valid and unmodified until the run that received it ends.
    Borrowed {
        dtype: DType,
        shape: Vec<usize>,
        ptr: *const u8,
        nbytes: usize,
    },
}

impl InputRef {
    pub fn shape(&self) -> &[usize] {
        match self {
            InputRef::Owned(t) => &t.shape,
            InputRef::Borrowed { shape, .. } => shape,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            InputRef::Owned(t) => t.dtype(),
            InputRef::Borrowed { dtype, .. } => *dtype,
        }
    }

    /// The bytes as a slice.
    ///
    /// # Safety
    /// See `Borrowed`.
    pub unsafe fn bytes(&self) -> Option<&[u8]> {
        match self {
            InputRef::Owned(_) => None,
            InputRef::Borrowed { ptr, nbytes, .. } => Some(std::slice::from_raw_parts(*ptr, *nbytes)),
        }
    }

    /// Materialise an owned tensor (copies a borrowed input).
    pub fn to_owned(&self) -> crate::error::Result<HostTensor> {
        match self {
            InputRef::Owned(t) => Ok(t.clone()),
            InputRef::Borrowed { dtype, shape, .. } => {
                let bytes = unsafe { self.bytes().unwrap() };
                HostTensor::from_bytes(*dtype, shape.clone(), bytes)
            }
        }
    }
}
