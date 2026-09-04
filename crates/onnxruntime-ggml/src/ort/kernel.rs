//! Moving tensors between an `OrtKernelContext` and host memory. The provider
//! registers itself on the CPU device, so every input and output here is
//! plain host memory owned by onnxruntime.

use ort_ep_sys::*;

use crate::error::Result;
use crate::host::tensor::HostTensor;
use crate::ir::DType;
use crate::ort::graph::read_value;
use crate::ort_call;

/// # Safety
/// `ctx` is the context onnxruntime passed to `Compute`.
pub unsafe fn read_inputs(ctx: *const OrtKernelContext) -> Result<Vec<HostTensor>> {
    let mut n = 0usize;
    ort_call!(KernelContext_GetInputCount(ctx, &mut n))?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut value: *const OrtValue = std::ptr::null();
        ort_call!(KernelContext_GetInput(ctx, i, &mut value))?;
        let t = read_value(value)?;
        tracing::trace!(index = i, tensor = %t.brief(), "kernel input");
        out.push(t);
    }
    Ok(out)
}

/// # Safety
/// `ctx` is the context onnxruntime passed to `Compute`.
pub unsafe fn output_count(ctx: *const OrtKernelContext) -> Result<usize> {
    let mut n = 0usize;
    ort_call!(KernelContext_GetOutputCount(ctx, &mut n))?;
    Ok(n)
}

/// # Safety
/// `ctx` is the context onnxruntime passed to `Compute`; `index` < output count.
pub unsafe fn write_output(ctx: *mut OrtKernelContext, index: usize, t: &HostTensor, dtype: DType) -> Result<()> {
    let dims: Vec<i64> = t.shape.iter().map(|&d| d as i64).collect();
    let mut value: *mut OrtValue = std::ptr::null_mut();
    ort_call!(KernelContext_GetOutput(ctx, index, dims.as_ptr(), dims.len(), &mut value))?;
    let bytes = t.to_bytes(dtype)?;
    if !bytes.is_empty() {
        let mut data: *mut std::os::raw::c_void = std::ptr::null_mut();
        ort_call!(GetTensorMutableData(value, &mut data))?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data as *mut u8, bytes.len());
    }
    tracing::trace!(index, tensor = %t.brief(), dtype = %dtype, "kernel output");
    Ok(())
}
