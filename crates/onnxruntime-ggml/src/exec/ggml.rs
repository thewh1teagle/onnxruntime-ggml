//! Safe-ish helpers over ggml tensors, and the one convention everything else
//! relies on: an ONNX shape `[d0, d1, .., dn-1]` is ggml `ne = [dn-1, .., d0]`
//! padded with 1s to four entries. ggml's `ne[0]` is the contiguous
//! (innermost) dimension, exactly like the last ONNX axis.

use std::ffi::CString;

use ggml_sys as g;

use crate::error::{Error, Result};
use crate::exec::value::{DeviceTensor, MAX_LOGICAL_RANK};
use crate::ir::DType;

pub type Ctx = *mut g::ggml_context;
pub type T = *mut g::ggml_tensor;

pub const MAX_RANK: usize = 4;

pub fn ne_of(shape: &[usize]) -> Result<[i64; 4]> {
    if shape.len() > MAX_RANK {
        return Err(Error::unsupported(format!("rank {} on device (max 4): {shape:?}", shape.len())));
    }
    let mut ne = [1i64; 4];
    for (i, &d) in shape.iter().rev().enumerate() {
        ne[i] = d as i64;
    }
    Ok(ne)
}

pub fn shape_arr(shape: &[usize]) -> [usize; MAX_LOGICAL_RANK] {
    let mut s = [1usize; MAX_LOGICAL_RANK];
    let n = shape.len().min(MAX_LOGICAL_RANK);
    s[..n].copy_from_slice(&shape[..n]);
    s
}

/// ggml axis index of ONNX axis `axis` for a tensor of `rank`.
pub fn gaxis(axis: usize, rank: usize) -> i32 {
    (rank - 1 - axis) as i32
}

pub fn gtype(dtype: DType) -> Result<g::ggml_type> {
    Ok(match dtype {
        DType::F32 => g::ggml_type_GGML_TYPE_F32,
        DType::F16 => g::ggml_type_GGML_TYPE_F16,
        DType::I32 => g::ggml_type_GGML_TYPE_I32,
        other => return Err(Error::unsupported(format!("{other} on device"))),
    })
}

pub fn dev(t: T, shape: &[usize]) -> DeviceTensor {
    DeviceTensor { t, rank: shape.len(), shape: shape_arr(shape) }
}

/// Allocate a leaf whose logical shape may have any rank: the ggml tensor gets
/// the folded shape, the `DeviceTensor` keeps the logical one.
pub unsafe fn new_tensor(ctx: Ctx, dtype: DType, shape: &[usize]) -> Result<DeviceTensor> {
    // Below the ggml rank limit the ne stays the exact reverse-padded shape;
    // only above it does the logical shape fold.
    let folded = if shape.len() <= MAX_RANK { shape.to_vec() } else { crate::exec::fold::fold4(shape)? };
    let ne = ne_of(&folded)?;
    let n_dims = folded.len().clamp(1, 4) as i32;
    let t = g::ggml_new_tensor(ctx, gtype(dtype)?, n_dims, ne.as_ptr());
    if t.is_null() {
        return Err(Error::ggml("ggml_new_tensor returned null (context out of memory?)"));
    }
    if shape.len() > MAX_RANK {
        tracing::trace!(onnx = ?shape, ne = ?&ne[..n_dims as usize], "rank > 4 leaf, folded");
    }
    Ok(dev(t, shape))
}

/// A tensor of an explicit ggml type (quantised weights).
pub unsafe fn new_tensor_typed(ctx: Ctx, ty: g::ggml_type, shape: &[usize]) -> Result<DeviceTensor> {
    let ne = ne_of(shape)?;
    let n_dims = shape.len().clamp(1, 4) as i32;
    let t = g::ggml_new_tensor(ctx, ty, n_dims, ne.as_ptr());
    if t.is_null() {
        return Err(Error::ggml("ggml_new_tensor returned null"));
    }
    Ok(dev(t, shape))
}

pub unsafe fn set_name(t: T, name: &str) {
    let trimmed: String = name.chars().take(60).collect();
    if let Ok(c) = CString::new(trimmed) {
        g::ggml_set_name(t, c.as_ptr());
    }
}

pub unsafe fn is_contiguous(t: T) -> bool {
    g::ggml_is_contiguous(t) && rows_dense(t)
}

/// Is the innermost dimension packed? `ggml_is_contiguous` skips the `nb[0]`
/// check when `ne[0]` is one block (always the case for `ne[0] == 1` with an
/// f32 tensor), so a permuted view of shape `[.., 1]` passes it while keeping a
/// strided `nb[0]`. CPU kernels that assert `nb00 == sizeof(float)` (im2col,
/// mul_mat) then abort, while Metal happily reads the strides.
pub unsafe fn rows_dense(t: T) -> bool {
    (*t).nb[0] == g::ggml_type_size((*t).type_)
}

/// A contiguous copy when the tensor is a strided view, else the tensor itself.
pub unsafe fn contig(ctx: Ctx, d: DeviceTensor) -> DeviceTensor {
    if g::ggml_is_contiguous(d.t) && rows_dense(d.t) {
        d
    } else {
        DeviceTensor { t: g::ggml_cont(ctx, d.t), ..d }
    }
}

pub unsafe fn nelements(t: T) -> usize {
    g::ggml_nelements(t) as usize
}

pub unsafe fn nbytes(t: T) -> usize {
    g::ggml_nbytes(t)
}

pub unsafe fn ne(t: T) -> [i64; 4] {
    (*t).ne
}

pub unsafe fn nb(t: T) -> [usize; 4] {
    (*t).nb
}

/// Reshape (row-major, like ONNX). Makes the source contiguous first.
pub unsafe fn reshape(ctx: Ctx, d: DeviceTensor, shape: &[usize]) -> Result<DeviceTensor> {
    let ne = ne_of(shape)?;
    let total: usize = shape.iter().product();
    if total != d.numel() {
        return Err(Error::shape(format!("reshape {:?} -> {shape:?}", d.shape())));
    }
    let src = contig(ctx, d);
    let t = g::ggml_reshape_4d(ctx, src.t, ne[0], ne[1], ne[2], ne[3]);
    Ok(dev(t, shape))
}

/// ONNX Transpose: `out[i] = in[perm[i]]`. Returns a strided view.
pub unsafe fn permute(ctx: Ctx, d: DeviceTensor, perm: &[usize]) -> Result<DeviceTensor> {
    let rank = d.rank;
    if perm.len() != rank || rank > MAX_RANK {
        return Err(Error::shape(format!("perm {perm:?} for rank {rank}")));
    }
    // ggml_permute(a, ax0..ax3): input ggml axis j goes to output ggml axis ax_j.
    // ONNX: input axis perm[i] goes to output axis i.
    let mut axes = [0i32, 1, 2, 3];
    for (i, &p) in perm.iter().enumerate() {
        let in_g = rank - 1 - p;
        let out_g = rank - 1 - i;
        axes[in_g] = out_g as i32;
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| d.shape[p]).collect();
    let t = g::ggml_permute(ctx, d.t, axes[0], axes[1], axes[2], axes[3]);
    Ok(dev(t, &out_shape))
}

/// A strided view selecting `count[i]` elements from `start[i]` along each ONNX axis.
pub unsafe fn view_slice(ctx: Ctx, d: DeviceTensor, start: &[usize], count: &[usize]) -> Result<DeviceTensor> {
    let rank = d.rank;
    if rank > MAX_RANK || start.len() != rank || count.len() != rank {
        return Err(Error::shape(format!("view {start:?}/{count:?} on rank {rank}")));
    }
    let nb = nb(d.t);
    let mut offset = 0usize;
    for i in 0..rank {
        if start[i] + count[i] > d.shape[i] {
            return Err(Error::shape(format!("slice {start:?}+{count:?} exceeds {:?}", d.shape())));
        }
        offset += start[i] * nb[rank - 1 - i];
    }
    let ne = ne_of(count)?;
    let t = g::ggml_view_4d(ctx, d.t, ne[0], ne[1], ne[2], ne[3], nb[1], nb[2], nb[3], offset);
    Ok(dev(t, count))
}

/// Broadcast `d` to `shape` by repetition (a real copy).
pub unsafe fn repeat_to(ctx: Ctx, d: DeviceTensor, shape: &[usize]) -> Result<DeviceTensor> {
    let ne = ne_of(shape)?;
    let t = g::ggml_repeat_4d(ctx, d.t, ne[0], ne[1], ne[2], ne[3]);
    Ok(dev(t, shape))
}

/// `a * s + b` elementwise.
pub unsafe fn scale_bias(ctx: Ctx, d: DeviceTensor, s: f32, b: f32) -> DeviceTensor {
    DeviceTensor { t: g::ggml_scale_bias(ctx, d.t, s, b), ..d }
}

pub fn describe(d: &DeviceTensor) -> String {
    unsafe {
        let ne = ne(d.t);
        let name = std::ffi::CStr::from_ptr(g::ggml_get_name(d.t)).to_string_lossy();
        format!(
            "{name}: onnx{:?} ne{:?} nb0={} contiguous={}",
            d.shape(),
            &ne[..d.rank.clamp(1, MAX_RANK)],
            (*d.t).nb[0],
            is_contiguous(d.t)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_convention() {
        assert_eq!(ne_of(&[2, 3, 4]).unwrap(), [4, 3, 2, 1]);
        assert_eq!(ne_of(&[]).unwrap(), [1, 1, 1, 1]);
        assert!(ne_of(&[1, 2, 3, 4, 5]).is_err());
        assert_eq!(gaxis(0, 3), 2);
        assert_eq!(gaxis(2, 3), 0);
    }
}
