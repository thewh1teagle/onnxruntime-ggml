//! Logical shapes of any rank on four-dimensional ggml tensors.
//!
//! ggml tensors have at most four dimensions, but most higher-rank ONNX
//! tensors are still representable: dims of size 1 cost nothing and adjacent
//! dims can be merged whenever the elements are contiguous across them. So a
//! `DeviceTensor` carries a *logical* ONNX shape of any rank and a ggml tensor
//! whose `ne` is a **folding** of it.
//!
//! The invariant is deliberately simple: the ggml tensor holds the elements in
//! the row-major order of the logical shape. Every op here therefore starts by
//! making the tensor contiguous and re-folding it the way that op needs
//! (`[outer, axis, inner]` for anything that works along one axis), which is
//! pure metadata on a contiguous tensor. Shapes that do not fold are declined
//! with `Error::Unsupported` so the runtime falls back to the host.

use ggml_sys as g;

use crate::error::{Error, Result};
use crate::exec::ggml::{self, contig, Ctx, MAX_RANK};
use crate::exec::value::{DeviceTensor, MAX_LOGICAL_RANK};

/// Fold a logical ONNX shape into at most four dims: drop the size-1 dims,
/// then merge adjacent dims from the outermost side (the innermost dims are
/// the ones ggml ops care about, so they are kept apart the longest).
pub fn fold4(shape: &[usize]) -> Result<Vec<usize>> {
    if shape.len() > MAX_LOGICAL_RANK {
        return Err(Error::unsupported(format!("rank {} on device (max {MAX_LOGICAL_RANK})", shape.len())));
    }
    if shape.contains(&0) {
        return Err(Error::unsupported(format!("empty tensor {shape:?} on device")));
    }
    let mut dims: Vec<usize> = shape.iter().copied().filter(|&d| d != 1).collect();
    if dims.is_empty() {
        // all-ones (or scalar): one dim is enough, none for a true scalar
        return Ok(if shape.is_empty() { Vec::new() } else { vec![1] });
    }
    while dims.len() > MAX_RANK {
        let merged = dims[0] * dims[1];
        dims.remove(0);
        dims[0] = merged;
    }
    Ok(dims)
}

/// `true` when this shape needs the folding path at all.
pub fn needs_fold(shape: &[usize]) -> bool {
    shape.len() > MAX_RANK
}

/// The three groups an axis-wise op works on: everything before `axis`, the
/// axis itself, everything after it.
fn groups3(shape: &[usize], axis: usize) -> Result<(usize, usize, usize)> {
    if axis >= shape.len() {
        return Err(Error::shape(format!("axis {axis} on shape {shape:?}")));
    }
    if shape.contains(&0) {
        return Err(Error::unsupported(format!("empty tensor {shape:?} on device")));
    }
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    Ok((outer, shape[axis], inner))
}

/// A contiguous 3-D view `ne = [inner, dim_axis, outer]` of `d`.
unsafe fn as3(ctx: Ctx, d: DeviceTensor, axis: usize) -> Result<(ggml::T, usize, usize, usize)> {
    let shape = d.shape();
    let (outer, dim, inner) = groups3(&shape, axis)?;
    let src = contig(ctx, d);
    let t = g::ggml_reshape_3d(ctx, src.t, inner as i64, dim as i64, outer as i64);
    Ok((t, outer, dim, inner))
}

fn trace(what: &str, d: &DeviceTensor) {
    if needs_fold(&d.shape()) {
        unsafe {
            let ne = ggml::ne(d.t);
            tracing::trace!(op = what, onnx = ?d.shape(), ne = ?ne, "rank > 4 stays on device");
        }
    }
}

/// Reinterpret `d`'s elements under a new logical shape of any rank: a pure
/// metadata change once the tensor is contiguous.
pub unsafe fn reshape_logical(ctx: Ctx, d: DeviceTensor, shape: &[usize]) -> Result<DeviceTensor> {
    let total: usize = shape.iter().product();
    if total != d.numel() {
        return Err(Error::shape(format!("reshape {:?} -> {shape:?}", d.shape())));
    }
    // At rank <= 4 the ne stays the exact reverse-padded shape: the rest of the
    // emitters read `ne` positionally there. Only above that does it fold.
    let folded = if shape.len() <= MAX_RANK { shape.to_vec() } else { fold4(shape)? };
    let ne = ggml::ne_of(&folded)?;
    let src = contig(ctx, d);
    if ggml::ne(src.t) == ne {
        return Ok(ggml::dev(src.t, shape));
    }
    let t = g::ggml_reshape_4d(ctx, src.t, ne[0], ne[1], ne[2], ne[3]);
    let out = ggml::dev(t, shape);
    trace("reshape", &out);
    Ok(out)
}

/// `count` elements from `start` along one logical axis: a view, whatever the
/// rank. The other axes are untouched.
pub unsafe fn view_axis(ctx: Ctx, d: DeviceTensor, axis: usize, start: usize, count: usize) -> Result<DeviceTensor> {
    let shape = d.shape();
    let (_, dim, _) = groups3(&shape, axis)?;
    if start + count > dim || count == 0 {
        return Err(Error::shape(format!("slice {start}+{count} on axis {axis} of {shape:?}")));
    }
    let (t3, outer, _, inner) = as3(ctx, d, axis)?;
    let nb = ggml::nb(t3);
    let t = g::ggml_view_3d(ctx, t3, inner as i64, count as i64, outer as i64, nb[1], nb[2], start * nb[1]);
    let mut out_shape = shape;
    out_shape[axis] = count;
    let out = ggml::dev(t, &out_shape);
    trace("view", &out);
    Ok(out)
}

/// Concatenate two tensors of any rank along one logical axis.
pub unsafe fn concat_axis(ctx: Ctx, a: DeviceTensor, b: DeviceTensor, axis: usize) -> Result<DeviceTensor> {
    let (sa, sb) = (a.shape(), b.shape());
    if sa.len() != sb.len() {
        return Err(Error::shape(format!("concat {sa:?} with {sb:?}")));
    }
    if sa.iter().zip(&sb).enumerate().any(|(i, (x, y))| i != axis && x != y) {
        return Err(Error::shape(format!("concat {sa:?} with {sb:?} on axis {axis}")));
    }
    let (ta, _, da, _) = as3(ctx, a, axis)?;
    let (tb, _, db, _) = as3(ctx, b, axis)?;
    let t = g::ggml_concat(ctx, ta, tb, 1);
    let mut out_shape = sa;
    out_shape[axis] = da + db;
    let out = ggml::dev(t, &out_shape);
    trace("concat", &out);
    Ok(out)
}

/// ONNX Transpose of any rank. Only permutations that survive dropping the
/// size-1 axes (four or fewer real axes left) are expressible; the rest are
/// declined.
pub unsafe fn permute_logical(ctx: Ctx, d: DeviceTensor, perm: &[usize]) -> Result<DeviceTensor> {
    let shape = d.shape();
    if perm.len() != shape.len() {
        return Err(Error::shape(format!("perm {perm:?} for shape {shape:?}")));
    }
    let keep: Vec<usize> = (0..shape.len()).filter(|&i| shape[i] != 1).collect();
    if keep.len() > MAX_RANK {
        return Err(Error::unsupported(format!("transpose {perm:?} of {shape:?}: {} real axes", keep.len())));
    }
    let reduced: Vec<usize> = keep.iter().map(|&i| shape[i]).collect();
    let rperm: Vec<usize> =
        perm.iter().filter(|&&p| shape[p] != 1).map(|&p| keep.iter().position(|&k| k == p).unwrap()).collect();
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    if reduced.is_empty() {
        return reshape_logical(ctx, d, &out_shape);
    }
    let src = reshape_logical(ctx, d, &reduced)?;
    let permuted = ggml::permute(ctx, src, &rperm)?;
    let out = ggml::dev(permuted.t, &out_shape);
    trace("transpose", &out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folding() {
        assert_eq!(fold4(&[2, 3, 4]).unwrap(), vec![2, 3, 4]);
        assert_eq!(fold4(&[7, 6, 2, 1, 16, 64]).unwrap(), vec![42, 2, 16, 64]);
        assert_eq!(fold4(&[266, 2, 2, 1, 8, 64]).unwrap(), vec![532, 2, 8, 64]);
        assert_eq!(fold4(&[1, 1, 1, 1, 1]).unwrap(), vec![1]);
        assert_eq!(fold4(&[]).unwrap(), Vec::<usize>::new());
        assert!(fold4(&[2, 0, 3]).is_err());
        assert!(fold4(&[2; 9]).is_err());
    }

    #[test]
    fn folding_preserves_element_count() {
        for shape in [vec![7, 6, 2, 1, 16, 64], vec![266, 2, 2, 1, 8, 64], vec![2, 3, 4, 5, 6]] {
            let n: usize = shape.iter().product();
            assert_eq!(fold4(&shape).unwrap().iter().product::<usize>(), n);
        }
    }
}
