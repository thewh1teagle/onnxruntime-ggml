//! Structural ops on host tensors: shape, reshape, gather, slice, concat, and
//! the index-generating ops (Range, ConstantOfShape). Any dtype, any rank.

use crate::error::{Error, Result};
use crate::host::broadcast::{broadcast_index, broadcast_shapes};
use crate::host::eval::{need, norm_axis};
use crate::host::tensor::{numel_of, strides_of, Data, HostTensor};
use crate::ir::{DType, Node};

pub fn eval(node: &Node, inputs: &[Option<&HostTensor>]) -> Result<Vec<HostTensor>> {
    let out = match node.op.as_str() {
        "Shape" => {
            let x = need(node, inputs, 0)?;
            let start = norm_axis_or(node.attr_i("start", 0), x.rank(), 0);
            let end = norm_axis_or(node.attr_i("end", x.rank() as i64), x.rank(), x.rank());
            let dims: Vec<i64> = x.shape[start..end.max(start)].iter().map(|&d| d as i64).collect();
            HostTensor::i64(vec![dims.len()], dims)
        }
        "Constant" => {
            node.attr_tensor("value").cloned().ok_or_else(|| Error::model(format!("{node}: Constant without a tensor value")))?
        }
        "Identity" => need(node, inputs, 0)?.clone(),
        "Reshape" => {
            let x = need(node, inputs, 0)?;
            let shape = need(node, inputs, 1)?.as_i64().to_vec();
            x.reshaped(resolve_reshape(&x.shape, &shape, node.attr_i("allowzero", 0) != 0)?)?
        }
        "Unsqueeze" => {
            let x = need(node, inputs, 0)?;
            let axes = axes_of(node, inputs, 1)?;
            x.reshaped(unsqueeze_shape(&x.shape, &axes)?)?
        }
        "Squeeze" => {
            let x = need(node, inputs, 0)?;
            let axes = if inputs.get(1).copied().flatten().is_some() { Some(axes_of(node, inputs, 1)?) } else { None };
            x.reshaped(squeeze_shape(&x.shape, axes.as_deref())?)?
        }
        "Transpose" => {
            let x = need(node, inputs, 0)?;
            let perm = node.attr_ints("perm").unwrap_or_else(|| (0..x.rank() as i64).rev().collect());
            transpose(x, &perm)?
        }
        "Concat" => {
            let xs: Vec<&HostTensor> = inputs.iter().flatten().copied().collect();
            let axis = norm_axis(node.attr_i("axis", 0), xs[0].rank())?;
            concat(&xs, axis)?
        }
        "Slice" => {
            let x = need(node, inputs, 0)?;
            let starts = need(node, inputs, 1)?.as_i64().to_vec();
            let ends = need(node, inputs, 2)?.as_i64().to_vec();
            let axes = inputs.get(3).copied().flatten().map(|t| t.as_i64().to_vec());
            let steps = inputs.get(4).copied().flatten().map(|t| t.as_i64().to_vec());
            let plan = slice_plan(&x.shape, &starts, &ends, axes.as_deref(), steps.as_deref())?;
            slice(x, &plan)
        }
        "Gather" => {
            let x = need(node, inputs, 0)?;
            let idx = need(node, inputs, 1)?;
            let axis = norm_axis(node.attr_i("axis", 0), x.rank())?;
            gather(x, idx, axis)?
        }
        "Split" => {
            let x = need(node, inputs, 0)?;
            let axis = norm_axis(node.attr_i("axis", 0), x.rank())?;
            let sizes = match inputs.get(1).copied().flatten() {
                Some(t) => t.as_i64().iter().map(|&v| v as usize).collect::<Vec<_>>(),
                None => {
                    let n = node.attr_i("num_outputs", node.outputs.len() as i64) as usize;
                    let total = x.shape[axis];
                    let chunk = total.div_ceil(n);
                    (0..n).map(|i| chunk.min(total - (chunk * i).min(total))).collect()
                }
            };
            return split(x, axis, &sizes);
        }
        "Range" => {
            let start = need(node, inputs, 0)?;
            let limit = need(node, inputs, 1)?;
            let delta = need(node, inputs, 2)?;
            range(start, limit, delta)?
        }
        "ConstantOfShape" => {
            let shape: Vec<usize> = need(node, inputs, 0)?.as_i64().iter().map(|&v| v.max(0) as usize).collect();
            let n = numel_of(&shape);
            match node.attr_tensor("value") {
                Some(v) => {
                    let one = v.gather_flat(vec![], &[0]);
                    let data = match one.data {
                        Data::F32(v) => Data::F32(vec![v[0]; n]),
                        Data::F64(v) => Data::F64(vec![v[0]; n]),
                        Data::I64(v) => Data::I64(vec![v[0]; n]),
                        Data::I32(v) => Data::I32(vec![v[0]; n]),
                        Data::I8(v) => Data::I8(vec![v[0]; n]),
                        Data::U8(v) => Data::U8(vec![v[0]; n]),
                        Data::Bool(v) => Data::Bool(vec![v[0]; n]),
                    };
                    HostTensor { shape, data }
                }
                None => HostTensor::f32(shape, vec![0.0; n]),
            }
        }
        "Expand" => {
            let x = need(node, inputs, 0)?;
            let target: Vec<usize> = need(node, inputs, 1)?.as_i64().iter().map(|&v| v as usize).collect();
            let shape = broadcast_shapes(&x.shape, &target)?;
            let idx = broadcast_index(&x.shape, &shape);
            x.gather_flat(shape, &idx)
        }
        "Cast" => {
            let x = need(node, inputs, 0)?;
            let to = DType::from_onnx(node.attr_i("to", 1) as i32)?;
            x.cast(to)
        }
        "Where" => {
            let c = need(node, inputs, 0)?;
            let x = need(node, inputs, 1)?;
            let y = need(node, inputs, 2)?;
            where_(c, x, y)?
        }
        other => return Err(Error::unsupported(format!("host shape op {other}"))),
    };
    Ok(vec![out])
}

fn norm_axis_or(axis: i64, rank: usize, default: usize) -> usize {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 {
        0
    } else if a > r {
        default.min(rank)
    } else {
        a as usize
    }
}

fn axes_of(node: &Node, inputs: &[Option<&HostTensor>], i: usize) -> Result<Vec<i64>> {
    if let Some(t) = inputs.get(i).copied().flatten() {
        return Ok(t.as_i64().to_vec());
    }
    node.attr_ints("axes").ok_or_else(|| Error::model(format!("{node}: no axes")))
}

/// ONNX Reshape semantics: 0 copies the input dim (unless allowzero), -1 infers.
pub fn resolve_reshape(input: &[usize], shape: &[i64], allowzero: bool) -> Result<Vec<usize>> {
    let total = numel_of(input);
    let mut out = Vec::with_capacity(shape.len());
    let mut infer = None;
    for (i, &d) in shape.iter().enumerate() {
        if d == -1 {
            if infer.is_some() {
                return Err(Error::shape(format!("reshape {shape:?} has two -1 dims")));
            }
            infer = Some(i);
            out.push(1);
        } else if d == 0 && !allowzero {
            out.push(*input.get(i).ok_or_else(|| Error::shape(format!("reshape {shape:?}: 0 beyond input rank")))?);
        } else {
            out.push(d as usize);
        }
    }
    if let Some(i) = infer {
        let known: usize = out.iter().product();
        if known == 0 || !total.is_multiple_of(known) {
            return Err(Error::shape(format!("cannot infer -1 reshaping {input:?} to {shape:?}")));
        }
        out[i] = total / known;
    }
    if numel_of(&out) != total {
        return Err(Error::shape(format!("reshape {input:?} to {shape:?} changes element count")));
    }
    Ok(out)
}

pub fn unsqueeze_shape(input: &[usize], axes: &[i64]) -> Result<Vec<usize>> {
    let rank = input.len() + axes.len();
    let mut axes: Vec<usize> = axes.iter().map(|&a| norm_axis(a, rank)).collect::<Result<_>>()?;
    axes.sort_unstable();
    let mut out = Vec::with_capacity(rank);
    let mut src = input.iter();
    for i in 0..rank {
        if axes.contains(&i) {
            out.push(1);
        } else {
            out.push(*src.next().ok_or_else(|| Error::shape("unsqueeze axes overlap"))?);
        }
    }
    Ok(out)
}

pub fn squeeze_shape(input: &[usize], axes: Option<&[i64]>) -> Result<Vec<usize>> {
    match axes {
        None => Ok(input.iter().copied().filter(|&d| d != 1).collect()),
        Some(axes) => {
            let axes: Vec<usize> = axes.iter().map(|&a| norm_axis(a, input.len())).collect::<Result<_>>()?;
            for &a in &axes {
                if input[a] != 1 {
                    return Err(Error::shape(format!("squeeze axis {a} of {input:?} is not 1")));
                }
            }
            Ok(input.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, &d)| d).collect())
        }
    }
}

pub fn transpose(x: &HostTensor, perm: &[i64]) -> Result<HostTensor> {
    let rank = x.rank();
    if perm.len() != rank {
        return Err(Error::shape(format!("perm {perm:?} for rank {rank}")));
    }
    let perm: Vec<usize> = perm.iter().map(|&p| norm_axis(p, rank)).collect::<Result<_>>()?;
    let out_shape: Vec<usize> = perm.iter().map(|&p| x.shape[p]).collect();
    let in_strides = strides_of(&x.shape);
    let n = x.numel();
    let mut index = Vec::with_capacity(n);
    let mut idx = vec![0usize; rank];
    for _ in 0..n {
        let mut flat = 0;
        for d in 0..rank {
            flat += idx[d] * in_strides[perm[d]];
        }
        index.push(flat);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(x.gather_flat(out_shape, &index))
}

pub fn concat(xs: &[&HostTensor], axis: usize) -> Result<HostTensor> {
    let first = xs[0];
    let mut out_shape = first.shape.clone();
    out_shape[axis] = xs.iter().map(|t| t.shape[axis]).sum();
    for t in xs {
        if t.rank() != first.rank() {
            return Err(Error::shape("concat ranks differ"));
        }
        for d in 0..t.rank() {
            if d != axis && t.shape[d] != first.shape[d] {
                return Err(Error::shape(format!("concat shapes {:?} vs {:?} on axis {axis}", t.shape, first.shape)));
            }
        }
    }
    let outer: usize = out_shape[..axis].iter().product();
    let inner: usize = out_shape[axis + 1..].iter().product();
    let n = numel_of(&out_shape);
    // Build the result by copying slabs; dtype follows the first input.
    let dtype = first.dtype();
    let mut out = HostTensor::zeros(dtype, out_shape.clone());
    let mut pos = 0usize;
    let _ = n;
    // Cast once per input, not once per slab: `cast` copies the whole tensor
    // (even when the dtype already matches), and the slab loop runs `outer`
    // times, so casting inside it turned a memcpy into an O(outer) full copy.
    let src: Vec<std::borrow::Cow<HostTensor>> = xs
        .iter()
        .map(|t| if t.dtype() == dtype { std::borrow::Cow::Borrowed(*t) } else { std::borrow::Cow::Owned(t.cast(dtype)) })
        .collect();
    for o in 0..outer {
        for t in &src {
            let slab = t.shape[axis] * inner;
            let src_start = o * slab;
            copy_slab(&mut out, pos, t, src_start, slab);
            pos += slab;
        }
    }
    Ok(out)
}

fn copy_slab(dst: &mut HostTensor, dst_start: usize, src: &HostTensor, src_start: usize, len: usize) {
    match (&mut dst.data, &src.data) {
        (Data::F32(d), Data::F32(s)) => d[dst_start..dst_start + len].copy_from_slice(&s[src_start..src_start + len]),
        (Data::F64(d), Data::F64(s)) => d[dst_start..dst_start + len].copy_from_slice(&s[src_start..src_start + len]),
        (Data::I64(d), Data::I64(s)) => d[dst_start..dst_start + len].copy_from_slice(&s[src_start..src_start + len]),
        (Data::I32(d), Data::I32(s)) => d[dst_start..dst_start + len].copy_from_slice(&s[src_start..src_start + len]),
        (Data::I8(d), Data::I8(s)) => d[dst_start..dst_start + len].copy_from_slice(&s[src_start..src_start + len]),
        (Data::U8(d), Data::U8(s)) => d[dst_start..dst_start + len].copy_from_slice(&s[src_start..src_start + len]),
        (Data::Bool(d), Data::Bool(s)) => d[dst_start..dst_start + len].copy_from_slice(&s[src_start..src_start + len]),
        _ => unreachable!("copy_slab after cast"),
    }
}

/// Per-axis (start, count, step) after ONNX clamping rules.
#[derive(Clone, Debug, PartialEq)]
pub struct SlicePlan {
    pub start: Vec<usize>,
    pub count: Vec<usize>,
    pub step: Vec<i64>,
}

pub fn slice_plan(
    shape: &[usize],
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Result<SlicePlan> {
    let rank = shape.len();
    let mut plan = SlicePlan { start: vec![0; rank], count: shape.to_vec(), step: vec![1; rank] };
    for (k, (&s, &e)) in starts.iter().zip(ends.iter()).enumerate() {
        let axis = match axes {
            Some(a) => norm_axis(a[k], rank)?,
            None => k,
        };
        let step = steps.map(|st| st[k]).unwrap_or(1);
        if step == 0 {
            return Err(Error::shape("slice step 0"));
        }
        let dim = shape[axis] as i64;
        let (start, end) = if step > 0 {
            let s = if s < 0 { s + dim } else { s }.clamp(0, dim);
            let e = if e < 0 { e + dim } else { e }.clamp(0, dim);
            (s, e)
        } else {
            let s = if s < 0 { s + dim } else { s }.clamp(-1, dim - 1);
            let e = if e < 0 { e + dim } else { e }.clamp(-1, dim - 1);
            (s, e)
        };
        let count =
            if step > 0 { ((end - start).max(0) + step - 1) / step } else { ((start - end).max(0) + (-step) - 1) / (-step) };
        plan.start[axis] = start.max(0) as usize;
        plan.count[axis] = count.max(0) as usize;
        plan.step[axis] = step;
    }
    Ok(plan)
}

pub fn slice(x: &HostTensor, plan: &SlicePlan) -> HostTensor {
    let rank = x.rank();
    let in_strides = strides_of(&x.shape);
    let n = numel_of(&plan.count);
    let mut index = Vec::with_capacity(n);
    let mut idx = vec![0usize; rank];
    for _ in 0..n {
        let mut flat = 0i64;
        for d in 0..rank {
            flat += (plan.start[d] as i64 + idx[d] as i64 * plan.step[d]) * in_strides[d] as i64;
        }
        index.push(flat as usize);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < plan.count[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    x.gather_flat(plan.count.clone(), &index)
}

pub fn gather(x: &HostTensor, idx: &HostTensor, axis: usize) -> Result<HostTensor> {
    let dim = x.shape[axis] as i64;
    let ids: Vec<usize> = idx
        .as_i64()
        .iter()
        .map(|&i| {
            let i = if i < 0 { i + dim } else { i };
            if i < 0 || i >= dim {
                Err(Error::shape(format!("gather index {i} out of range {dim}")))
            } else {
                Ok(i as usize)
            }
        })
        .collect::<Result<_>>()?;
    let outer: usize = x.shape[..axis].iter().product();
    let inner: usize = x.shape[axis + 1..].iter().product();
    let mut out_shape = Vec::new();
    out_shape.extend_from_slice(&x.shape[..axis]);
    out_shape.extend_from_slice(&idx.shape);
    out_shape.extend_from_slice(&x.shape[axis + 1..]);
    let mut index = Vec::with_capacity(numel_of(&out_shape));
    for o in 0..outer {
        for &i in &ids {
            let base = (o * x.shape[axis] + i) * inner;
            index.extend(base..base + inner);
        }
    }
    Ok(x.gather_flat(out_shape, &index))
}

pub fn split(x: &HostTensor, axis: usize, sizes: &[usize]) -> Result<Vec<HostTensor>> {
    if sizes.iter().sum::<usize>() != x.shape[axis] {
        return Err(Error::shape(format!("split sizes {sizes:?} do not cover {}", x.shape[axis])));
    }
    let mut outs = Vec::with_capacity(sizes.len());
    let mut start = 0i64;
    for &size in sizes {
        let plan = slice_plan(&x.shape, &[start], &[start + size as i64], Some(&[axis as i64]), None)?;
        outs.push(slice(x, &plan));
        start += size as i64;
    }
    Ok(outs)
}

pub fn range(start: &HostTensor, limit: &HostTensor, delta: &HostTensor) -> Result<HostTensor> {
    if start.dtype().is_float() {
        let (s, l, d) = (start.scalar_f64()?, limit.scalar_f64()?, delta.scalar_f64()?);
        if d == 0.0 {
            return Err(Error::shape("range delta 0"));
        }
        let n = ((l - s) / d).ceil().max(0.0) as usize;
        let data: Vec<f32> = (0..n).map(|i| (s + i as f64 * d) as f32).collect();
        Ok(HostTensor::f32(vec![n], data))
    } else {
        let (s, l, d) = (start.scalar_i64()?, limit.scalar_i64()?, delta.scalar_i64()?);
        if d == 0 {
            return Err(Error::shape("range delta 0"));
        }
        let n = if d > 0 { (l - s + d - 1).div_euclid(d).max(0) } else { (s - l + (-d) - 1).div_euclid(-d).max(0) } as usize;
        let data: Vec<i64> = (0..n).map(|i| s + i as i64 * d).collect();
        Ok(HostTensor::i64(vec![n], data))
    }
}

pub fn where_(c: &HostTensor, x: &HostTensor, y: &HostTensor) -> Result<HostTensor> {
    let shape = broadcast_shapes(&broadcast_shapes(&c.shape, &x.shape)?, &y.shape)?;
    let ci = broadcast_index(&c.shape, &shape);
    let xi = broadcast_index(&x.shape, &shape);
    let yi = broadcast_index(&y.shape, &shape);
    let cond = c.as_bool();
    let n = numel_of(&shape);
    let dtype = if x.dtype() == y.dtype() {
        x.dtype()
    } else if x.dtype().is_float() || y.dtype().is_float() {
        DType::F32
    } else {
        DType::I64
    };
    let x = x.cast(dtype);
    let y = y.cast(dtype);
    let data = match (&x.data, &y.data) {
        (Data::F32(a), Data::F32(b)) => Data::F32((0..n).map(|i| if cond[ci[i]] { a[xi[i]] } else { b[yi[i]] }).collect()),
        (Data::F64(a), Data::F64(b)) => Data::F64((0..n).map(|i| if cond[ci[i]] { a[xi[i]] } else { b[yi[i]] }).collect()),
        (Data::I64(a), Data::I64(b)) => Data::I64((0..n).map(|i| if cond[ci[i]] { a[xi[i]] } else { b[yi[i]] }).collect()),
        (Data::I32(a), Data::I32(b)) => Data::I32((0..n).map(|i| if cond[ci[i]] { a[xi[i]] } else { b[yi[i]] }).collect()),
        (Data::I8(a), Data::I8(b)) => Data::I8((0..n).map(|i| if cond[ci[i]] { a[xi[i]] } else { b[yi[i]] }).collect()),
        (Data::U8(a), Data::U8(b)) => Data::U8((0..n).map(|i| if cond[ci[i]] { a[xi[i]] } else { b[yi[i]] }).collect()),
        (Data::Bool(a), Data::Bool(b)) => Data::Bool((0..n).map(|i| if cond[ci[i]] { a[xi[i]] } else { b[yi[i]] }).collect()),
        _ => unreachable!("where after cast"),
    };
    Ok(HostTensor { shape, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_rules() {
        assert_eq!(resolve_reshape(&[2, 3, 4], &[0, -1], false).unwrap(), vec![2, 12]);
        assert_eq!(resolve_reshape(&[6], &[2, 3], false).unwrap(), vec![2, 3]);
        assert!(resolve_reshape(&[6], &[4, -1], false).is_err());
    }

    #[test]
    fn unsqueeze_squeeze() {
        assert_eq!(unsqueeze_shape(&[3], &[0, 2]).unwrap(), vec![1, 3, 1]);
        assert_eq!(unsqueeze_shape(&[3], &[-1]).unwrap(), vec![3, 1]);
        assert_eq!(squeeze_shape(&[1, 3, 1], Some(&[0])).unwrap(), vec![3, 1]);
        assert_eq!(squeeze_shape(&[1, 3, 1], None).unwrap(), vec![3]);
    }

    #[test]
    fn transpose_2d() {
        let x = HostTensor::f32(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let t = transpose(&x, &[1, 0]).unwrap();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.as_f32().to_vec(), vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn concat_axis1() {
        let a = HostTensor::i64(vec![2, 1], vec![1, 2]);
        let b = HostTensor::i64(vec![2, 2], vec![3, 4, 5, 6]);
        let c = concat(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(c.as_i64().to_vec(), vec![1, 3, 4, 2, 5, 6]);
    }

    #[test]
    fn slice_negative_and_step() {
        let x = HostTensor::i64(vec![5], vec![0, 1, 2, 3, 4]);
        let p = slice_plan(&[5], &[-3], &[i64::MAX], None, None).unwrap();
        assert_eq!(slice(&x, &p).as_i64().to_vec(), vec![2, 3, 4]);
        let p = slice_plan(&[5], &[0], &[5], None, Some(&[2])).unwrap();
        assert_eq!(slice(&x, &p).as_i64().to_vec(), vec![0, 2, 4]);
        let p = slice_plan(&[5], &[4], &[-6], None, Some(&[-1])).unwrap();
        assert_eq!(slice(&x, &p).as_i64().to_vec(), vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn gather_rows_and_select() {
        let x = HostTensor::f32(vec![3, 2], vec![0., 1., 10., 11., 20., 21.]);
        let idx = HostTensor::i64(vec![2], vec![2, 0]);
        let g = gather(&x, &idx, 0).unwrap();
        assert_eq!(g.shape, vec![2, 2]);
        assert_eq!(g.as_f32().to_vec(), vec![20., 21., 0., 1.]);
        let scalar = HostTensor::const_i64(1);
        let g = gather(&x, &scalar, 1).unwrap();
        assert_eq!(g.shape, vec![3]);
        assert_eq!(g.as_f32().to_vec(), vec![1., 11., 21.]);
    }

    #[test]
    fn range_and_where() {
        let r = range(&HostTensor::const_i64(0), &HostTensor::const_i64(5), &HostTensor::const_i64(2)).unwrap();
        assert_eq!(r.as_i64().to_vec(), vec![0, 2, 4]);
        let c = HostTensor::bool(vec![2], vec![true, false]);
        let w = where_(&c, &HostTensor::scalar_f32(1.0), &HostTensor::scalar_f32(-1.0)).unwrap();
        assert_eq!(w.as_f32().to_vec(), vec![1.0, -1.0]);
    }
}
