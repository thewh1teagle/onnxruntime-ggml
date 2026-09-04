//! Structural ops on device tensors. Most are views (no copy): reshape after
//! a `cont`, permute, strided slices, splits. Concat and Expand copy.

use ggml_sys as g;

use crate::error::{Error, Result};
use crate::exec::ggml::{self, contig, dev, gaxis};
use crate::exec::runtime::{In, Run};
use crate::exec::value::{DeviceTensor, Value};
use crate::host::broadcast::broadcast_shapes;
use crate::host::eval::norm_axis;
use crate::host::eval_shape::{resolve_reshape, slice_plan, squeeze_shape, unsqueeze_shape};
use crate::ir::{DType, Node};

fn need<'a>(node: &Node, ins: &'a [Option<In>], i: usize) -> Result<&'a In> {
    ins.get(i).and_then(|x| x.as_ref()).ok_or_else(|| Error::model(format!("{node}: missing input {i}")))
}

pub fn emit(run: &mut Run, node: &Node, ins: &[Option<In>]) -> Result<Vec<Value>> {
    let op = node.op.as_str();
    let outs: Vec<DeviceTensor> = match op {
        "Identity" => {
            let x = need(node, ins, 0)?;
            return Ok(vec![x.v.clone()]);
        }
        "Reshape" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let shape = run.host_param(need(node, ins, 1)?, "reshape shape")?.as_i64().to_vec();
            let target = resolve_reshape(&x.shape(), &shape, node.attr_i("allowzero", 0) != 0)?;
            vec![unsafe { ggml::reshape(run.ctx, x, &target)? }]
        }
        "Unsqueeze" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let axes = match ins.get(1).and_then(|a| a.as_ref()) {
                Some(a) => run.host_param(a, "unsqueeze axes")?.as_i64().to_vec(),
                None => node.attr_ints("axes").ok_or_else(|| Error::model("Unsqueeze without axes"))?,
            };
            let target = unsqueeze_shape(&x.shape(), &axes)?;
            vec![unsafe { ggml::reshape(run.ctx, x, &target)? }]
        }
        "Squeeze" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let axes = match ins.get(1).and_then(|a| a.as_ref()) {
                Some(a) => Some(run.host_param(a, "squeeze axes")?.as_i64().to_vec()),
                None => node.attr_ints("axes"),
            };
            let target = squeeze_shape(&x.shape(), axes.as_deref())?;
            vec![unsafe { ggml::reshape(run.ctx, x, &target)? }]
        }
        "Transpose" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let perm: Vec<usize> = match node.attr_ints("perm") {
                Some(p) => p.iter().map(|&a| norm_axis(a, x.rank)).collect::<Result<_>>()?,
                None => (0..x.rank).rev().collect(),
            };
            vec![unsafe { ggml::permute(run.ctx, x, &perm)? }]
        }
        "Slice" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let starts = run.host_param(need(node, ins, 1)?, "slice starts")?.as_i64().to_vec();
            let ends = run.host_param(need(node, ins, 2)?, "slice ends")?.as_i64().to_vec();
            let axes = match ins.get(3).and_then(|a| a.as_ref()) {
                Some(a) => Some(run.host_param(a, "slice axes")?.as_i64().to_vec()),
                None => None,
            };
            let steps = match ins.get(4).and_then(|a| a.as_ref()) {
                Some(a) => Some(run.host_param(a, "slice steps")?.as_i64().to_vec()),
                None => None,
            };
            let plan = slice_plan(&x.shape(), &starts, &ends, axes.as_deref(), steps.as_deref())?;
            if plan.step.iter().any(|&s| s != 1) {
                return Err(Error::unsupported("slice with step != 1"));
            }
            if plan.count.iter().any(|&c| c == 0) {
                return Err(Error::unsupported("empty slice"));
            }
            vec![unsafe { ggml::view_slice(run.ctx, x, &plan.start, &plan.count)? }]
        }
        "Concat" => {
            let present: Vec<&In> = ins.iter().flatten().collect();
            if present.is_empty() {
                return Err(Error::model("Concat without inputs"));
            }
            let rank = present[0].v.rank();
            let axis = norm_axis(node.attr_i("axis", 0), rank)?;
            let mut acc: Option<DeviceTensor> = None;
            for i in present {
                if i.v.numel_is_zero() {
                    continue;
                }
                let d = run.dev_f32(i)?;
                let d = unsafe { contig(run.ctx, d) };
                acc = Some(match acc {
                    None => d,
                    Some(a) => {
                        let mut shape = a.shape();
                        shape[axis] += d.shape[axis];
                        let t = unsafe { g::ggml_concat(run.ctx, a.t, d.t, gaxis(axis, rank)) };
                        dev(t, &shape)
                    }
                });
            }
            vec![acc.ok_or_else(|| Error::unsupported("concat of only empty tensors"))?]
        }
        "Gather" => {
            let data = need(node, ins, 0)?;
            let idx = run.host_param(need(node, ins, 1)?, "gather indices")?.clone();
            let rank = data.v.rank();
            let axis = norm_axis(node.attr_i("axis", 0), rank)?;
            let x = run.dev_f32(data)?;
            vec![gather(run, x, &idx, axis)?]
        }
        "Split" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let axis = norm_axis(node.attr_i("axis", 0), x.rank)?;
            let total = x.shape[axis];
            let sizes: Vec<usize> = match ins.get(1).and_then(|a| a.as_ref()) {
                Some(a) => run.host_param(a, "split sizes")?.as_i64().iter().map(|&v| v as usize).collect(),
                None => {
                    let n = node.attr_i("num_outputs", node.outputs.len() as i64) as usize;
                    let chunk = total.div_ceil(n);
                    (0..n).map(|i| chunk.min(total.saturating_sub(chunk * i))).collect()
                }
            };
            if sizes.iter().sum::<usize>() != total {
                return Err(Error::shape(format!("split sizes {sizes:?} vs {total}")));
            }
            let mut outs = Vec::with_capacity(sizes.len());
            let mut start = vec![0usize; x.rank];
            for &size in &sizes {
                let mut count = x.shape();
                count[axis] = size;
                outs.push(unsafe { ggml::view_slice(run.ctx, x, &start, &count)? });
                start[axis] += size;
            }
            outs
        }
        "Expand" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let target: Vec<usize> = run.host_param(need(node, ins, 1)?, "expand shape")?.as_i64().iter().map(|&v| v as usize).collect();
            let shape = broadcast_shapes(&x.shape(), &target)?;
            if shape == x.shape() {
                vec![x]
            } else {
                let x = unsafe { contig(run.ctx, x) };
                vec![unsafe { ggml::repeat_to(run.ctx, x, &shape)? }]
            }
        }
        other => return Err(Error::unsupported(format!("device shape op {other}"))),
    };
    let mut values = Vec::with_capacity(outs.len());
    for (d, name) in outs.into_iter().zip(node.outputs.iter()) {
        unsafe { ggml::set_name(d.t, name) };
        values.push(Value::Device(d));
    }
    Ok(values)
}

trait ValueExt {
    fn numel_is_zero(&self) -> bool;
}

impl ValueExt for Value {
    fn numel_is_zero(&self) -> bool {
        self.shape().iter().any(|&d| d == 0)
    }
}

/// Two Gather shapes ggml can serve: a single index (a strided view) and
/// row lookup on a 2-D table (`ggml_get_rows`). Anything else is declined and
/// the runtime evaluates it on the host.
fn gather(run: &mut Run, x: DeviceTensor, idx: &crate::host::tensor::HostTensor, axis: usize) -> Result<DeviceTensor> {
    let ctx = run.ctx;
    let dim = x.shape[axis] as i64;
    if idx.numel() == 1 {
        let mut k = idx.scalar_i64()?;
        if k < 0 {
            k += dim;
        }
        if k < 0 || k >= dim {
            return Err(Error::shape(format!("gather index {k} out of range {dim}")));
        }
        let mut start = vec![0usize; x.rank];
        let mut count = x.shape();
        start[axis] = k as usize;
        count[axis] = 1;
        let v = unsafe { ggml::view_slice(ctx, x, &start, &count)? };
        if idx.rank() == 0 {
            let mut shape = x.shape();
            shape.remove(axis);
            return unsafe { ggml::reshape(ctx, v, &shape) };
        }
        return Ok(v);
    }
    if x.rank == 2 && axis == 0 {
        let rows = x.shape[0] as i64;
        let cols = x.shape[1];
        let ids: Vec<i64> = idx.as_i64().iter().map(|&i| if i < 0 { i + rows } else { i }).collect();
        if ids.iter().any(|&i| i < 0 || i >= rows) {
            return Err(Error::shape("gather index out of range"));
        }
        let flat = crate::host::tensor::HostTensor::i32(vec![ids.len()], ids.iter().map(|&i| i as i32).collect());
        let id = run.upload(&flat, "gather_idx")?;
        let x = unsafe { contig(ctx, x) };
        let t = unsafe { g::ggml_get_rows(ctx, x.t, id.t) };
        let mut shape = idx.shape.clone();
        shape.push(cols);
        return unsafe { ggml::reshape(ctx, dev(t, &[ids.len(), cols]), &shape) };
    }
    let _ = DType::I32;
    Err(Error::unsupported(format!("gather axis {axis} on rank {} with {} indices", x.rank, idx.numel())))
}
