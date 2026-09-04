//! Elementwise ops, activations, softmax, layer norm, reductions and Where.
//!
//! ggml's binary ops broadcast `b` into `a` only (`b` repeats to `a`'s shape),
//! while ONNX broadcasts both ways, so `broadcast_pair` first repeats whichever
//! side is smaller than the result.

use ggml_sys as g;

use crate::error::{Error, Result};
use crate::exec::ggml::{self, contig, dev};
use crate::exec::runtime::{In, Run};
use crate::exec::value::{DeviceTensor, Value};
use crate::host::broadcast::{broadcast_all, broadcast_shapes, broadcasts_into};
use crate::host::eval::norm_axis;
use crate::host::tensor::HostTensor;
use crate::ir::{DType, Node};

fn need<'a>(node: &Node, ins: &'a [Option<In>], i: usize) -> Result<&'a In> {
    ins.get(i).and_then(|x| x.as_ref()).ok_or_else(|| Error::model(format!("{node}: missing input {i}")))
}

pub fn emit(run: &mut Run, node: &Node, ins: &[Option<In>]) -> Result<Vec<Value>> {
    let op = node.op.as_str();
    let out = match op {
        "Add" | "Sub" | "Mul" | "Div" => {
            let a = need(node, ins, 0)?;
            let b = need(node, ins, 1)?;
            let shape = broadcast_shapes(&a.v.shape(), &b.v.shape())?;
            let da = run.dev_f32(a)?;
            let db = run.dev_f32(b)?;
            binary(run, op, da, db, &shape)?
        }
        "Sqrt" | "Exp" | "Log" | "Sin" | "Cos" | "Sigmoid" | "Elu" | "Relu" | "Tanh" | "Neg" | "Abs" | "GeluErf" => {
            if op == "Elu" && (node.attr_f("alpha", 1.0) - 1.0).abs() > 1e-6 {
                return Err(Error::unsupported("Elu with alpha != 1"));
            }
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let x = unsafe { contig(run.ctx, x) };
            let ctx = run.ctx;
            let t = unsafe {
                match op {
                    "Sqrt" => g::ggml_sqrt(ctx, x.t),
                    "Exp" => g::ggml_exp(ctx, x.t),
                    "Log" => g::ggml_log(ctx, x.t),
                    "Sin" => g::ggml_sin(ctx, x.t),
                    "Cos" => g::ggml_cos(ctx, x.t),
                    "Sigmoid" => g::ggml_sigmoid(ctx, x.t),
                    "Elu" => g::ggml_elu(ctx, x.t),
                    "Relu" => g::ggml_relu(ctx, x.t),
                    "Tanh" => g::ggml_tanh(ctx, x.t),
                    "Neg" => g::ggml_neg(ctx, x.t),
                    "Abs" => g::ggml_abs(ctx, x.t),
                    "GeluErf" => g::ggml_gelu_erf(ctx, x.t),
                    _ => unreachable!(),
                }
            };
            DeviceTensor { t, ..x }
        }
        "Erf" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            erf_approx(run, x)
        }
        "Reciprocal" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let x = unsafe { contig(run.ctx, x) };
            let one = run.scalar(1.0)?;
            let ones = unsafe { ggml::repeat_to(run.ctx, one, &x.shape())? };
            DeviceTensor { t: unsafe { g::ggml_div(run.ctx, ones.t, x.t) }, ..x }
        }
        "Softmax" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let axis = norm_axis(node.attr_i("axis", -1), x.rank.max(1))?;
            softmax(run, x, axis)?
        }
        "LayerNormalization" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let scale = run.dev_f32(need(node, ins, 1)?)?;
            let bias = match ins.get(2).and_then(|b| b.as_ref()) {
                Some(b) => Some(run.dev_f32(b)?),
                None => None,
            };
            let axis = norm_axis(node.attr_i("axis", -1), x.rank.max(1))?;
            layer_norm(run, x, scale, bias, axis, node.attr_f("epsilon", 1e-5))?
        }
        "ReduceMean" | "ReduceSum" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let axes = match ins.get(1).and_then(|a| a.as_ref()) {
                Some(a) => run.host_param(a, "reduce axes")?.as_i64().to_vec(),
                None => node.attr_ints("axes").unwrap_or_default(),
            };
            if axes.len() != 1 {
                return Err(Error::unsupported(format!("{op} over {} axes", axes.len())));
            }
            let axis = norm_axis(axes[0], x.rank.max(1))?;
            reduce_last(run, op, x, axis, node.attr_i("keepdims", 1) != 0)?
        }
        "Where" => {
            let c = need(node, ins, 0)?;
            let x = need(node, ins, 1)?;
            let y = need(node, ins, 2)?;
            where_(run, c, x, y)?
        }
        "Cast" => {
            let to = DType::from_onnx(node.attr_i("to", 1) as i32)?;
            if !to.is_float() {
                return Err(Error::unsupported("cast to int on device"));
            }
            let x = need(node, ins, 0)?;
            return Ok(vec![match &x.v {
                Value::Device(d) => Value::Device(*d),
                Value::Staged(t) | Value::Host(t) => Value::staged_of(t.cast(DType::F32)),
            }]);
        }
        "Clip" => {
            let x = run.dev_f32(need(node, ins, 0)?)?;
            let lo = match ins.get(1).and_then(|a| a.as_ref()) {
                Some(a) => run.host_param(a, "clip min")?.scalar_f64()? as f32,
                None => f32::NEG_INFINITY,
            };
            let hi = match ins.get(2).and_then(|a| a.as_ref()) {
                Some(a) => run.host_param(a, "clip max")?.scalar_f64()? as f32,
                None => f32::INFINITY,
            };
            let x = unsafe { contig(run.ctx, x) };
            DeviceTensor { t: unsafe { g::ggml_clamp(run.ctx, x.t, lo, hi) }, ..x }
        }
        other => return Err(Error::unsupported(format!("device op {other}"))),
    };
    unsafe { ggml::set_name(out.t, &node.outputs[0]) };
    Ok(vec![Value::Device(out)])
}

/// Shape `a` and `b` for a ggml binary op producing `out`: `a` must equal
/// `out`, `b` must broadcast into it.
pub fn broadcast_pair(run: &mut Run, a: DeviceTensor, b: DeviceTensor, out: &[usize]) -> Result<(DeviceTensor, DeviceTensor)> {
    unsafe {
        // ggml's binary kernels read strided operands on every backend; only a
        // repeat needs a contiguous source.
        let a = if a.shape() == out { a } else { ggml::repeat_to(run.ctx, contig(run.ctx, a), out)? };
        let b = if broadcasts_into(&b.shape(), out) { b } else { ggml::repeat_to(run.ctx, contig(run.ctx, b), out)? };
        Ok((ggml::dense_rows(run.ctx, a), ggml::dense_rows(run.ctx, b)))
    }
}

pub fn binary(run: &mut Run, op: &str, a: DeviceTensor, b: DeviceTensor, out: &[usize]) -> Result<DeviceTensor> {
    let ctx = run.ctx;
    // commutative ops can put the larger operand first without a repeat
    let (a, b, swapped) =
        if a.shape() != out && b.shape() == out && matches!(op, "Add" | "Mul") { (b, a, true) } else { (a, b, false) };
    let _ = swapped;
    let (a, b) = broadcast_pair(run, a, b, out)?;
    let t = unsafe {
        match op {
            "Add" => g::ggml_add(ctx, a.t, b.t),
            "Sub" => g::ggml_sub(ctx, a.t, b.t),
            "Mul" => g::ggml_mul(ctx, a.t, b.t),
            "Div" => g::ggml_div(ctx, a.t, b.t),
            _ => unreachable!(),
        }
    };
    Ok(dev(t, out))
}

/// erf(x) ~ sgn(x) * sqrt(1 - exp(-x^2 (4/pi + a x^2) / (1 + a x^2))), a = 0.147.
/// Max error about 1.3e-4; a GELU-shaped Erf is fused at compile time instead.
fn erf_approx(run: &mut Run, x: DeviceTensor) -> DeviceTensor {
    const A: f32 = 0.147;
    let ctx = run.ctx;
    unsafe {
        let x = contig(ctx, x);
        let x2 = g::ggml_sqr(ctx, x.t);
        let num = g::ggml_mul(ctx, x2, g::ggml_scale_bias(ctx, x2, A, 4.0 / std::f32::consts::PI));
        let den = g::ggml_scale_bias(ctx, x2, A, 1.0);
        let r = g::ggml_div(ctx, num, den);
        let e = g::ggml_exp(ctx, g::ggml_neg(ctx, r));
        let s = g::ggml_sqrt(ctx, g::ggml_scale_bias(ctx, e, -1.0, 1.0));
        let t = g::ggml_mul(ctx, s, g::ggml_sgn(ctx, x.t));
        DeviceTensor { t, ..x }
    }
}

/// Move ONNX `axis` to the last position (a permutation that swaps two axes).
fn swap_to_last(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(axis, rank - 1);
    perm
}

fn softmax(run: &mut Run, x: DeviceTensor, axis: usize) -> Result<DeviceTensor> {
    let ctx = run.ctx;
    unsafe {
        if axis + 1 == x.rank || x.rank == 0 {
            let x = contig(ctx, x);
            return Ok(DeviceTensor { t: g::ggml_soft_max(ctx, x.t), ..x });
        }
        let perm = swap_to_last(x.rank, axis);
        let p = contig(ctx, ggml::permute(ctx, x, &perm)?);
        let s = DeviceTensor { t: g::ggml_soft_max(ctx, p.t), ..p };
        ggml::permute(ctx, s, &perm)
    }
}

fn layer_norm(
    run: &mut Run,
    x: DeviceTensor,
    scale: DeviceTensor,
    bias: Option<DeviceTensor>,
    axis: usize,
    eps: f32,
) -> Result<DeviceTensor> {
    let ctx = run.ctx;
    let shape = x.shape();
    let inner: usize = shape[axis..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    unsafe {
        let flat = if axis + 1 == x.rank { contig(ctx, x) } else { ggml::reshape(ctx, x, &[outer, inner])? };
        let normed = DeviceTensor { t: g::ggml_norm(ctx, flat.t, eps), ..flat };
        let s = if scale.numel() == inner {
            ggml::reshape(ctx, scale, &[inner])?
        } else {
            return Err(Error::shape("layernorm scale size"));
        };
        let mut y = DeviceTensor { t: g::ggml_mul(ctx, normed.t, s.t), ..normed };
        if let Some(b) = bias {
            let b = ggml::reshape(ctx, b, &[inner])?;
            y = DeviceTensor { t: g::ggml_add(ctx, y.t, b.t), ..y };
        }
        if axis + 1 == x.rank {
            Ok(y)
        } else {
            ggml::reshape(ctx, y, &shape)
        }
    }
}

fn reduce_last(run: &mut Run, op: &str, x: DeviceTensor, axis: usize, keepdims: bool) -> Result<DeviceTensor> {
    let ctx = run.ctx;
    let rank = x.rank.max(1);
    let mut in_shape = x.shape();
    if in_shape.is_empty() {
        in_shape.push(1);
    }
    unsafe {
        let perm = swap_to_last(rank, axis);
        let src = if axis + 1 == rank { contig(ctx, x) } else { contig(ctx, ggml::permute(ctx, x, &perm)?) };
        let t = match op {
            "ReduceMean" => g::ggml_mean(ctx, src.t),
            _ => g::ggml_sum_rows(ctx, src.t),
        };
        let mut reduced_shape = src.shape();
        *reduced_shape.last_mut().unwrap() = 1;
        let mut r = dev(t, &reduced_shape);
        if axis + 1 != rank {
            r = ggml::permute(ctx, r, &perm)?;
        }
        let mut out_shape = in_shape.clone();
        out_shape[axis] = 1;
        if !keepdims {
            out_shape.remove(axis);
        }
        ggml::reshape(ctx, r, &out_shape)
    }
}

/// The single huge/non-finite value a host tensor is filled with, if it is one.
///
/// Attention mask fills arrive both as an `-inf` scalar and, from
/// `ConstantOfShape`, as a whole tensor of `-inf` (pocket-tts does the latter: a
/// `[1, 16, S, S]` block per attention layer). Either way `where_` must not
/// multiply it by a 0/1 mask, because `-inf * 0` is NaN; the additive-bias path
/// below keeps it exact. Uniformity is what matters here, not the element count.
fn huge_fill(t: &HostTensor) -> Option<f32> {
    if !matches!(t.dtype(), DType::F32 | DType::F16 | DType::F64) {
        return None;
    }
    let v = t.as_f32();
    let first = *v.first()?;
    if first.is_nan() || (first.is_finite() && first.abs() < 1e30) {
        return None;
    }
    v.iter().all(|&x| x == first).then_some(first)
}

fn where_(run: &mut Run, c: &In, x: &In, y: &In) -> Result<DeviceTensor> {
    let cond = run.host_param(c, "where condition")?.clone();
    let out = broadcast_all(&[&cond.shape, &x.v.shape(), &y.v.shape()])?;
    let cb = cond.as_bool();
    // c ? x : y  with y filled with a huge/inf value: x + (c ? 0 : y). This keeps
    // -inf exact for masks and uploads only the (much smaller) cond-shaped bias.
    if let Some(yv) = y.v.host().and_then(huge_fill) {
        let bias = HostTensor::f32(cond.shape.clone(), cb.iter().map(|&b| if b { 0.0 } else { yv }).collect());
        let bd = run.upload(&bias, "where_bias")?;
        let xd = run.dev_f32(x)?;
        return binary(run, "Add", xd, bd, &out);
    }
    if let Some(xv) = x.v.host().and_then(huge_fill) {
        let bias = HostTensor::f32(cond.shape.clone(), cb.iter().map(|&b| if b { xv } else { 0.0 }).collect());
        let bd = run.upload(&bias, "where_bias")?;
        let yd = run.dev_f32(y)?;
        return binary(run, "Add", yd, bd, &out);
    }
    // general: x*c + y*(1-c)
    let cf = HostTensor::f32(cond.shape.clone(), cb.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect());
    let nf = HostTensor::f32(cond.shape.clone(), cb.iter().map(|&b| if b { 0.0 } else { 1.0 }).collect());
    let cd = run.upload(&cf, "where_c")?;
    let nd = run.upload(&nf, "where_notc")?;
    let xd = run.dev_f32(x)?;
    let yd = run.dev_f32(y)?;
    let xm = binary(run, "Mul", xd, cd, &out)?;
    let ym = binary(run, "Mul", yd, nd, &out)?;
    binary(run, "Add", xm, ym, &out)
}
