//! Generic ONNX normalization, indexing and resampling emitters.
use crate::error::{Error, Result};
use crate::exec::ggml::{self, contig, dev};
use crate::exec::ops_binary::binary;
use crate::exec::ops_shape::gather;
use crate::exec::runtime::{In, Run};
use crate::exec::value::{DeviceTensor, Value};
use crate::host::eval::norm_axis;
use crate::host::eval_resize::Resize;
use crate::host::tensor::HostTensor;
use crate::ir::Node;
use ggml_sys as g;

pub const DEVICE_OPS: &[&str] = &["LeakyRelu", "InstanceNormalization", "Resize", "CumSum", "Flatten", "Pad", "Pow"];
// These have host implementations, but no faithful ggml kernel yet. In
// particular ggml_round uses ties-away while ONNX Round requires ties-to-even.
pub const HOST_OPS: &[&str] = &[
    "IsNaN",
    "Round",
    "Atan",
    "GatherElements",
    "ScatterElements",
    "ScatterND",
    "TopK",
    "RandomNormalLike",
    "RandomUniformLike",
];

fn need<'a>(n: &Node, ins: &'a [Option<In>], i: usize) -> Result<&'a In> {
    ins.get(i).and_then(|v| v.as_ref()).ok_or_else(|| Error::model(format!("{n}: missing input {i}")))
}

pub fn emit(run: &mut Run, n: &Node, ins: &[Option<In>]) -> Result<Vec<Value>> {
    if crate::host::eval_control::OPS.contains(&n.op.as_str()) {
        return Err(Error::unsupported("control flow and sequence bookkeeping execute in the provider's host interpreter"));
    }
    if HOST_OPS.contains(&n.op.as_str()) {
        return Err(Error::unsupported(format!("{} has no ggml emitter", n.op)));
    }
    let x = run.dev_f32(need(n, ins, 0)?)?;
    let ctx = run.ctx;
    let out = match n.op.as_str() {
        "Pow" => {
            let exponent = run.host_param(need(n, ins, 1)?, "Pow exponent")?;
            if exponent.numel() != 1 || exponent.rank() > x.rank {
                return Err(Error::unsupported("device Pow requires a scalar exponent without rank expansion"));
            }
            let e = exponent.scalar_f64()?;
            if !e.is_finite() || e.fract() != 0. || !(-16.0..=16.0).contains(&e) {
                return Err(Error::unsupported("device Pow supports small integer scalar exponents"));
            }
            let mut power = e.abs() as u32;
            let mut base = x;
            let mut result = None;
            while power != 0 {
                if power & 1 != 0 {
                    result = Some(match result {
                        None => base,
                        Some(y) => binary(run, "Mul", y, base, &x.shape())?,
                    });
                }
                power >>= 1;
                if power != 0 {
                    base = DeviceTensor { t: unsafe { g::ggml_sqr(ctx, contig(ctx, base).t) }, ..base };
                }
            }
            let y = match result {
                Some(y) => y,
                None => {
                    let one = run.upload(&HostTensor::f32(vec![], vec![1.]), "pow_one")?;
                    unsafe { ggml::repeat_to(ctx, one, &x.shape())? }
                }
            };
            if e < 0. {
                let one = run.upload(&HostTensor::f32(vec![], vec![1.]), "pow_one")?;
                binary(run, "Div", one, y, &x.shape())?
            } else {
                y
            }
        }
        "LeakyRelu" => {
            DeviceTensor { t: unsafe { g::ggml_leaky_relu(ctx, contig(ctx, x).t, n.attr_f("alpha", 0.01), false) }, ..x }
        }
        "Flatten" => {
            let axis = n.attr_i("axis", 1);
            let axis = if axis < 0 { axis + x.rank as i64 } else { axis };
            if axis < 0 || axis > x.rank as i64 {
                return Err(Error::shape("Flatten axis"));
            }
            let shape = x.shape();
            unsafe { ggml::reshape(ctx, x, &[shape[..axis as usize].iter().product(), shape[axis as usize..].iter().product()])? }
        }
        "InstanceNormalization" => {
            if x.rank < 3 {
                return Err(Error::shape("InstanceNormalization rank"));
            }
            let s = run.dev_f32(need(n, ins, 1)?)?;
            let b = run.dev_f32(need(n, ins, 2)?)?;
            let shape = x.shape();
            let (batch, channels) = (shape[0], shape[1]);
            let spatial: usize = shape[2..].iter().product();
            if s.shape() != [channels] || b.shape() != [channels] {
                return Err(Error::shape("InstanceNormalization scale/bias"));
            }
            unsafe {
                let flat = ggml::reshape(ctx, contig(ctx, x), &[batch, channels, spatial])?;
                let normalized = dev(g::ggml_norm(ctx, flat.t, n.attr_f("epsilon", 1e-5)), &[batch, channels, spatial]);
                let s = ggml::reshape(ctx, s, &[1, channels, 1])?;
                let b = ggml::reshape(ctx, b, &[1, channels, 1])?;
                let y = binary(run, "Mul", normalized, s, &normalized.shape())?;
                let y = binary(run, "Add", y, b, &y.shape())?;
                ggml::reshape(ctx, y, &shape)?
            }
        }
        "Resize" => {
            let param = |i| ins.get(i).and_then(|v: &Option<In>| v.as_ref()).and_then(|v| v.v.host());
            let resize = Resize::new(n, &x.shape(), param(2), param(3))?;
            let mut x = x;
            for axis in 0..x.rank {
                if x.shape[axis] == resize.shape[axis] && resize.scales[axis] == 1. {
                    continue;
                }
                let (lo, hi, weights) = resize.coordinates(n, axis, x.shape[axis])?;
                let low = gather(run, x, &HostTensor::i64(vec![lo.len()], lo), axis)?;
                if resize.linear {
                    let high = gather(run, x, &HostTensor::i64(vec![hi.len()], hi), axis)?;
                    let shape = low.shape();
                    let mut wshape = vec![1; x.rank];
                    wshape[axis] = weights.len();
                    let w = run.upload(&HostTensor::f32(wshape.clone(), weights.clone()), "resize_weight")?;
                    let iw =
                        run.upload(&HostTensor::f32(wshape, weights.iter().map(|w| 1. - w).collect()), "resize_inverse_weight")?;
                    let a = binary(run, "Mul", low, iw, &shape)?;
                    let b = binary(run, "Mul", high, w, &shape)?;
                    x = binary(run, "Add", a, b, &shape)?;
                } else {
                    x = low;
                }
            }
            x
        }
        "CumSum" => {
            let axis = norm_axis(run.host_param(need(n, ins, 1)?, "CumSum axis")?.scalar_i64()?, x.rank)?;
            let mut perm: Vec<usize> = (0..x.rank).collect();
            perm.swap(axis, x.rank - 1);
            let mut x = x;
            let shape = x.shape();
            let dim = shape[axis];
            if n.attr_i("reverse", 0) != 0 {
                x = gather(run, x, &HostTensor::i64(vec![dim], (0..dim as i64).rev().collect()), axis)?;
            }
            unsafe {
                let transposed = contig(ctx, ggml::permute(ctx, x, &perm)?);
                let mut y = DeviceTensor { t: g::ggml_cumsum(ctx, transposed.t), ..transposed };
                if n.attr_i("exclusive", 0) != 0 {
                    let mut padded_shape = y.shape();
                    *padded_shape.last_mut().unwrap() += 1;
                    let padded = dev(g::ggml_pad_ext(ctx, y.t, 1, 0, 0, 0, 0, 0, 0, 0), &padded_shape);
                    y = ggml::view_slice(ctx, padded, &vec![0; y.rank], &y.shape())?;
                }
                y = ggml::permute(ctx, y, &perm)?;
                if n.attr_i("reverse", 0) != 0 {
                    y = gather(run, y, &HostTensor::i64(vec![dim], (0..dim as i64).rev().collect()), axis)?;
                }
                y
            }
        }
        "Pad" => pad(run, n, ins, x)?,
        _ => return Err(Error::unsupported(&n.op)),
    };
    unsafe {
        ggml::set_name(out.t, &n.outputs[0]);
    }
    Ok(vec![Value::Device(out)])
}

fn pad(run: &mut Run, n: &Node, ins: &[Option<In>], mut x: DeviceTensor) -> Result<DeviceTensor> {
    let pads = run.host_param(need(n, ins, 1)?, "Pad pads")?.as_i64().into_owned();
    if pads.len() != 2 * x.rank {
        return Err(Error::shape("Pad rank"));
    }
    let mode = n.attr_str("mode").unwrap_or("constant");
    let fill = match ins.get(2).and_then(|v| v.as_ref()) {
        Some(v) => run.host_param(v, "Pad value")?.scalar_f64()? as f32,
        None => 0.,
    };
    // Native zero padding avoids multiplying NaNs or infinities by a mask.
    if mode == "constant" {
        if fill != 0. || pads.iter().any(|&p| p < 0 || p > i32::MAX as i64) {
            return Err(Error::unsupported("device Pad constant value/cropping"));
        }
        let mut p = [0i32; 8];
        let mut shape = x.shape();
        for a in 0..x.rank {
            let gaxis = x.rank - 1 - a;
            p[2 * gaxis] = pads[a] as i32;
            p[2 * gaxis + 1] = pads[a + x.rank] as i32;
            shape[a] += (pads[a] + pads[a + x.rank]) as usize;
        }
        let x = unsafe { contig(run.ctx, x) };
        return Ok(dev(unsafe { g::ggml_pad_ext(run.ctx, x.t, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]) }, &shape));
    }
    for axis in 0..x.rank {
        let dim = x.shape[axis] as i64;
        let before = pads[axis];
        let after = pads[axis + x.rank];
        if before == 0 && after == 0 {
            continue;
        }
        if dim + before + after <= 0 {
            return Err(Error::unsupported("device Pad empty dimension"));
        }
        let mut ids = Vec::new();
        for i in 0..dim + before + after {
            let mut c = i - before;
            if c < 0 || c >= dim {
                c = match mode {
                    "edge" if dim > 0 => c.clamp(0, dim - 1),
                    "reflect" if dim > 1 => {
                        let v = c.rem_euclid(2 * (dim - 1));
                        if v >= dim {
                            2 * (dim - 1) - v
                        } else {
                            v
                        }
                    }
                    _ => return Err(Error::unsupported("device Pad mode")),
                };
            }
            ids.push(c);
        }
        x = gather(run, x, &HostTensor::i64(vec![ids.len()], ids), axis)?;
    }
    Ok(x)
}
