//! Elementwise, comparison, reduction and softmax ops on host tensors.
//! Integer inputs stay integer through Add/Sub/Mul/Div; everything else
//! computes in f32.

use crate::error::{Error, Result};
use crate::host::broadcast::{broadcast_index, broadcast_shapes};
use crate::host::eval::{need, norm_axis};
use crate::host::tensor::{numel_of, Data, HostTensor};
use crate::ir::{DType, Node};

pub fn eval(node: &Node, inputs: &[Option<&HostTensor>]) -> Result<Vec<HostTensor>> {
    let op = node.op.as_str();
    let out = match op {
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Max" | "Min" => {
            let a = need(node, inputs, 0)?;
            let b = need(node, inputs, 1)?;
            binary(op, a, b)?
        }
        "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" | "Equal" | "And" | "Or" => {
            let a = need(node, inputs, 0)?;
            let b = need(node, inputs, 1)?;
            compare(op, a, b)?
        }
        "Not" => {
            let a = need(node, inputs, 0)?;
            HostTensor::bool(a.shape.clone(), a.as_bool().iter().map(|v| !v).collect())
        }
        "Neg" | "Abs" | "Sqrt" | "Exp" | "Log" | "Sin" | "Cos" | "Tanh" | "Sigmoid" | "Elu" | "Relu" | "Erf"
        | "Reciprocal" | "Floor" | "Ceil" | "GeluErf" => {
            let a = need(node, inputs, 0)?;
            unary(op, a, node.attr_f("alpha", 1.0))?
        }
        "Clip" => {
            let a = need(node, inputs, 0)?;
            let lo = inputs.get(1).copied().flatten().map(|t| t.scalar_f64()).transpose()?.unwrap_or(f64::NEG_INFINITY);
            let hi = inputs.get(2).copied().flatten().map(|t| t.scalar_f64()).transpose()?.unwrap_or(f64::INFINITY);
            HostTensor::f32(a.shape.clone(), a.as_f32().iter().map(|&v| (v as f64).clamp(lo, hi) as f32).collect())
        }
        "ReduceMean" | "ReduceSum" | "ReduceProd" | "ReduceMax" | "ReduceMin" => {
            let a = need(node, inputs, 0)?;
            let axes = match inputs.get(1).copied().flatten() {
                Some(t) => Some(t.as_i64().to_vec()),
                None => node.attr_ints("axes"),
            };
            let keepdims = node.attr_i("keepdims", 1) != 0;
            reduce(op, a, axes.as_deref(), keepdims, node.attr_i("noop_with_empty_axes", 0) != 0)?
        }
        "Softmax" => {
            let a = need(node, inputs, 0)?;
            let axis = norm_axis(node.attr_i("axis", -1), a.rank())?;
            softmax(a, axis)?
        }
        other => return Err(Error::unsupported(format!("host math op {other}"))),
    };
    Ok(vec![out])
}

fn result_dtype(a: DType, b: DType) -> DType {
    if a == b {
        a
    } else if a.is_float() || b.is_float() {
        DType::F32
    } else {
        DType::I64
    }
}

pub fn binary(op: &str, a: &HostTensor, b: &HostTensor) -> Result<HostTensor> {
    let shape = broadcast_shapes(&a.shape, &b.shape)?;
    let ai = broadcast_index(&a.shape, &shape);
    let bi = broadcast_index(&b.shape, &shape);
    let n = numel_of(&shape);
    let dtype = result_dtype(a.dtype(), b.dtype());
    if dtype.is_float() || op == "Pow" {
        let x = a.as_f32();
        let y = b.as_f32();
        let f: fn(f32, f32) -> f32 = match op {
            "Add" => |x, y| x + y,
            "Sub" => |x, y| x - y,
            "Mul" => |x, y| x * y,
            "Div" => |x, y| x / y,
            "Pow" => |x, y| x.powf(y),
            "Max" => |x, y| x.max(y),
            "Min" => |x, y| x.min(y),
            _ => unreachable!(),
        };
        let data: Vec<f32> = (0..n).map(|i| f(x[ai[i]], y[bi[i]])).collect();
        let out = HostTensor::f32(shape, data);
        Ok(if dtype == DType::F64 { out.cast(DType::F64) } else { out })
    } else {
        let x = a.as_i64();
        let y = b.as_i64();
        let f: fn(i64, i64) -> i64 = match op {
            "Add" => |x, y| x.wrapping_add(y),
            "Sub" => |x, y| x.wrapping_sub(y),
            "Mul" => |x, y| x.wrapping_mul(y),
            "Div" => |x, y| if y == 0 { 0 } else { x / y },
            "Max" => |x, y| x.max(y),
            "Min" => |x, y| x.min(y),
            _ => unreachable!(),
        };
        let data: Vec<i64> = (0..n).map(|i| f(x[ai[i]], y[bi[i]])).collect();
        Ok(HostTensor::i64(shape, data).cast(dtype))
    }
}

pub fn compare(op: &str, a: &HostTensor, b: &HostTensor) -> Result<HostTensor> {
    let shape = broadcast_shapes(&a.shape, &b.shape)?;
    let ai = broadcast_index(&a.shape, &shape);
    let bi = broadcast_index(&b.shape, &shape);
    let n = numel_of(&shape);
    let data: Vec<bool> = match op {
        "And" | "Or" => {
            let x = a.as_bool();
            let y = b.as_bool();
            if op == "And" {
                (0..n).map(|i| x[ai[i]] && y[bi[i]]).collect()
            } else {
                (0..n).map(|i| x[ai[i]] || y[bi[i]]).collect()
            }
        }
        _ => {
            let x = a.as_f64();
            let y = b.as_f64();
            let f: fn(f64, f64) -> bool = match op {
                "Greater" => |x, y| x > y,
                "GreaterOrEqual" => |x, y| x >= y,
                "Less" => |x, y| x < y,
                "LessOrEqual" => |x, y| x <= y,
                "Equal" => |x, y| x == y,
                _ => unreachable!(),
            };
            (0..n).map(|i| f(x[ai[i]], y[bi[i]])).collect()
        }
    };
    Ok(HostTensor::bool(shape, data))
}

/// Abramowitz-Stegun style erf, accurate to about 1e-7; the same formula the
/// ggml emitter approximates when a lone Erf is not part of a GELU.
pub fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

pub fn unary(op: &str, a: &HostTensor, alpha: f32) -> Result<HostTensor> {
    if !a.dtype().is_float() && matches!(op, "Neg" | "Abs") {
        let v = a.as_i64();
        let data: Vec<i64> = match op {
            "Neg" => v.iter().map(|x| -x).collect(),
            _ => v.iter().map(|x| x.abs()).collect(),
        };
        return Ok(HostTensor::i64(a.shape.clone(), data).cast(a.dtype()));
    }
    let x = a.as_f32();
    let data: Vec<f32> = match op {
        "Neg" => x.iter().map(|v| -v).collect(),
        "Abs" => x.iter().map(|v| v.abs()).collect(),
        "Sqrt" => x.iter().map(|v| v.sqrt()).collect(),
        "Exp" => x.iter().map(|v| v.exp()).collect(),
        "Log" => x.iter().map(|v| v.ln()).collect(),
        "Sin" => x.iter().map(|v| v.sin()).collect(),
        "Cos" => x.iter().map(|v| v.cos()).collect(),
        "Tanh" => x.iter().map(|v| v.tanh()).collect(),
        "Sigmoid" => x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        "Elu" => x.iter().map(|&v| if v >= 0.0 { v } else { alpha * (v.exp() - 1.0) }).collect(),
        "Relu" => x.iter().map(|v| v.max(0.0)).collect(),
        "Erf" => x.iter().map(|&v| erf(v)).collect(),
        "GeluErf" => x.iter().map(|&v| 0.5 * v * (1.0 + erf(v / std::f32::consts::SQRT_2))).collect(),
        "Reciprocal" => x.iter().map(|v| 1.0 / v).collect(),
        "Floor" => x.iter().map(|v| v.floor()).collect(),
        "Ceil" => x.iter().map(|v| v.ceil()).collect(),
        other => return Err(Error::unsupported(format!("host unary {other}"))),
    };
    let out = HostTensor::f32(a.shape.clone(), data);
    Ok(if a.dtype() == DType::F64 { out.cast(DType::F64) } else { out })
}

pub fn reduce(op: &str, a: &HostTensor, axes: Option<&[i64]>, keepdims: bool, noop_empty: bool) -> Result<HostTensor> {
    let rank = a.rank();
    let axes: Vec<usize> = match axes {
        Some(ax) if !ax.is_empty() => ax.iter().map(|&x| norm_axis(x, rank)).collect::<Result<_>>()?,
        Some(_) if noop_empty => return Ok(a.clone()),
        _ => (0..rank).collect(),
    };
    let out_shape_kept: Vec<usize> = (0..rank).map(|d| if axes.contains(&d) { 1 } else { a.shape[d] }).collect();
    let out_shape: Vec<usize> = if keepdims {
        out_shape_kept.clone()
    } else {
        (0..rank).filter(|d| !axes.contains(d)).map(|d| a.shape[d]).collect()
    };
    let n_out = numel_of(&out_shape_kept);
    // map every input element to its output slot
    let strides = a.strides();
    let out_strides = crate::host::tensor::strides_of(&out_shape_kept);
    let n_in = a.numel();
    let mut slot = Vec::with_capacity(n_in);
    for flat in 0..n_in {
        let mut rem = flat;
        let mut o = 0;
        for d in 0..rank {
            let i = rem / strides[d];
            rem %= strides[d];
            if !axes.contains(&d) {
                o += i * out_strides[d];
            }
        }
        slot.push(o);
    }
    let count = if n_out == 0 { 0 } else { n_in / n_out };
    let is_int = !a.dtype().is_float();
    if is_int && matches!(op, "ReduceProd" | "ReduceSum" | "ReduceMax" | "ReduceMin") {
        let x = a.as_i64();
        let init = match op {
            "ReduceProd" => 1,
            "ReduceSum" => 0,
            "ReduceMax" => i64::MIN,
            _ => i64::MAX,
        };
        let mut out = vec![init; n_out];
        for (i, &v) in x.iter().enumerate() {
            let o = &mut out[slot[i]];
            *o = match op {
                "ReduceProd" => o.wrapping_mul(v),
                "ReduceSum" => o.wrapping_add(v),
                "ReduceMax" => (*o).max(v),
                _ => (*o).min(v),
            };
        }
        return Ok(HostTensor::i64(out_shape, out).cast(a.dtype()));
    }
    let x = a.as_f32();
    let init = match op {
        "ReduceProd" => 1.0,
        "ReduceMax" => f32::NEG_INFINITY,
        "ReduceMin" => f32::INFINITY,
        _ => 0.0,
    };
    let mut out = vec![init; n_out];
    for (i, &v) in x.iter().enumerate() {
        let o = &mut out[slot[i]];
        *o = match op {
            "ReduceProd" => *o * v,
            "ReduceMax" => o.max(v),
            "ReduceMin" => o.min(v),
            _ => *o + v,
        };
    }
    if op == "ReduceMean" && count > 0 {
        for o in &mut out {
            *o /= count as f32;
        }
    }
    let t = HostTensor::f32(out_shape, out);
    Ok(if a.dtype() == DType::F64 { t.cast(DType::F64) } else { t })
}

pub fn softmax(a: &HostTensor, axis: usize) -> Result<HostTensor> {
    let x = a.as_f32();
    let outer: usize = a.shape[..axis].iter().product();
    let dim = a.shape[axis];
    let inner: usize = a.shape[axis + 1..].iter().product();
    let mut out = vec![0f32; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let idx = |k: usize| (o * dim + k) * inner + i;
            let max = (0..dim).map(|k| x[idx(k)]).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for k in 0..dim {
                let e = (x[idx(k)] - max).exp();
                out[idx(k)] = e;
                sum += e;
            }
            for k in 0..dim {
                out[idx(k)] /= sum;
            }
        }
    }
    Ok(HostTensor::f32(a.shape.clone(), out))
}

/// Used by tests and by the emitter's approximation check.
pub fn data_is_float(d: &Data) -> bool {
    matches!(d, Data::F32(_) | Data::F64(_))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_broadcast_and_int() {
        let a = HostTensor::i64(vec![2, 1], vec![1, 2]);
        let b = HostTensor::i64(vec![3], vec![10, 20, 30]);
        let c = binary("Add", &a, &b).unwrap();
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(c.as_i64().to_vec(), vec![11, 21, 31, 12, 22, 32]);
        assert_eq!(c.dtype(), DType::I64);
        let f = binary("Div", &HostTensor::scalar_f32(1.0), &HostTensor::f32(vec![2], vec![2.0, 4.0])).unwrap();
        assert_eq!(f.as_f32().to_vec(), vec![0.5, 0.25]);
    }

    #[test]
    fn compare_and_reduce() {
        let a = HostTensor::i64(vec![3], vec![1, 5, 3]);
        let c = compare("GreaterOrEqual", &a, &HostTensor::const_i64(3)).unwrap();
        assert_eq!(c.as_bool().to_vec(), vec![false, true, true]);
        let x = HostTensor::f32(vec![2, 2], vec![1., 2., 3., 4.]);
        let m = reduce("ReduceMean", &x, Some(&[-1]), true, false).unwrap();
        assert_eq!(m.shape, vec![2, 1]);
        assert_eq!(m.as_f32().to_vec(), vec![1.5, 3.5]);
        let p = reduce("ReduceProd", &HostTensor::i64(vec![3], vec![2, 3, 4]), None, false, false).unwrap();
        assert_eq!(p.shape, Vec::<usize>::new());
        assert_eq!(p.as_i64()[0], 24);
    }

    #[test]
    fn softmax_rows() {
        let x = HostTensor::f32(vec![1, 3], vec![0., 0., 0.]);
        let s = softmax(&x, 1).unwrap();
        for v in s.as_f32().iter() {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn erf_values() {
        assert!((erf(0.0)).abs() < 1e-6);
        assert!((erf(1.0) - 0.8427008).abs() < 1e-5);
        assert!((erf(-2.0) + 0.9953223).abs() < 1e-5);
    }
}
