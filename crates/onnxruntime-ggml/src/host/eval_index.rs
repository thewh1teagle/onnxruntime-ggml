//! Indexing operators used by packed recurrent sequences and resampling.
use crate::error::{Error, Result};
use crate::host::eval::{need, norm_axis};
use crate::host::tensor::{strides_of, HostTensor};
use crate::ir::Node;

pub const OPS: &[&str] = &["GatherElements", "ScatterElements", "ScatterND", "TopK", "Flatten", "CumSum", "Pad"];

fn index(v: i64, size: usize) -> Result<usize> {
    let v = if v < 0 { v + size as i64 } else { v };
    if v < 0 || v >= size as i64 {
        return Err(Error::shape("index out of bounds"));
    }
    Ok(v as usize)
}

pub fn eval(n: &Node, ins: &[Option<&HostTensor>]) -> Result<Vec<HostTensor>> {
    let x = need(n, ins, 0)?;
    let out = match n.op.as_str() {
        "Flatten" => {
            let a = n.attr_i("axis", 1);
            let a = if a < 0 { a + x.rank() as i64 } else { a };
            if a < 0 || a > x.rank() as i64 {
                return Err(Error::shape("Flatten axis"));
            }
            x.reshaped(vec![x.shape[..a as usize].iter().product(), x.shape[a as usize..].iter().product()])?
        }
        "GatherElements" | "ScatterElements" => {
            let ids = need(n, ins, 1)?;
            let axis = norm_axis(n.attr_i("axis", 0), x.rank())?;
            if ids.rank() != x.rank() {
                return Err(Error::shape("index rank"));
            }
            let st = strides_of(&x.shape);
            let ist = strides_of(&ids.shape);
            let ix = ids.as_i64();
            let mut offsets = Vec::with_capacity(ids.numel());
            for (i, &id) in ix.iter().enumerate() {
                let mut offset = 0;
                for a in 0..x.rank() {
                    let c = if a == axis { index(id, x.shape[a])? } else { i / ist[a] % ids.shape[a] };
                    if c >= x.shape[a] {
                        return Err(Error::shape("index shape exceeds data"));
                    }
                    offset += c * st[a];
                }
                offsets.push(offset);
            }
            if n.op == "GatherElements" {
                x.gather_flat(ids.shape.clone(), &offsets)
            } else {
                scatter(n, x, need(n, ins, 2)?, &offsets)?
            }
        }
        "ScatterND" => {
            let ids = need(n, ins, 1)?;
            let k = *ids.shape.last().ok_or_else(|| Error::shape("ScatterND indices rank"))?;
            if k == 0 || k > x.rank() {
                return Err(Error::shape("ScatterND tuple rank"));
            }
            let size: usize = x.shape[k..].iter().product();
            let st = strides_of(&x.shape);
            let ix = ids.as_i64();
            let mut offsets = Vec::new();
            for row in ix.chunks_exact(k) {
                let mut base = 0;
                for a in 0..k {
                    base += index(row[a], x.shape[a])? * st[a];
                }
                offsets.extend(base..base + size);
            }
            scatter(n, x, need(n, ins, 2)?, &offsets)?
        }
        "TopK" => {
            let k = need(n, ins, 1)?.scalar_i64()?;
            let axis = norm_axis(n.attr_i("axis", -1), x.rank())?;
            let dim = x.shape[axis];
            if k < 0 || k as usize > dim {
                return Err(Error::shape("TopK k"));
            }
            let k = k as usize;
            let outer: usize = x.shape[..axis].iter().product();
            let inner: usize = x.shape[axis + 1..].iter().product();
            let values = x.as_f64();
            let integers = x.dtype().is_int().then(|| x.as_i64());
            let mut gather = vec![0; outer * k * inner];
            let mut indices = vec![0; gather.len()];
            for o in 0..outer {
                for i in 0..inner {
                    let mut order: Vec<usize> = (0..dim).collect();
                    order.sort_by(|&a, &b| {
                        let (ai, bi) = ((o * dim + a) * inner + i, (o * dim + b) * inner + i);
                        let ord = if let Some(v) = &integers {
                            v[ai].cmp(&v[bi])
                        } else {
                            values[ai].partial_cmp(&values[bi]).unwrap_or(std::cmp::Ordering::Equal)
                        };
                        (if n.attr_i("largest", 1) != 0 { ord.reverse() } else { ord }).then(a.cmp(&b))
                    });
                    for (j, &selected) in order.iter().take(k).enumerate() {
                        let dst = (o * k + j) * inner + i;
                        gather[dst] = (o * dim + selected) * inner + i;
                        indices[dst] = selected as i64;
                    }
                }
            }
            let mut shape = x.shape.clone();
            shape[axis] = k;
            return Ok(vec![x.gather_flat(shape.clone(), &gather), HostTensor::i64(shape, indices)]);
        }
        "CumSum" => {
            let axis = norm_axis(need(n, ins, 1)?.scalar_i64()?, x.rank())?;
            let outer: usize = x.shape[..axis].iter().product();
            let inner: usize = x.shape[axis + 1..].iter().product();
            let dim = x.shape[axis];
            macro_rules! scan {
                ($input:expr, $variant:ident, $zero:expr, $add:expr) => {{
                    let input = $input;
                    let mut output = vec![$zero; input.len()];
                    for o in 0..outer {
                        for i in 0..inner {
                            let mut sum = $zero;
                            for t in 0..dim {
                                let a = if n.attr_i("reverse", 0) != 0 { dim - 1 - t } else { t };
                                let j = (o * dim + a) * inner + i;
                                if n.attr_i("exclusive", 0) != 0 {
                                    output[j] = sum;
                                    sum = $add(sum, input[j]);
                                } else {
                                    sum = $add(sum, input[j]);
                                    output[j] = sum;
                                }
                            }
                        }
                    }
                    super::tensor::Data::$variant(output)
                }};
            }
            use super::tensor::Data;
            let data = match &x.data {
                Data::I64(v) => scan!(v, I64, 0i64, i64::wrapping_add),
                Data::I32(v) => scan!(v, I32, 0i32, i32::wrapping_add),
                Data::F32(v) => scan!(v, F32, 0f32, |a, b| a + b),
                Data::F64(v) => scan!(v, F64, 0f64, |a, b| a + b),
                _ => return Err(Error::unsupported("CumSum element type")),
            };
            HostTensor::new(x.shape.clone(), data)?
        }
        "Pad" => pad(n, ins)?,
        _ => return Err(Error::unsupported(&n.op)),
    };
    Ok(vec![out])
}

fn scatter(n: &Node, x: &HostTensor, updates: &HostTensor, offsets: &[usize]) -> Result<HostTensor> {
    if offsets.len() != updates.numel() || x.dtype() != updates.dtype() {
        return Err(Error::shape("scatter updates"));
    }
    let reduction = n.attr_str("reduction").unwrap_or("none");
    if reduction != "none" {
        return Err(Error::unsupported("scatter reduction"));
    }
    // Byte copies preserve integer values, NaNs, and signed zero exactly.
    let mut data = x.to_bytes(x.dtype())?;
    let src = updates.to_bytes(x.dtype())?;
    let size = x.dtype().size();
    for (i, &dst) in offsets.iter().enumerate() {
        data[dst * size..(dst + 1) * size].copy_from_slice(&src[i * size..(i + 1) * size]);
    }
    HostTensor::from_bytes(x.dtype(), x.shape.clone(), &data)
}

pub fn pad(n: &Node, ins: &[Option<&HostTensor>]) -> Result<HostTensor> {
    let x = need(n, ins, 0)?;
    if ins.get(3).and_then(|v| *v).is_some() {
        return Err(Error::unsupported("Pad axes input"));
    }
    let pads = need(n, ins, 1)?.as_i64();
    if pads.len() != x.rank() * 2 {
        return Err(Error::shape("Pad pads rank"));
    }
    let mut shape = Vec::new();
    for a in 0..x.rank() {
        let size = x.shape[a] as i64 + pads[a] + pads[a + x.rank()];
        if size < 0 {
            return Err(Error::shape("Pad negative dimension"));
        }
        shape.push(size as usize);
    }
    let mode = n.attr_str("mode").unwrap_or("constant");
    let st = strides_of(&x.shape);
    let dstst = strides_of(&shape);
    let scalar = ins.get(2).and_then(|v| *v).cloned().unwrap_or_else(|| HostTensor::zeros(x.dtype(), vec![]));
    let fill = scalar.to_bytes(x.dtype())?;
    let src = x.to_bytes(x.dtype())?;
    let size = x.dtype().size();
    let count: usize = shape.iter().product();
    let mut data = Vec::with_capacity(count * size);
    for i in 0..count {
        let mut off = 0;
        let mut outside = false;
        for a in 0..x.rank() {
            let mut c = (i / dstst[a] % shape[a]) as i64 - pads[a];
            let dim = x.shape[a] as i64;
            if c < 0 || c >= dim {
                match mode {
                    "constant" => {
                        outside = true;
                        break;
                    }
                    "edge" if dim > 0 => c = c.clamp(0, dim - 1),
                    "reflect" if dim > 1 => {
                        c = c.rem_euclid(2 * (dim - 1));
                        if c >= dim {
                            c = 2 * (dim - 1) - c;
                        }
                    }
                    _ => return Err(Error::unsupported("Pad mode or empty dimension")),
                }
            }
            off += c as usize * st[a];
        }
        data.extend_from_slice(if outside { &fill[..size] } else { &src[off * size..(off + 1) * size] });
    }
    HostTensor::from_bytes(x.dtype(), shape, &data)
}
