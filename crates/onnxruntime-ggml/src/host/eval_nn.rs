//! The neural-network ops on host tensors: matmul, gemm, layer norm and 1-D
//! convolutions. Naive loops; these run for constant folding and for the odd
//! op ggml cannot express (grouped transposed convolution).

use crate::error::{Error, Result};
use crate::host::eval::{need, norm_axis};
use crate::host::eval_shape::transpose;
use crate::host::tensor::{numel_of, HostTensor};
use crate::ir::Node;

pub fn eval(node: &Node, inputs: &[Option<&HostTensor>]) -> Result<Vec<HostTensor>> {
    let out = match node.op.as_str() {
        "MatMul" => {
            let a = need(node, inputs, 0)?;
            let b = need(node, inputs, 1)?;
            if node.attr_i("__b_transposed", 0) != 0 {
                let bt = transpose(b, &[1, 0])?;
                matmul(a, &bt)?
            } else {
                matmul(a, b)?
            }
        }
        "Gemm" => {
            let a = need(node, inputs, 0)?;
            let b = need(node, inputs, 1)?;
            let a = if node.attr_i("transA", 0) != 0 { transpose(a, &[1, 0])? } else { a.clone() };
            let b = if node.attr_i("transB", 0) != 0 { transpose(b, &[1, 0])? } else { b.clone() };
            let mut y = matmul(&a, &b)?;
            let alpha = node.attr_f("alpha", 1.0);
            let beta = node.attr_f("beta", 1.0);
            if alpha != 1.0 {
                y = HostTensor::f32(y.shape.clone(), y.as_f32().iter().map(|v| v * alpha).collect());
            }
            if let Some(c) = inputs.get(2).copied().flatten() {
                let c = if beta != 1.0 {
                    HostTensor::f32(c.shape.clone(), c.as_f32().iter().map(|v| v * beta).collect())
                } else {
                    c.clone()
                };
                y = crate::host::eval_math::binary("Add", &y, &c)?;
            }
            y
        }
        "LayerNormalization" => {
            let x = need(node, inputs, 0)?;
            let scale = need(node, inputs, 1)?;
            let bias = inputs.get(2).copied().flatten();
            let axis = norm_axis(node.attr_i("axis", -1), x.rank())?;
            let eps = node.attr_f("epsilon", 1e-5);
            layer_norm(x, scale, bias, axis, eps)?
        }
        "Conv" => {
            let x = need(node, inputs, 0)?;
            let w = need(node, inputs, 1)?;
            let b = inputs.get(2).copied().flatten();
            let attrs = ConvAttrs::from_node(node)?;
            conv1d(x, w, b, &attrs)?
        }
        "ConvTranspose" => {
            let x = need(node, inputs, 0)?;
            let w = need(node, inputs, 1)?;
            let b = inputs.get(2).copied().flatten();
            let attrs = ConvAttrs::from_node(node)?;
            conv_transpose1d(x, w, b, &attrs)?
        }
        other => return Err(Error::unsupported(format!("host nn op {other}"))),
    };
    Ok(vec![out])
}

/// Batched matmul with numpy broadcasting on the leading dims.
pub fn matmul(a: &HostTensor, b: &HostTensor) -> Result<HostTensor> {
    let a2 = if a.rank() == 1 { a.reshaped(vec![1, a.shape[0]])? } else { a.clone() };
    let b2 = if b.rank() == 1 { b.reshaped(vec![b.shape[0], 1])? } else { b.clone() };
    let (m, k) = (a2.shape[a2.rank() - 2], a2.shape[a2.rank() - 1]);
    let (k2, n) = (b2.shape[b2.rank() - 2], b2.shape[b2.rank() - 1]);
    if k != k2 {
        return Err(Error::shape(format!("matmul {:?} x {:?}", a.shape, b.shape)));
    }
    let batch_a = &a2.shape[..a2.rank() - 2];
    let batch_b = &b2.shape[..b2.rank() - 2];
    let batch = crate::host::broadcast::broadcast_shapes(batch_a, batch_b)?;
    let nb = numel_of(&batch);
    let ai = crate::host::broadcast::broadcast_index(batch_a, &batch);
    let bi = crate::host::broadcast::broadcast_index(batch_b, &batch);
    let x = a2.as_f32();
    let y = b2.as_f32();
    let mut out = vec![0f32; nb * m * n];
    for bidx in 0..nb {
        let xa = &x[ai[bidx] * m * k..ai[bidx] * m * k + m * k];
        let yb = &y[bi[bidx] * k * n..bi[bidx] * k * n + k * n];
        let o = &mut out[bidx * m * n..(bidx + 1) * m * n];
        for i in 0..m {
            for p in 0..k {
                let av = xa[i * k + p];
                if av == 0.0 {
                    continue;
                }
                let row = &yb[p * n..p * n + n];
                let orow = &mut o[i * n..i * n + n];
                for j in 0..n {
                    orow[j] += av * row[j];
                }
            }
        }
    }
    let mut shape = batch;
    if a.rank() != 1 {
        shape.push(m);
    }
    if b.rank() != 1 {
        shape.push(n);
    }
    Ok(HostTensor::f32(shape, out))
}

pub fn layer_norm(x: &HostTensor, scale: &HostTensor, bias: Option<&HostTensor>, axis: usize, eps: f32) -> Result<HostTensor> {
    let v = x.as_f32();
    let inner: usize = x.shape[axis..].iter().product();
    let outer = x.numel() / inner.max(1);
    let s = scale.as_f32();
    let b = bias.map(|b| b.as_f32());
    if s.len() != inner {
        return Err(Error::shape(format!("layernorm scale {:?} vs normalized size {inner}", scale.shape)));
    }
    let mut out = vec![0f32; v.len()];
    for o in 0..outer {
        let row = &v[o * inner..(o + 1) * inner];
        let mean = row.iter().sum::<f32>() / inner as f32;
        let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / inner as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..inner {
            let mut y = (row[i] - mean) * inv * s[i];
            if let Some(b) = &b {
                y += b[i];
            }
            out[o * inner + i] = y;
        }
    }
    Ok(HostTensor::f32(x.shape.clone(), out))
}

#[derive(Clone, Debug)]
pub struct ConvAttrs {
    pub stride: usize,
    pub pad_left: usize,
    pub pad_right: usize,
    pub dilation: usize,
    pub group: usize,
}

impl ConvAttrs {
    pub fn from_node(node: &Node) -> Result<ConvAttrs> {
        let strides = node.attr_ints("strides").unwrap_or_else(|| vec![1]);
        let pads = node.attr_ints("pads").unwrap_or_else(|| vec![0, 0]);
        let dil = node.attr_ints("dilations").unwrap_or_else(|| vec![1]);
        if strides.len() != 1 || dil.len() != 1 || pads.len() != 2 {
            return Err(Error::unsupported(format!("{node}: only 1-D convolutions (strides {strides:?}, pads {pads:?})")));
        }
        if node.attr_str("auto_pad").is_some_and(|p| p != "NOTSET") {
            return Err(Error::unsupported(format!("{node}: auto_pad")));
        }
        Ok(ConvAttrs {
            stride: strides[0] as usize,
            pad_left: pads[0] as usize,
            pad_right: pads[1] as usize,
            dilation: dil[0] as usize,
            group: node.attr_i("group", 1) as usize,
        })
    }
}

/// x [N, C, L], w [M, C/g, K] -> [N, M, L_out]
pub fn conv1d(x: &HostTensor, w: &HostTensor, bias: Option<&HostTensor>, a: &ConvAttrs) -> Result<HostTensor> {
    if x.rank() != 3 || w.rank() != 3 {
        return Err(Error::shape(format!("conv1d wants 3-D tensors, got {:?} and {:?}", x.shape, w.shape)));
    }
    let (n, c, l) = (x.shape[0], x.shape[1], x.shape[2]);
    let (m, cg, k) = (w.shape[0], w.shape[1], w.shape[2]);
    if c != cg * a.group || m % a.group != 0 {
        return Err(Error::shape(format!("conv1d channels {c} vs weight {:?} group {}", w.shape, a.group)));
    }
    let span = a.dilation * (k - 1) + 1;
    let l_out = (l + a.pad_left + a.pad_right).saturating_sub(span) / a.stride + 1;
    let xv = x.as_f32();
    let wv = w.as_f32();
    let bv = bias.map(|b| b.as_f32());
    let mg = m / a.group;
    let mut out = vec![0f32; n * m * l_out];
    for bi in 0..n {
        for g in 0..a.group {
            for om in 0..mg {
                let oc = g * mg + om;
                for t in 0..l_out {
                    let mut acc = bv.as_ref().map(|b| b[oc]).unwrap_or(0.0);
                    for ic in 0..cg {
                        let xc = g * cg + ic;
                        for kk in 0..k {
                            let pos = t * a.stride + kk * a.dilation;
                            if pos < a.pad_left || pos - a.pad_left >= l {
                                continue;
                            }
                            acc += xv[(bi * c + xc) * l + pos - a.pad_left] * wv[(oc * cg + ic) * k + kk];
                        }
                    }
                    out[(bi * m + oc) * l_out + t] = acc;
                }
            }
        }
    }
    Ok(HostTensor::f32(vec![n, m, l_out], out))
}

/// x [N, C, L], w [C, M/g, K] -> [N, M, L_out], L_out = (L-1)*s + d*(K-1) + 1 - pads
pub fn conv_transpose1d(x: &HostTensor, w: &HostTensor, bias: Option<&HostTensor>, a: &ConvAttrs) -> Result<HostTensor> {
    if x.rank() != 3 || w.rank() != 3 {
        return Err(Error::shape(format!("conv_transpose1d wants 3-D tensors, got {:?} and {:?}", x.shape, w.shape)));
    }
    let (n, c, l) = (x.shape[0], x.shape[1], x.shape[2]);
    let (cw, mg, k) = (w.shape[0], w.shape[1], w.shape[2]);
    if cw != c || c % a.group != 0 {
        return Err(Error::shape(format!("conv_transpose1d channels {c} vs weight {:?} group {}", w.shape, a.group)));
    }
    let m = mg * a.group;
    let cg = c / a.group;
    let full = (l - 1) * a.stride + a.dilation * (k - 1) + 1;
    let l_out = full.saturating_sub(a.pad_left + a.pad_right);
    let xv = x.as_f32();
    let wv = w.as_f32();
    let bv = bias.map(|b| b.as_f32());
    let mut out = vec![0f32; n * m * l_out];
    for bi in 0..n {
        for g in 0..a.group {
            for ic in 0..cg {
                let xc = g * cg + ic;
                for om in 0..mg {
                    let oc = g * mg + om;
                    for t in 0..l {
                        let xval = xv[(bi * c + xc) * l + t];
                        if xval == 0.0 {
                            continue;
                        }
                        for kk in 0..k {
                            let pos = t * a.stride + kk * a.dilation;
                            if pos < a.pad_left || pos - a.pad_left >= l_out {
                                continue;
                            }
                            out[(bi * m + oc) * l_out + pos - a.pad_left] += xval * wv[(xc * mg + om) * k + kk];
                        }
                    }
                }
            }
        }
        if let Some(b) = &bv {
            for oc in 0..m {
                for t in 0..l_out {
                    out[(bi * m + oc) * l_out + t] += b[oc];
                }
            }
        }
    }
    Ok(HostTensor::f32(vec![n, m, l_out], out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_batched() {
        let a = HostTensor::f32(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = HostTensor::f32(vec![2, 2], vec![5., 6., 7., 8.]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.as_f32().to_vec(), vec![19., 22., 43., 50.]);
        let a3 = HostTensor::f32(vec![2, 1, 2], vec![1., 0., 0., 1.]);
        let c = matmul(&a3, &b).unwrap();
        assert_eq!(c.shape, vec![2, 1, 2]);
        assert_eq!(c.as_f32().to_vec(), vec![5., 6., 7., 8.]);
    }

    #[test]
    fn layer_norm_rows() {
        let x = HostTensor::f32(vec![1, 4], vec![1., 2., 3., 4.]);
        let s = HostTensor::f32(vec![4], vec![1.; 4]);
        let y = layer_norm(&x, &s, None, 1, 1e-5).unwrap();
        let v = y.as_f32();
        assert!((v[0] + 1.3416).abs() < 1e-3 && (v[3] - 1.3416).abs() < 1e-3);
    }

    #[test]
    fn conv_and_transpose() {
        let x = HostTensor::f32(vec![1, 1, 4], vec![1., 2., 3., 4.]);
        let w = HostTensor::f32(vec![1, 1, 2], vec![1., 1.]);
        let a = ConvAttrs { stride: 1, pad_left: 0, pad_right: 0, dilation: 1, group: 1 };
        let y = conv1d(&x, &w, None, &a).unwrap();
        assert_eq!(y.as_f32().to_vec(), vec![3., 5., 7.]);
        let wt = HostTensor::f32(vec![1, 1, 2], vec![1., 1.]);
        let a2 = ConvAttrs { stride: 2, pad_left: 0, pad_right: 0, dilation: 1, group: 1 };
        let y = conv_transpose1d(&HostTensor::f32(vec![1, 1, 2], vec![1., 2.]), &wt, None, &a2).unwrap();
        assert_eq!(y.as_f32().to_vec(), vec![1., 1., 2., 2.]);
        // depthwise transposed conv: two channels, group 2
        let x2 = HostTensor::f32(vec![1, 2, 1], vec![1., 10.]);
        let w2 = HostTensor::f32(vec![2, 1, 2], vec![1., 2., 3., 4.]);
        let a3 = ConvAttrs { stride: 2, pad_left: 0, pad_right: 0, dilation: 1, group: 2 };
        let y = conv_transpose1d(&x2, &w2, None, &a3).unwrap();
        assert_eq!(y.shape, vec![1, 2, 2]);
        assert_eq!(y.as_f32().to_vec(), vec![1., 2., 30., 40.]);
    }
}
