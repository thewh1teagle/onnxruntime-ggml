//! MatMul, Gemm and the 1-D convolutions.
//!
//! `ggml_mul_mat(a, b)` computes `b · aᵀ` with `a` as `ne = [K, N]` and `b` as
//! `ne = [K, M, ...]`, producing `ne = [N, M, ...]`, which is ONNX `[..., M, N]`.
//! So the weight operand must be laid out `[N, K]` row-major: Gemm's transB=1
//! layout, which `program::pretranspose_weights` also gives MatMul weights.

use ggml_sys as g;

use crate::error::{Error, Result};
use crate::exec::ggml::{self, contig, dev};
use crate::exec::ops_binary::binary;
use crate::exec::runtime::{In, Run};
use crate::exec::value::{DeviceTensor, Value};
use crate::host::broadcast::broadcast_shapes;
use crate::host::eval_nn::ConvAttrs;
use crate::ir::Node;

fn need<'a>(node: &Node, ins: &'a [Option<In>], i: usize) -> Result<&'a In> {
    ins.get(i).and_then(|x| x.as_ref()).ok_or_else(|| Error::model(format!("{node}: missing input {i}")))
}

pub fn emit(run: &mut Run, node: &Node, ins: &[Option<In>]) -> Result<Vec<Value>> {
    let out = match node.op.as_str() {
        "MatMul" => matmul(run, node, ins)?,
        "Gemm" => gemm(run, node, ins)?,
        "Conv" => conv(run, node, ins)?,
        "ConvTranspose" => conv_transpose(run, node, ins)?,
        other => return Err(Error::unsupported(format!("device nn op {other}"))),
    };
    unsafe { ggml::set_name(out.t, &node.outputs[0]) };
    Ok(vec![Value::Device(out)])
}

/// `a` as `[.., N, K]` row-major (ne = [K, N, ..]), `b` as `[.., M, K]`; result `[.., M, N]`.
fn mul_mat(run: &mut Run, a_nk: DeviceTensor, b_mk: DeviceTensor) -> Result<DeviceTensor> {
    let ctx = run.ctx;
    let (ra, rb) = (a_nk.rank, b_mk.rank);
    if ra < 2 || rb < 2 {
        return Err(Error::unsupported("matmul with a 1-D operand"));
    }
    let (n, k) = (a_nk.shape[ra - 2], a_nk.shape[ra - 1]);
    let (m, k2) = (b_mk.shape[rb - 2], b_mk.shape[rb - 1]);
    if k != k2 {
        return Err(Error::shape(format!("matmul inner dims {k} vs {k2}")));
    }
    let batch_a = &a_nk.shape()[..ra - 2];
    let batch_b = &b_mk.shape()[..rb - 2];
    let batch = broadcast_shapes(batch_a, batch_b)?;
    unsafe {
        let mut a = contig(ctx, a_nk);
        let mut b = contig(ctx, b_mk);
        // ggml broadcasts a over b's batch dims; if a's batch is larger, b must be repeated
        if batch_b != batch.as_slice() {
            let mut target = batch.clone();
            target.extend([m, k]);
            b = ggml::repeat_to(ctx, b, &target)?;
        }
        if batch_a != batch.as_slice() && !batch_a.iter().all(|&d| d == 1) {
            let mut target = batch.clone();
            target.extend([n, k]);
            a = ggml::repeat_to(ctx, a, &target)?;
        }
        let t = g::ggml_mul_mat(ctx, a.t, b.t);
        let mut shape = batch;
        shape.extend([m, n]);
        Ok(dev(t, &shape))
    }
}

/// Swap the last two axes (a view).
fn transpose_last2(run: &mut Run, x: DeviceTensor) -> Result<DeviceTensor> {
    let mut perm: Vec<usize> = (0..x.rank).collect();
    let r = x.rank;
    perm.swap(r - 1, r - 2);
    unsafe { ggml::permute(run.ctx, x, &perm) }
}

fn matmul(run: &mut Run, node: &Node, ins: &[Option<In>]) -> Result<DeviceTensor> {
    let a = run.dev_f32(need(node, ins, 0)?)?;
    let b_in = need(node, ins, 1)?;
    let b = run.dev_f32(b_in)?;
    if a.rank < 2 || b.rank < 2 {
        return Err(Error::unsupported("MatMul with a 1-D operand"));
    }
    let b_nk = if node.attr_i("__b_transposed", 0) != 0 { b } else { transpose_last2(run, b)? };
    mul_mat(run, b_nk, a)
}

fn gemm(run: &mut Run, node: &Node, ins: &[Option<In>]) -> Result<DeviceTensor> {
    let mut a = run.dev_f32(need(node, ins, 0)?)?;
    let mut b = run.dev_f32(need(node, ins, 1)?)?;
    if a.rank != 2 || b.rank != 2 {
        return Err(Error::shape("Gemm wants 2-D operands"));
    }
    if node.attr_i("transA", 0) != 0 {
        a = transpose_last2(run, a)?;
    }
    if node.attr_i("transB", 0) == 0 {
        b = transpose_last2(run, b)?;
    }
    let mut y = mul_mat(run, b, a)?;
    let alpha = node.attr_f("alpha", 1.0);
    if (alpha - 1.0).abs() > 1e-7 {
        y = DeviceTensor { t: unsafe { g::ggml_scale(run.ctx, y.t, alpha) }, ..y };
    }
    if let Some(c_in) = ins.get(2).and_then(|c| c.as_ref()) {
        let mut c = run.dev_f32(c_in)?;
        let beta = node.attr_f("beta", 1.0);
        if (beta - 1.0).abs() > 1e-7 {
            c = DeviceTensor { t: unsafe { g::ggml_scale(run.ctx, contig(run.ctx, c).t, beta) }, ..c };
        }
        let out = broadcast_shapes(&y.shape(), &c.shape())?;
        y = binary(run, "Add", y, c, &out)?;
    }
    Ok(y)
}

fn add_bias(run: &mut Run, y: DeviceTensor, bias: Option<DeviceTensor>) -> Result<DeviceTensor> {
    let Some(b) = bias else { return Ok(y) };
    // [N, M, L] + [M] -> bias as [1, M, 1]
    let m = y.shape[1];
    if b.numel() != m {
        return Err(Error::shape(format!("conv bias {} vs channels {m}", b.numel())));
    }
    let b3 = unsafe { ggml::reshape(run.ctx, b, &[1, m, 1])? };
    let shape = y.shape();
    binary(run, "Add", y, b3, &shape)
}

fn conv(run: &mut Run, node: &Node, ins: &[Option<In>]) -> Result<DeviceTensor> {
    let attrs = ConvAttrs::from_node(node)?;
    if attrs.group != 1 {
        return Err(Error::unsupported("grouped Conv"));
    }
    let x = run.dev_f32(need(node, ins, 0)?)?;
    let w = run.dev_f32(need(node, ins, 1)?)?;
    let bias = match ins.get(2).and_then(|b| b.as_ref()) {
        Some(b) => Some(run.dev_f32(b)?),
        None => None,
    };
    if x.rank != 3 || w.rank != 3 {
        return Err(Error::unsupported("Conv other than 1-D"));
    }
    let (n, c, l) = (x.shape[0], x.shape[1], x.shape[2]);
    let (m, cw, k) = (w.shape[0], w.shape[1], w.shape[2]);
    if c != cw {
        return Err(Error::shape(format!("conv channels {c} vs weight {cw}")));
    }
    let ctx = run.ctx;
    unsafe {
        // Restate both operands at their exact 3-D `ne`: a leaf's ggml shape is a
        // *folding* of its ONNX shape, which drops size-1 dims, and im2col reads
        // ne[0..2] as (L, C) on the input and (K, C) on the kernel.
        let mut xin = ggml::reshape(ctx, x, &[n, c, l])?;
        if attrs.pad_left > 0 || attrs.pad_right > 0 {
            let t = g::ggml_pad_ext(ctx, xin.t, attrs.pad_left as i32, attrs.pad_right as i32, 0, 0, 0, 0, 0, 0);
            xin = dev(t, &[n, c, l + attrs.pad_left + attrs.pad_right]);
        }
        let wk = ggml::reshape(ctx, w, &[m, cw, k])?;
        tracing::trace!(
            node = %node,
            input = %ggml::describe(&x),
            padded = %ggml::describe(&xin),
            weight = %ggml::describe(&wk),
            pads = ?[attrs.pad_left, attrs.pad_right],
            "conv im2col"
        );
        let span = attrs.dilation * (k - 1) + 1;
        let l_out = (xin.shape[2]).saturating_sub(span) / attrs.stride + 1;
        // im2col: ne = [C*K, L_out, N]
        let cols = g::ggml_im2col(
            ctx,
            wk.t,
            xin.t,
            attrs.stride as i32,
            0,
            0,
            0,
            attrs.dilation as i32,
            0,
            false,
            g::ggml_type_GGML_TYPE_F32,
        );
        let cols2 = g::ggml_reshape_2d(ctx, cols, (c * k) as i64, (l_out * n) as i64);
        let w2 = g::ggml_reshape_2d(ctx, wk.t, (c * k) as i64, m as i64);
        let mm = g::ggml_mul_mat(ctx, cols2, w2); // ne = [L_out*N, M]
        let y = g::ggml_reshape_3d(ctx, mm, l_out as i64, m as i64, n as i64); // ONNX [N, M, L_out]
        add_bias(run, dev(y, &[n, m, l_out]), bias)
    }
}

fn conv_transpose(run: &mut Run, node: &Node, ins: &[Option<In>]) -> Result<DeviceTensor> {
    let attrs = ConvAttrs::from_node(node)?;
    if attrs.group != 1 {
        return Err(Error::unsupported("grouped ConvTranspose"));
    }
    if attrs.dilation != 1 {
        return Err(Error::unsupported("dilated ConvTranspose"));
    }
    if node.attr_ints("output_padding").is_some_and(|p| p.iter().any(|&v| v != 0)) {
        return Err(Error::unsupported("ConvTranspose output_padding"));
    }
    let x = run.dev_f32(need(node, ins, 0)?)?;
    let w = run.dev_f32(need(node, ins, 1)?)?;
    let bias = match ins.get(2).and_then(|b| b.as_ref()) {
        Some(b) => Some(run.dev_f32(b)?),
        None => None,
    };
    if x.rank != 3 || w.rank != 3 {
        return Err(Error::unsupported("ConvTranspose other than 1-D"));
    }
    let (n, c, l) = (x.shape[0], x.shape[1], x.shape[2]);
    let (cw, m, k) = (w.shape[0], w.shape[1], w.shape[2]);
    if c != cw {
        return Err(Error::shape(format!("conv_transpose channels {c} vs weight {cw}")));
    }
    let ctx = run.ctx;
    unsafe {
        // See `conv`: the leaf `ne` is a folding, ggml_conv_transpose_1d is not.
        let xin = ggml::reshape(ctx, x, &[n, c, l])?;
        let wk = ggml::reshape(ctx, w, &[cw, m, k])?;
        tracing::trace!(
            node = %node,
            input = %ggml::describe(&x),
            packed = %ggml::describe(&xin),
            weight = %ggml::describe(&wk),
            pads = ?[attrs.pad_left, attrs.pad_right],
            "conv_transpose"
        );
        let t = g::ggml_conv_transpose_1d(ctx, wk.t, xin.t, attrs.stride as i32, 0, 1);
        let full = (l - 1) * attrs.stride + k;
        let mut y = dev(t, &[n, m, full]);
        if attrs.pad_left > 0 || attrs.pad_right > 0 {
            let l_out = full.saturating_sub(attrs.pad_left + attrs.pad_right);
            y = ggml::view_slice(ctx, y, &[0, 0, attrs.pad_left], &[n, m, l_out])?;
            y = contig(ctx, y);
        }
        add_bias(run, y, bias)
    }
}
