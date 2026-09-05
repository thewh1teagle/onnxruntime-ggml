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
        let t = ggml::mul_mat(ctx, a.t, b.t);
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
        let symmetric = attrs.pad_left == attrs.pad_right;
        if !symmetric && (attrs.pad_left > 0 || attrs.pad_right > 0) {
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
        let padding = if symmetric { attrs.pad_left } else { 0 };
        let l_out = (xin.shape[2] + 2 * padding).saturating_sub(span) / attrs.stride + 1;
        // im2col: ne = [C*K, L_out, N]
        let cols = g::ggml_im2col(
            ctx,
            wk.t,
            xin.t,
            attrs.stride as i32,
            0,
            padding as i32,
            0,
            attrs.dilation as i32,
            0,
            false,
            g::ggml_type_GGML_TYPE_F32,
        );
        let cols2 = g::ggml_reshape_2d(ctx, cols, (c * k) as i64, (l_out * n) as i64);
        let w2 = g::ggml_reshape_2d(ctx, wk.t, (c * k) as i64, m as i64);
        // Keep the weight in src0 so backend matmul kernels reuse its rows.
        let mm = ggml::mul_mat(ctx, w2, cols2); // ne = [M, L_out*N]
        let y = dev(g::ggml_reshape_3d(ctx, mm, m as i64, l_out as i64, n as i64), &[n, l_out, m]);
        let y = ggml::permute(ctx, y, &[0, 2, 1])?;
        add_bias(run, y, bias)
    }
}

fn conv_transpose(run: &mut Run, node: &Node, ins: &[Option<In>]) -> Result<DeviceTensor> {
    let attrs = ConvAttrs::from_node(node)?;
    if attrs.dilation != 1 {
        return Err(Error::unsupported("dilated ConvTranspose"));
    }
    if attrs.group != 1 {
        return conv_transpose_depthwise(run, node, ins, &attrs);
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
        if run.prog.backend.options.conv_transpose_matmul && n == 1 {
            // cols[t, m, k] = sum_c x[c, t] * w[c, m, k]: one matmul. Then
            // out[m, t*s + k] += cols[t, m, k] for each k: K strided accumulates
            // (ggml_acc into a view of the output). ggml's own conv_transpose_1d
            // kernel is a naive loop, ten times slower for mimi's layers.
            let full = (l - 1) * attrs.stride + k + attrs.output_padding;
            let f = std::mem::size_of::<f32>();
            let wt = if node.attr_i("__w_prepacked", 0) != 0 {
                // [M*K, C] constant prepacked at compile: ne=[C, M*K] resident, no per-run copy
                run.dev_f32(need(node, ins, node.inputs.len() - 1)?)?.t
            } else {
                let w2 = g::ggml_reshape_2d(ctx, contig(ctx, wk).t, (m * k) as i64, c as i64); // ne=[M*K, C]
                g::ggml_cont(ctx, g::ggml_transpose(ctx, w2)) // ne=[C, M*K]
            };
            let x2 = g::ggml_reshape_2d(ctx, contig(ctx, xin).t, l as i64, c as i64); // ne=[L, C]
            let xt = g::ggml_cont(ctx, g::ggml_transpose(ctx, x2)); // ne=[C, L]
            let cols = ggml::mul_mat(ctx, wt, xt); // ne=[M*K, L]
            let cols3 = g::ggml_reshape_3d(ctx, cols, k as i64, m as i64, l as i64); // ne=[K, M, L]
            let perm = g::ggml_cont(ctx, g::ggml_permute(ctx, cols3, 2, 0, 1, 3)); // ne=[M, L, K]
            let zero = run.scalar(0.0)?;
            // transposed output ne=[M, full]: positions on ne1, so a stride-s view is expressible
            let mut out_t = g::ggml_repeat_4d(ctx, zero.t, m as i64, full as i64, 1, 1);
            for kk in 0..k {
                let slice = g::ggml_view_2d(ctx, perm, m as i64, l as i64, m * f, kk * m * l * f); // ne=[M, L], contiguous
                out_t = g::ggml_acc(ctx, out_t, slice, attrs.stride * m * f, full * m * f, full * m * f, kk * m * f);
            }
            let out = g::ggml_cont(ctx, g::ggml_transpose(ctx, out_t)); // ne=[full, M] = ONNX [1, M, full]
            tracing::trace!(node = %node, l, c, m, k, stride = attrs.stride, full, "conv_transpose as matmul + acc");
            let mut y = dev(out, &[n, m, full]);
            if attrs.pad_left > 0 || attrs.pad_right > 0 {
                let l_out = full.saturating_sub(attrs.pad_left + attrs.pad_right);
                y = ggml::view_slice(ctx, y, &[0, 0, attrs.pad_left], &[n, m, l_out])?;
                y = contig(ctx, y);
            }
            return add_bias(run, y, bias);
        }
        if n != 1 {
            return Err(Error::unsupported("native ggml ConvTranspose requires batch size one"));
        }
        let t = g::ggml_conv_transpose_1d(ctx, wk.t, xin.t, attrs.stride as i32, 0, 1);
        // ggml's Metal conv_transpose_1d is far slower than its CPU kernel for these
        // sizes (5 ms vs well under 1 ms for mimi's layers); pin the node to the CPU
        // backend and let the scheduler move the small operands.
        if run.prog.backend.gpu && run.prog.backend.options.conv_transpose_cpu {
            g::ggml_backend_sched_set_tensor_backend(run.prog.backend.sched, t, run.prog.backend.cpu());
            tracing::trace!(node = %node, "conv_transpose pinned to the cpu backend");
        }
        let full = (l - 1) * attrs.stride + k + attrs.output_padding;
        let t =
            if attrs.output_padding > 0 { g::ggml_pad_ext(ctx, t, 0, attrs.output_padding as i32, 0, 0, 0, 0, 0, 0) } else { t };
        let mut y = dev(t, &[n, m, full]);
        if attrs.pad_left > 0 || attrs.pad_right > 0 {
            let l_out = full.saturating_sub(attrs.pad_left + attrs.pad_right);
            y = ggml::view_slice(ctx, y, &[0, 0, attrs.pad_left], &[n, m, l_out])?;
            y = contig(ctx, y);
        }
        add_bias(run, y, bias)
    }
}

/// Depthwise ConvTranspose (`group == channels`): `out[c, t*s + k] += x[c, t] * w[c, k]`,
/// which is K broadcast multiplies and K strided accumulates. mimi's final
/// upsampler is one of these; on the host it forced a flush every frame.
fn conv_transpose_depthwise(run: &mut Run, node: &Node, ins: &[Option<In>], attrs: &ConvAttrs) -> Result<DeviceTensor> {
    let x = run.dev_f32(need(node, ins, 0)?)?;
    let w = run.dev_f32(need(node, ins, 1)?)?;
    let bias = match ins.get(2).and_then(|b| b.as_ref()) {
        Some(b) => Some(run.dev_f32(b)?),
        None => None,
    };
    if x.rank != 3 || w.rank != 3 {
        return Err(Error::unsupported("depthwise ConvTranspose other than 1-D"));
    }
    let (n, c, l) = (x.shape[0], x.shape[1], x.shape[2]);
    let (cw, mg, k) = (w.shape[0], w.shape[1], w.shape[2]);
    if n != 1 || cw != c || mg != 1 || attrs.group != c {
        return Err(Error::unsupported(format!("ConvTranspose group {} for channels {c}, weight {:?}", attrs.group, w.shape())));
    }
    let ctx = run.ctx;
    unsafe {
        let f = std::mem::size_of::<f32>();
        let full = (l - 1) * attrs.stride + k + attrs.output_padding;
        let xin = contig(ctx, ggml::reshape(ctx, x, &[c, l])?); // ne=[L, C]
        let wk = contig(ctx, ggml::reshape(ctx, w, &[c, k])?); // ne=[K, C]
                                                               // x transposed to ne=[C, L] so a stride-s view over positions is expressible
        let xt = g::ggml_cont(ctx, g::ggml_transpose(ctx, xin.t)); // ne=[C, L]
        let zero = run.scalar(0.0)?;
        let mut out_t = g::ggml_repeat_4d(ctx, zero.t, c as i64, full as i64, 1, 1); // ne=[C, full]
        for kk in 0..k {
            // w[:, kk] as a column ne=[C, 1] broadcast over L
            let wcol = g::ggml_view_2d(ctx, wk.t, 1, c as i64, k * f, kk * f); // ne=[1, C] strided
            let wcol = g::ggml_cont(ctx, g::ggml_transpose(ctx, wcol)); // ne=[C, 1]
            let term = g::ggml_mul(ctx, xt, wcol); // ne=[C, L]
            out_t = g::ggml_acc(ctx, out_t, term, attrs.stride * c * f, full * c * f, full * c * f, kk * c * f);
        }
        let out = g::ggml_cont(ctx, g::ggml_transpose(ctx, out_t)); // ne=[full, C]
        tracing::trace!(node = %node, c, l, k, stride = attrs.stride, full, "depthwise conv_transpose on device");
        let mut y = dev(out, &[n, c, full]);
        if attrs.pad_left > 0 || attrs.pad_right > 0 {
            let l_out = full.saturating_sub(attrs.pad_left + attrs.pad_right);
            y = ggml::view_slice(ctx, y, &[0, 0, attrs.pad_left], &[n, c, l_out])?;
            y = contig(ctx, y);
        }
        add_bias(run, y, bias)
    }
}
