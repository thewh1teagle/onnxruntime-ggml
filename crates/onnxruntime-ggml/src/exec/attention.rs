//! `FusedAttention(q, k, v[, mask])` on `ggml_flash_attn_ext`.
//!
//! Layouts as the exporters produce them (see `fusion::fuse_attention`):
//! - q `[B, H, T, D]` or `[B*H, T, D]`
//! - k `[B, H, D, S]` or `[B*H, D, S]` (already transposed for `Q·Kᵀ`)
//! - v `[B, H, S, D]` or `[B*H, S, D]`
//! - mask: host bool broadcastable to `[.., T, S]`; true keeps a score
//!
//! ggml wants q `ne=[D, T, H, B]`, k and v `ne=[D, S, H, B]` (the reverse of
//! `[B, H, T, D]`, so q needs nothing, k a swap of its last two axes, v a
//! `cont`), a mask `ne=[S, T]` in f16 with 0 / -inf, and returns
//! `ne=[D, H, T, B]`, that is ONNX `[B, T, H, D]`. The output is presented in
//! the original `[B, H, T, D]` layout so the exporter's trailing Transpose and
//! Reshape keep working unchanged.

use ggml_sys as g;

use crate::error::{Error, Result};
use crate::exec::ggml::{self, contig, dev};
use crate::exec::runtime::{In, Run};
use crate::exec::value::{DeviceTensor, Value};
use crate::host::broadcast::broadcast_index;
use crate::host::tensor::HostTensor;
use crate::ir::{DType, Node};

/// Queries at least this long go to flash attention; shorter ones (decode
/// steps) are faster as three plain kernels, measured on Metal.
pub const FLASH_MIN_T: usize = 32;

fn need<'a>(node: &Node, ins: &'a [Option<In>], i: usize) -> Result<&'a In> {
    ins.get(i).and_then(|x| x.as_ref()).ok_or_else(|| Error::model(format!("{node}: missing input {i}")))
}

pub fn emit(run: &mut Run, node: &Node, ins: &[Option<In>]) -> Result<Vec<Value>> {
    let q = run.dev_f32(need(node, ins, 0)?)?;
    let k = run.dev_f32(need(node, ins, 1)?)?;
    let v = run.dev_f32(need(node, ins, 2)?)?;
    let mask_in = ins.get(3).and_then(|m| m.as_ref());
    let scale = node.attr_f("scale", 1.0);

    let rank = q.rank;
    if rank != 3 && rank != 4 || k.rank != rank || v.rank != rank {
        return Err(Error::unsupported(format!("attention ranks q{} k{} v{}", q.rank, k.rank, v.rank)));
    }
    let (b, h, t, d) =
        if rank == 4 { (q.shape[0], q.shape[1], q.shape[2], q.shape[3]) } else { (1, q.shape[0], q.shape[1], q.shape[2]) };
    let (kb, kh, kd, s) =
        if rank == 4 { (k.shape[0], k.shape[1], k.shape[2], k.shape[3]) } else { (1, k.shape[0], k.shape[1], k.shape[2]) };
    let (vb, vh, vs, vd) =
        if rank == 4 { (v.shape[0], v.shape[1], v.shape[2], v.shape[3]) } else { (1, v.shape[0], v.shape[1], v.shape[2]) };
    if kb != b || vb != b || kh != h || vh != h || kd != d || vd != d || vs != s {
        return Err(Error::unsupported(format!("attention shapes q{:?} k{:?} v{:?}", q.shape(), k.shape(), v.shape())));
    }
    if d % 8 != 0 || t == 0 || s == 0 {
        return Err(Error::unsupported(format!("attention head dim {d}, T {t}, S {s}")));
    }
    let ctx = run.ctx;
    unsafe {
        // A rank-3 [B*H, ..] tensor has the same ne as its rank-4 [1, H, ..] reading:
        // relabel instead of reshaping, which would copy a strided K.
        let relabel = |x: DeviceTensor, shape: [usize; 4]| DeviceTensor { t: x.t, rank: 4, shape: ggml::shape_arr(&shape) };
        // q: [B,H,T,D] -> ne=[D,T,H,B] as is
        let qg = contig(ctx, relabel(q, [b, h, t, d]));
        // k: [B,H,D,S] -> swap D,S -> [B,H,S,D] -> ne=[D,S,H,B] (one copy)
        let kg = contig(ctx, ggml::permute(ctx, relabel(k, [b, h, d, s]), &[0, 1, 3, 2])?);
        let vg = contig(ctx, relabel(v, [b, h, s, d]));
        use crate::exec::backend::Attention;
        let mode = match run.prog.backend.options.attention {
            Attention::Auto if t >= FLASH_MIN_T => Attention::FlashF32,
            Attention::Auto => Attention::Matmul,
            m => m,
        };
        let mask_host = match mask_in {
            Some(m) => Some(build_mask(run.host_param(m, "attention mask")?, b, h, t, s)?),
            None => None,
        };
        let final_ = if mode == Attention::Matmul {
            // scores[T,S] = Q·Kᵀ: a = K rows of length D (ne=[D,S,H,B]), b = Q (ne=[D,T,H,B]) -> ne=[S,T,H,B]
            let scores = g::ggml_mul_mat(ctx, kg.t, qg.t);
            let mask_t = match &mask_host {
                Some(m) => run.upload_raw(DType::F32, &[t, s], m.to_bytes(DType::F32)?, "attn_mask")?.t,
                None => std::ptr::null_mut(),
            };
            let p = g::ggml_soft_max_ext(ctx, scores, mask_t, scale, 0.0);
            // out[T,D] = P·V: a = Vᵀ rows of length S (ne=[S,D,H,B]), b = P (ne=[S,T,H,B]) -> ne=[D,T,H,B] = [B,H,T,D]
            let vt = contig(ctx, ggml::permute(ctx, vg, &[0, 1, 3, 2])?);
            let out = g::ggml_mul_mat(ctx, vt.t, p);
            tracing::trace!(node = %node, b, h, t, s, d, scale, masked = mask_in.is_some(), "attention (matmul path)");
            let res = dev(out, &[b, h, t, d]);
            if rank == 4 {
                res
            } else {
                ggml::reshape(ctx, res, &[h, t, d])?
            }
        } else {
            let f16 = mode == Attention::Flash;
            let kk = if f16 { g::ggml_cast(ctx, kg.t, g::ggml_type_GGML_TYPE_F16) } else { kg.t };
            let vv = if f16 { g::ggml_cast(ctx, vg.t, g::ggml_type_GGML_TYPE_F16) } else { vg.t };
            let mask_t = match &mask_host {
                Some(m) => run.upload_raw(DType::F16, &[t, s], m.to_bytes(DType::F16)?, "attn_mask")?.t,
                None => std::ptr::null_mut(),
            };
            let out = g::ggml_flash_attn_ext(ctx, qg.t, kk, vv, mask_t, scale, 0.0, 0.0);
            g::ggml_flash_attn_ext_set_prec(out, g::ggml_prec_GGML_PREC_F32);
            tracing::trace!(node = %node, b, h, t, s, d, scale, f16, masked = mask_in.is_some(), "flash attention");
            // result ne=[D,H,T,B] = ONNX [B,T,H,D]; present as [B,H,T,D] (a view)
            let res = dev(out, &[b, t, h, d]);
            let as_bhtd = ggml::permute(ctx, res, &[0, 2, 1, 3])?;
            if rank == 4 {
                as_bhtd
            } else {
                ggml::reshape(ctx, as_bhtd, &[h, t, d])?
            }
        };
        ggml::set_name(final_.t, &node.outputs[0]);
        Ok(vec![Value::Device(final_)])
    }
}

/// f16 mask `[T, S]`: 0 where the condition keeps the score, -inf elsewhere.
/// The condition may carry batch/head dims, which must be 1.
fn build_mask(cond: &HostTensor, b: usize, h: usize, t: usize, s: usize) -> Result<HostTensor> {
    let target = vec![b, h, t, s];
    let c = cond.as_bool();
    if crate::host::broadcast::broadcast_shapes(&cond.shape, &target).map(|s| s != target).unwrap_or(true) {
        return Err(Error::unsupported(format!("attention mask shape {:?} for [{b}, {h}, {t}, {s}]", cond.shape)));
    }
    let idx = broadcast_index(&cond.shape, &target);
    // require the mask to be identical across batch and heads
    let plane = t * s;
    let mut out = vec![0f32; plane];
    for (i, &src) in idx.iter().enumerate().take(plane) {
        out[i] = if c[src] { 0.0 } else { f32::NEG_INFINITY };
    }
    for (i, &src) in idx.iter().enumerate().skip(plane) {
        let want = if c[src] { 0.0 } else { f32::NEG_INFINITY };
        if out[i % plane] != want {
            return Err(Error::unsupported("attention mask differs across batch or heads"));
        }
    }
    Ok(HostTensor::f32(vec![t, s], out))
}

#[allow(dead_code)]
fn _unused(_: DeviceTensor) {}
