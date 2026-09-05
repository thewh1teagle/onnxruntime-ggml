//! Lower standard ONNX LSTM to ggml matrix products and gate operations.
use crate::error::{Error, Result};
use crate::exec::ggml::{self, contig, dev};
use crate::exec::ops_binary::binary;
use crate::exec::runtime::{In, Run};
use crate::exec::value::{DeviceTensor, Value};
use crate::host::eval_lstm::Config;
use crate::host::tensor::HostTensor;
use crate::ir::Node;
use ggml_sys as g;

unsafe fn section(run: &Run, x: DeviceTensor, axis: usize, start: usize, count: usize) -> Result<DeviceTensor> {
    let mut starts = vec![0; x.rank];
    starts[axis] = start;
    let mut shape = x.shape();
    shape[axis] = count;
    ggml::view_slice(run.ctx, x, &starts, &shape)
}

// Balanced concatenation keeps sequence assembly O(T log T), rather than
// repeatedly copying the whole prefix of a recurrent output.
unsafe fn concatenate(run: &Run, mut xs: Vec<DeviceTensor>, axis: usize) -> Result<DeviceTensor> {
    while xs.len() > 1 {
        let mut next = Vec::new();
        for pair in xs.chunks(2) {
            if pair.len() == 1 {
                next.push(pair[0]);
                continue;
            }
            let a = contig(run.ctx, pair[0]);
            let b = contig(run.ctx, pair[1]);
            let mut shape = a.shape();
            shape[axis] += b.shape[axis];
            next.push(dev(g::ggml_concat(run.ctx, a.t, b.t, (a.rank - 1 - axis) as i32), &shape));
        }
        xs = next;
    }
    xs.pop().ok_or_else(|| Error::unsupported("empty LSTM sequence on device"))
}

pub fn emit(run: &mut Run, n: &Node, ins: &[Option<In>]) -> Result<Vec<Value>> {
    let get = |i| ins.get(i).and_then(|v: &Option<In>| v.as_ref());
    let need = |i| get(i).ok_or_else(|| Error::model(format!("{n}: missing input {i}")));
    let c = Config::new(n, &need(0)?.v.shape(), &need(1)?.v.shape(), &need(2)?.v.shape())?;
    let lengths = c.lengths(get(4).and_then(|v| v.v.host()))?;
    if c.sequence == 0 || c.batch == 0 {
        return Err(Error::unsupported("empty LSTM device input"));
    }
    // A single recurrent op must fit the graph arena. Larger sequences remain
    // supported by the host reference until chunked device execution is added.
    if c.sequence * c.directions > 1200 {
        return Err(Error::unsupported("LSTM sequence exceeds device graph capacity"));
    }
    tracing::debug!(node = %n, sequence = c.sequence, batch = c.batch, hidden = c.hidden, directions = c.directions, "lowering LSTM to ggml");
    let (s, batch, h, dirs) = (c.sequence, c.batch, c.hidden, c.directions);
    let mut x = run.dev_f32(need(0)?)?;
    let w = run.dev_f32(need(1)?)?;
    let r = run.dev_f32(need(2)?)?;
    let bias = get(3).map(|v| run.dev_f32(v)).transpose()?;
    let ih = get(5).map(|v| run.dev_f32(v)).transpose()?;
    let ic = get(6).map(|v| run.dev_f32(v)).transpose()?;
    let peepholes = get(7).map(|v| run.dev_f32(v)).transpose()?;
    let state_shape = if c.layout == 0 { vec![dirs, batch, h] } else { vec![batch, dirs, h] };
    for v in [ih, ic].into_iter().flatten() {
        if v.shape() != state_shape {
            return Err(Error::shape("LSTM state shape"));
        }
    }
    if bias.is_some_and(|b| b.shape() != [dirs, 8 * h]) || peepholes.is_some_and(|p| p.shape() != [dirs, 3 * h]) {
        return Err(Error::shape("LSTM bias/peepholes"));
    }
    let ctx = run.ctx;
    let clip = n.attr_f("clip", f32::INFINITY);
    let mut ys = Vec::new();
    let mut hs = Vec::new();
    let mut cs = Vec::new();
    unsafe {
        if c.layout == 1 {
            x = ggml::permute(ctx, x, &[1, 0, 2])?;
        }
        let flat_x = ggml::reshape(ctx, x, &[s * batch, c.input])?;
        for d in 0..dirs {
            let wd = ggml::reshape(ctx, section(run, w, 0, d, 1)?, &[4 * h, c.input])?;
            let rd = ggml::reshape(ctx, section(run, r, 0, d, 1)?, &[4 * h, h])?;
            let mut proj = dev(ggml::mul_mat(ctx, wd.t, flat_x.t), &[s * batch, 4 * h]);
            if let Some(b) = bias {
                let bd = ggml::reshape(ctx, section(run, b, 0, d, 1)?, &[8 * h])?;
                let bw = section(run, bd, 0, 0, 4 * h)?;
                let br = section(run, bd, 0, 4 * h, 4 * h)?;
                let bsum = binary(run, "Add", bw, br, &[4 * h])?;
                proj = binary(run, "Add", proj, bsum, &proj.shape())?;
            }
            proj = ggml::reshape(ctx, proj, &[s, batch, 4 * h])?;
            let mut initial = |state: Option<DeviceTensor>| -> Result<DeviceTensor> {
                match state {
                    Some(v) => {
                        let v = if c.layout == 1 { ggml::permute(ctx, v, &[1, 0, 2])? } else { v };
                        ggml::reshape(ctx, section(run, v, 0, d, 1)?, &[batch, h])
                    }
                    None => run.upload(&HostTensor::f32(vec![batch, h], vec![0.; batch * h]), "lstm_zero"),
                }
            };
            let mut ht = initial(ih)?;
            let mut ct = initial(ic)?;
            let p = peepholes.map(|p| section(run, p, 0, d, 1).and_then(|p| ggml::reshape(ctx, p, &[3 * h]))).transpose()?;
            let mut steps = Vec::new();
            for step in 0..s {
                let t = if c.reverse(n, d) { s - 1 - step } else { step };
                let xp = ggml::reshape(ctx, section(run, proj, 0, t, 1)?, &[batch, 4 * h])?;
                let recurrence = dev(ggml::mul_mat(ctx, rd.t, contig(ctx, ht).t), &[batch, 4 * h]);
                let gates = binary(run, "Add", xp, recurrence, &[batch, 4 * h])?;
                let mut gate = |gate_id, cell: DeviceTensor, sigmoid: bool| -> Result<DeviceTensor> {
                    let mut v = section(run, gates, 1, gate_id * h, h)?;
                    if let Some(p) = p {
                        if gate_id < 3 {
                            let p = section(run, p, 0, gate_id * h, h)?;
                            let cp = binary(run, "Mul", cell, p, &[batch, h])?;
                            v = binary(run, "Add", v, cp, &[batch, h])?;
                        }
                    }
                    v = contig(ctx, v);
                    if clip.is_finite() {
                        v.t = g::ggml_clamp(ctx, v.t, -clip, clip);
                    }
                    v.t = if sigmoid { g::ggml_sigmoid(ctx, v.t) } else { g::ggml_tanh(ctx, v.t) };
                    Ok(v)
                };
                let i = gate(0, ct, true)?;
                let f = if n.attr_i("input_forget", 0) != 0 {
                    DeviceTensor { t: g::ggml_scale_bias(ctx, i.t, -1., 1.), ..i }
                } else {
                    gate(2, ct, true)?
                };
                let candidate = gate(3, ct, false)?;
                let fc = binary(run, "Mul", f, ct, &[batch, h])?;
                let ic = binary(run, "Mul", i, candidate, &[batch, h])?;
                let mut next_c = binary(run, "Add", fc, ic, &[batch, h])?;
                // Output gate peepholes read the new cell state.
                let mut o = section(run, gates, 1, h, h)?;
                if let Some(p) = p {
                    let po = section(run, p, 0, h, h)?;
                    let cp = binary(run, "Mul", next_c, po, &[batch, h])?;
                    o = binary(run, "Add", o, cp, &[batch, h])?;
                }
                let mut o = contig(ctx, o);
                if clip.is_finite() {
                    o.t = g::ggml_clamp(ctx, o.t, -clip, clip);
                }
                o.t = g::ggml_sigmoid(ctx, o.t);
                let tc = DeviceTensor { t: g::ggml_tanh(ctx, next_c.t), ..next_c };
                let mut next_h = binary(run, "Mul", o, tc, &[batch, h])?;
                let mut y = next_h;
                if lengths.iter().any(|&l| t >= l) {
                    // Select rows rather than multiplying by a zero mask: padded
                    // input values may be NaN, which must not enter recurrent state.
                    let zero = run.upload(&HostTensor::f32(vec![batch, h], vec![0.; batch * h]), "lstm_padding")?;
                    let ids = HostTensor::i64(
                        vec![batch],
                        lengths.iter().enumerate().map(|(b, &l)| (b + if t < l { 0 } else { batch }) as i64).collect(),
                    );
                    let stack_y = concatenate(run, vec![next_h, zero], 0)?;
                    y = crate::exec::ops_shape::gather(run, stack_y, &ids, 0)?;
                    let stack_h = concatenate(run, vec![next_h, ht], 0)?;
                    next_h = crate::exec::ops_shape::gather(run, stack_h, &ids, 0)?;
                    let stack_c = concatenate(run, vec![next_c, ct], 0)?;
                    next_c = crate::exec::ops_shape::gather(run, stack_c, &ids, 0)?;
                }
                ht = next_h;
                ct = next_c;
                steps.push(ggml::reshape(ctx, y, &[1, batch, h])?);
            }
            if c.reverse(n, d) {
                steps.reverse();
            }
            ys.push(ggml::reshape(ctx, concatenate(run, steps, 0)?, &[s, 1, batch, h])?);
            hs.push(ggml::reshape(ctx, ht, &[1, batch, h])?);
            cs.push(ggml::reshape(ctx, ct, &[1, batch, h])?);
        }
        let mut y = concatenate(run, ys, 1)?;
        let mut h = concatenate(run, hs, 0)?;
        let mut c_out = concatenate(run, cs, 0)?;
        if c.layout == 1 {
            y = ggml::permute(ctx, y, &[2, 0, 1, 3])?;
            h = ggml::permute(ctx, h, &[1, 0, 2])?;
            c_out = ggml::permute(ctx, c_out, &[1, 0, 2])?;
        }
        Ok(vec![Value::Device(y), Value::Device(h), Value::Device(c_out)])
    }
}
