//! ONNX LSTM reference, using ONNX's i/o/f/c gate order.
use crate::error::{Error, Result};
use crate::host::eval::need;
use crate::host::tensor::HostTensor;
use crate::ir::Node;

pub struct Config {
    pub sequence: usize,
    pub batch: usize,
    pub input: usize,
    pub hidden: usize,
    pub directions: usize,
    pub layout: usize,
}

impl Config {
    pub fn new(n: &Node, x: &[usize], w: &[usize], r: &[usize]) -> Result<Self> {
        if x.len() != 3 || w.len() != 3 || r.len() != 3 {
            return Err(Error::shape("LSTM input ranks"));
        }
        if n.attrs.contains_key("activations")
            || n.attrs.contains_key("activation_alpha")
            || n.attrs.contains_key("activation_beta")
        {
            return Err(Error::unsupported("LSTM custom activations"));
        }
        let layout = n.attr_i("layout", 0);
        if !matches!(layout, 0 | 1) {
            return Err(Error::shape("LSTM layout"));
        }
        let directions = match n.attr_str("direction").unwrap_or("forward") {
            "forward" | "reverse" => 1,
            "bidirectional" => 2,
            _ => return Err(Error::shape("LSTM direction")),
        };
        if n.attr_f("clip", f32::INFINITY).is_nan()
            || n.attr_f("clip", f32::INFINITY) < 0.
            || !matches!(n.attr_i("input_forget", 0), 0 | 1)
        {
            return Err(Error::model("LSTM clip/input_forget"));
        }
        let hidden = n.attr_i("hidden_size", 0);
        if hidden <= 0 {
            return Err(Error::shape("LSTM hidden_size"));
        }
        let hidden = hidden as usize;
        if w != [directions, 4 * hidden, x[2]] || r != [directions, 4 * hidden, hidden] {
            return Err(Error::shape("LSTM weights"));
        }
        Ok(Self {
            sequence: x[layout as usize],
            batch: x[1 - layout as usize],
            input: x[2],
            hidden,
            directions,
            layout: layout as usize,
        })
    }
    pub fn reverse(&self, n: &Node, d: usize) -> bool {
        d == 1 || n.attr_str("direction") == Some("reverse")
    }
    pub fn lengths(&self, lengths: Option<&HostTensor>) -> Result<Vec<usize>> {
        if let Some(t) = lengths {
            if t.shape != [self.batch] {
                return Err(Error::shape("LSTM sequence_lens shape"));
            }
            t.as_i64()
                .iter()
                .map(|&v| {
                    if v < 0 || v > self.sequence as i64 {
                        Err(Error::shape("LSTM sequence_lens bounds"))
                    } else {
                        Ok(v as usize)
                    }
                })
                .collect()
        } else {
            Ok(vec![self.sequence; self.batch])
        }
    }
}

pub fn eval(n: &Node, ins: &[Option<&HostTensor>]) -> Result<Vec<HostTensor>> {
    let (x, w, r) = (need(n, ins, 0)?, need(n, ins, 1)?, need(n, ins, 2)?);
    let c = Config::new(n, &x.shape, &w.shape, &r.shape)?;
    let get = |i| ins.get(i).copied().flatten();
    let lengths = c.lengths(get(4))?;
    let state_shape = if c.layout == 0 { vec![c.directions, c.batch, c.hidden] } else { vec![c.batch, c.directions, c.hidden] };
    for i in [5, 6] {
        if let Some(t) = get(i) {
            if t.shape != state_shape {
                return Err(Error::shape("LSTM initial state shape"));
            }
        }
    }
    if get(3).is_some_and(|b| b.shape != [c.directions, 8 * c.hidden])
        || get(7).is_some_and(|p| p.shape != [c.directions, 3 * c.hidden])
    {
        return Err(Error::shape("LSTM bias/peepholes"));
    }
    let (x, w, r) = (x.as_f32(), w.as_f32(), r.as_f32());
    let b = get(3).map(|t| t.as_f32());
    let p = get(7).map(|t| t.as_f32());
    let ih = get(5).map(|t| t.as_f32());
    let ic = get(6).map(|t| t.as_f32());
    let (s, batch, h, input, dirs) = (c.sequence, c.batch, c.hidden, c.input, c.directions);
    let mut y = vec![0.; s * dirs * batch * h];
    let mut yh = vec![0.; dirs * batch * h];
    let mut yc = yh.clone();
    let clip = n.attr_f("clip", f32::INFINITY);
    let sigmoid = |v: f32| 1. / (1. + (-v.clamp(-clip, clip)).exp());
    let tanh = |v: f32| v.clamp(-clip, clip).tanh();
    for d in 0..dirs {
        for (bn, &length) in lengths.iter().enumerate() {
            let state_offset = if c.layout == 0 { (d * batch + bn) * h } else { (bn * dirs + d) * h };
            let mut ht = ih.as_ref().map(|v| v[state_offset..state_offset + h].to_vec()).unwrap_or_else(|| vec![0.; h]);
            let mut ct = ic.as_ref().map(|v| v[state_offset..state_offset + h].to_vec()).unwrap_or_else(|| vec![0.; h]);
            for step in 0..length {
                let t = if c.reverse(n, d) { length - 1 - step } else { step };
                let xo = if c.layout == 0 { (t * batch + bn) * input } else { (bn * s + t) * input };
                let mut gates = vec![0.; 4 * h];
                for (g, gate) in gates.iter_mut().enumerate() {
                    let wo = (d * 4 * h + g) * input;
                    let ro = (d * 4 * h + g) * h;
                    *gate = x[xo..xo + input].iter().zip(&w[wo..wo + input]).map(|(a, b)| a * b).sum::<f32>()
                        + ht.iter().zip(&r[ro..ro + h]).map(|(a, b)| a * b).sum::<f32>();
                    if let Some(b) = &b {
                        *gate += b[d * 8 * h + g] + b[d * 8 * h + 4 * h + g];
                    }
                }
                for j in 0..h {
                    let pi = p.as_ref().map_or(0., |p| p[d * 3 * h + j] * ct[j]);
                    let pf = p.as_ref().map_or(0., |p| p[d * 3 * h + 2 * h + j] * ct[j]);
                    let i = sigmoid(gates[j] + pi);
                    let f = if n.attr_i("input_forget", 0) != 0 { 1. - i } else { sigmoid(gates[2 * h + j] + pf) };
                    ct[j] = f * ct[j] + i * tanh(gates[3 * h + j]);
                    let po = p.as_ref().map_or(0., |p| p[d * 3 * h + h + j] * ct[j]);
                    ht[j] = sigmoid(gates[h + j] + po) * ct[j].tanh();
                    let yo =
                        if c.layout == 0 { ((t * dirs + d) * batch + bn) * h + j } else { ((bn * s + t) * dirs + d) * h + j };
                    y[yo] = ht[j];
                }
            }
            yh[state_offset..state_offset + h].copy_from_slice(&ht);
            yc[state_offset..state_offset + h].copy_from_slice(&ct);
        }
    }
    let yshape = if c.layout == 0 { vec![s, dirs, batch, h] } else { vec![batch, s, dirs, h] };
    Ok(vec![HostTensor::f32(yshape, y), HostTensor::f32(state_shape.clone(), yh), HostTensor::f32(state_shape, yc)])
}
