//! Per-node random streams. ONNX fixes the distribution, not the RNG algorithm.
use crate::error::{Error, Result};
use crate::host::eval::need;
use crate::host::tensor::{Data, HostTensor};
use crate::ir::{DType, Node};
use std::collections::HashMap;

pub const OPS: &[&str] = &["RandomNormalLike", "RandomUniformLike"];

#[derive(Default)]
pub struct Streams(HashMap<String, u64>);

impl Streams {
    pub fn eval(&mut self, n: &Node, ins: &[Option<&HostTensor>]) -> Result<Vec<HostTensor>> {
        let x = need(n, ins, 0)?;
        let dtype = if n.attrs.contains_key("dtype") { DType::from_onnx(n.attr_i("dtype", 1) as i32)? } else { x.dtype() };
        if !dtype.is_float() {
            return Err(Error::unsupported("random output dtype"));
        }
        let state = self.0.entry(format!("{}:{}", n.name, n.outputs.join(","))).or_insert_with(|| {
            if n.attrs.contains_key("seed") {
                n.attr_f("seed", 0.).to_bits() as u64
            } else {
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as u64
            }
        });
        let uniform = |state: &mut u64| -> f64 {
            // SplitMix64; take 53 bits so the result is exactly in [0, 1).
            *state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(x.numel());
        match n.op.as_str() {
            "RandomUniformLike" => {
                let (low, high) = (n.attr_f("low", 0.) as f64, n.attr_f("high", 1.) as f64);
                if !low.is_finite() || !high.is_finite() || low >= high {
                    return Err(Error::model("random uniform bounds"));
                }
                for _ in 0..x.numel() {
                    let value = low + (high - low) * uniform(state);
                    out.push(if dtype == DType::F32 { (value as f32).min((high as f32).next_down()) as f64 } else { value });
                }
            }
            "RandomNormalLike" => {
                let (mean, scale) = (n.attr_f("mean", 0.) as f64, n.attr_f("scale", 1.) as f64);
                if !mean.is_finite() || !scale.is_finite() || scale < 0. {
                    return Err(Error::model("random normal parameters"));
                }
                while out.len() < x.numel() {
                    let radius = (-2. * (1. - uniform(state)).ln()).sqrt();
                    let theta = 2. * std::f64::consts::PI * uniform(state);
                    out.push(mean + scale * radius * theta.cos());
                    if out.len() < x.numel() {
                        out.push(mean + scale * radius * theta.sin());
                    }
                }
            }
            _ => return Err(Error::unsupported(&n.op)),
        }
        Ok(vec![HostTensor::new(x.shape.clone(), Data::F64(out))?.cast(dtype)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Attr;
    #[test]
    fn zero_scale_is_the_mean() {
        let x = HostTensor::zeros(DType::F32, vec![3, 7]);
        let mut n = Node::new("RandomNormalLike", "random", &["x"], &["y"]);
        n.attrs.insert("mean".into(), Attr::Float(0.25));
        n.attrs.insert("scale".into(), Attr::Float(0.));
        let out = Streams::default().eval(&n, &[Some(&x)]).unwrap();
        assert!(out[0].as_f64().iter().all(|x| *x == 0.25));
    }

    #[test]
    fn reproducible_streams_advance_and_have_expected_distributions() {
        let x = HostTensor::zeros(DType::F32, vec![100_000]);
        for op in OPS {
            let mut n = Node::new(op, "random", &["x"], &["y"]);
            n.attrs.insert("seed".into(), Attr::Float(42.));
            let mut a = Streams::default();
            let mut b = Streams::default();
            let first = a.eval(&n, &[Some(&x)]).unwrap();
            assert_eq!(first, b.eval(&n, &[Some(&x)]).unwrap());
            assert_ne!(first, a.eval(&n, &[Some(&x)]).unwrap());
            let v = first[0].as_f64();
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let var = v.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / v.len() as f64;
            if *op == "RandomUniformLike" {
                assert!(v.iter().all(|v| *v >= 0. && *v < 1.));
                assert!((mean - 0.5).abs() < 0.01);
                assert!((var - 1. / 12.).abs() < 0.01);
            } else {
                assert!(mean.abs() < 0.02);
                assert!((var - 1.).abs() < 0.03);
            }
        }
    }
}
