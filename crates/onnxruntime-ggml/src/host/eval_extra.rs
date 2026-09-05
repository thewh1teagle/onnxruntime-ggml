//! Standard elementwise and normalization operators.
use crate::error::{Error, Result};
use crate::host::eval::need;
use crate::host::tensor::{Data, HostTensor};
use crate::ir::Node;

pub const OPS: &[&str] = &["LeakyRelu", "IsNaN", "Round", "Atan", "InstanceNormalization"];

pub fn eval(n: &Node, ins: &[Option<&HostTensor>]) -> Result<Vec<HostTensor>> {
    let x = need(n, ins, 0)?;
    if n.op == "IsNaN" {
        return Ok(vec![HostTensor::bool(x.shape.clone(), x.as_f64().iter().map(|v| v.is_nan()).collect())]);
    }
    let values = x.as_f64();
    let output = match n.op.as_str() {
        "LeakyRelu" => values.iter().map(|&v| if v >= 0. { v } else { v * n.attr_f("alpha", 0.01) as f64 }).collect(),
        "Round" => values.iter().map(|v| v.round_ties_even()).collect(),
        "Atan" => values.iter().map(|v| v.atan()).collect(),
        "InstanceNormalization" => {
            if x.rank() < 3 {
                return Err(Error::shape("InstanceNormalization rank must be >= 3"));
            }
            let scale = need(n, ins, 1)?;
            let bias = need(n, ins, 2)?;
            let channels = x.shape[1];
            if scale.shape != [channels] || bias.shape != [channels] {
                return Err(Error::shape("InstanceNormalization scale/bias"));
            }
            let spatial: usize = x.shape[2..].iter().product();
            if spatial == 0 {
                return Err(Error::shape("InstanceNormalization empty spatial dimensions"));
            }
            let (scale, bias) = (scale.as_f64(), bias.as_f64());
            let mut out = Vec::with_capacity(values.len());
            for (i, row) in values.chunks_exact(spatial).enumerate() {
                let mean = row.iter().sum::<f64>() / spatial as f64;
                let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / spatial as f64;
                let factor = scale[i % channels] / (variance + n.attr_f("epsilon", 1e-5) as f64).sqrt();
                out.extend(row.iter().map(|v| (v - mean) * factor + bias[i % channels]));
            }
            out
        }
        _ => return Err(Error::unsupported(&n.op)),
    };
    Ok(vec![HostTensor::new(x.shape.clone(), Data::F64(output))?.cast(x.dtype())])
}
