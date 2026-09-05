//! ONNX Resize coordinates shared by the host reference and ggml emitter.
use crate::error::{Error, Result};
use crate::host::eval::need;
use crate::host::tensor::{strides_of, Data, HostTensor};
use crate::ir::Node;

pub struct Resize {
    pub shape: Vec<usize>,
    pub scales: Vec<f64>,
    pub linear: bool,
}

impl Resize {
    pub fn new(n: &Node, shape: &[usize], scales: Option<&HostTensor>, sizes: Option<&HostTensor>) -> Result<Self> {
        if n.attr_i("antialias", 0) != 0 || n.attrs.contains_key("axes") || n.attr_i("exclude_outside", 0) != 0 {
            return Err(Error::unsupported("Resize antialias/axes/exclude_outside"));
        }
        if n.attr_str("keep_aspect_ratio_policy").is_some_and(|v| v != "stretch") {
            return Err(Error::unsupported("Resize aspect ratio policy"));
        }
        let mode = n.attr_str("mode").unwrap_or("nearest");
        if !matches!(mode, "nearest" | "linear") {
            return Err(Error::unsupported("Resize interpolation mode"));
        }
        let (output, scales) = if let Some(sizes) = sizes.filter(|s| s.numel() > 0) {
            let values = sizes.as_i64();
            if values.len() != shape.len() || values.iter().any(|&s| s <= 0) || shape.contains(&0) {
                return Err(Error::shape("Resize sizes"));
            }
            let out: Vec<usize> = values.iter().map(|&s| s as usize).collect();
            let scales = out.iter().zip(shape).map(|(&o, &i)| o as f64 / i as f64).collect();
            (out, scales)
        } else {
            let scales = scales.ok_or_else(|| Error::model("Resize needs scales or sizes"))?.as_f64().into_owned();
            if scales.len() != shape.len() || scales.iter().any(|s| !s.is_finite() || *s <= 0.) {
                return Err(Error::shape("Resize scales"));
            }
            let out: Vec<usize> = shape.iter().zip(&scales).map(|(&s, &scale)| (s as f64 * scale).floor() as usize).collect();
            if out.contains(&0) || shape.contains(&0) {
                return Err(Error::shape("Resize empty dimension"));
            }
            (out, scales)
        };
        Ok(Self { shape: output, scales, linear: mode == "linear" })
    }

    pub fn coordinates(&self, n: &Node, axis: usize, input: usize) -> Result<(Vec<i64>, Vec<i64>, Vec<f32>)> {
        let output = self.shape[axis];
        let scale = self.scales[axis] as f32;
        let mut low = Vec::with_capacity(output);
        let mut high = Vec::with_capacity(output);
        let mut weights = Vec::with_capacity(output);
        for i in 0..output {
            let coord = match n.attr_str("coordinate_transformation_mode").unwrap_or("half_pixel") {
                "half_pixel" => (i as f32 + 0.5) / scale - 0.5,
                "pytorch_half_pixel" => {
                    if output > 1 {
                        (i as f32 + 0.5) / scale - 0.5
                    } else {
                        0.
                    }
                }
                "align_corners" => {
                    if output > 1 {
                        i as f32 * (input - 1) as f32 / (output - 1) as f32
                    } else {
                        0.
                    }
                }
                "asymmetric" => i as f32 / scale,
                "tf_half_pixel_for_nn" => (i as f32 + 0.5) / scale,
                _ => return Err(Error::unsupported("Resize coordinate transformation")),
            };
            let (lo, hi, weight) = if self.linear {
                (coord.floor(), coord.floor() + 1., coord - coord.floor())
            } else {
                // Float coordinate transforms can put an exact half tie a few ulps away.
                // Match ORT's nearest-mode tolerance (upsamplebase.h, kNearestModeEps).
                let v = match n.attr_str("nearest_mode").unwrap_or("round_prefer_floor") {
                    "floor" => coord.floor(),
                    "ceil" => coord.ceil(),
                    "round_prefer_floor" => coord.floor() + if coord - coord.floor() > 0.5 + 1e-6 { 1. } else { 0. },
                    "round_prefer_ceil" => coord.floor() + if coord - coord.floor() >= 0.5 - 1e-6 { 1. } else { 0. },
                    _ => return Err(Error::unsupported("Resize nearest mode")),
                };
                (v, v, 0.)
            };
            low.push(lo.clamp(0., (input - 1) as f32) as i64);
            high.push(hi.clamp(0., (input - 1) as f32) as i64);
            weights.push(weight);
        }
        Ok((low, high, weights))
    }
}

pub fn eval(n: &Node, ins: &[Option<&HostTensor>]) -> Result<Vec<HostTensor>> {
    let x = need(n, ins, 0)?;
    let resize = Resize::new(n, &x.shape, ins.get(2).copied().flatten(), ins.get(3).copied().flatten())?;
    let mut x = x.clone();
    for axis in 0..x.rank() {
        if x.shape[axis] == resize.shape[axis] && resize.scales[axis] == 1. {
            continue;
        }
        let (lo, hi, weights) = resize.coordinates(n, axis, x.shape[axis])?;
        let old = x.shape.clone();
        let mut shape = old.clone();
        shape[axis] = resize.shape[axis];
        let srcst = strides_of(&old);
        let dstst = strides_of(&shape);
        let count: usize = shape.iter().product();
        let mut indices = Vec::with_capacity(count);
        let mut data = Vec::with_capacity(count);
        let src = x.as_f64();
        for i in 0..count {
            let k = i / dstst[axis] % shape[axis];
            let base: usize = (0..shape.len()).filter(|&a| a != axis).map(|a| (i / dstst[a] % shape[a]) * srcst[a]).sum();
            let l = base + lo[k] as usize * srcst[axis];
            if resize.linear {
                let h = base + hi[k] as usize * srcst[axis];
                let w = weights[k] as f64;
                data.push(src[l] * (1. - w) + src[h] * w);
            } else {
                indices.push(l);
            }
        }
        x = if resize.linear { HostTensor::new(shape, Data::F64(data))?.cast(x.dtype()) } else { x.gather_flat(shape, &indices) };
    }
    Ok(vec![x])
}
