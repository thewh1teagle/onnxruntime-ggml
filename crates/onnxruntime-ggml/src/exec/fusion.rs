//! Compile-time pattern rewrites: subgraphs an exporter emitted as a dozen
//! primitive nodes, matched back into the single op ggml has a kernel for.
//!
//! Every rewrite works the same way: find the nodes, check that nothing else
//! reads the intermediates, then `splice` one new node in at the position of
//! the earliest node removed (which is after every producer it reads).

use std::collections::{HashMap, HashSet};

use crate::ir::{Graph, Node};

/// Replace `remove` (indices into `graph.nodes`) with `new`, placed where the
/// earliest removed node was.
fn splice(graph: &mut Graph, remove: &[usize], new: Node) {
    let first = *remove.iter().min().expect("splice with no nodes to remove");
    let removed: HashSet<usize> = remove.iter().copied().collect();
    let mut new = Some(new);
    let mut nodes = Vec::with_capacity(graph.nodes.len());
    for (i, n) in graph.nodes.drain(..).enumerate() {
        if i == first {
            nodes.push(new.take().expect("splice inserts once"));
        }
        if !removed.contains(&i) {
            nodes.push(n);
        }
    }
    graph.nodes = nodes;
}

fn const_scalar(graph: &Graph, name: &str) -> Option<f64> {
    graph.constants.get(name).filter(|t| t.numel() == 1).and_then(|t| t.scalar_f64().ok())
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-4 * b.abs().max(1.0)
}

/// `0.5 * x * (1 + erf(x / sqrt(2)))` in the shapes torch exports it, into one GeluErf node.
pub fn fuse_gelu(graph: &mut Graph) -> usize {
    let mut fused = 0usize;
    loop {
        let producers = graph.producers();
        let consumers = graph.consumer_counts();
        let mut found: Option<(Vec<usize>, String, String, String)> = None;
        for (ei, erf) in graph.nodes.iter().enumerate() {
            if erf.op != "Erf" {
                continue;
            }
            let Some(&di) = erf.input(0).and_then(|n| producers.get(n)) else { continue };
            let div = &graph.nodes[di];
            let x = match div.op.as_str() {
                "Div" if const_scalar(graph, &div.inputs[1]).is_some_and(|c| near(c, std::f64::consts::SQRT_2)) => {
                    div.inputs[0].clone()
                }
                "Mul" if const_scalar(graph, &div.inputs[1]).is_some_and(|c| near(c, std::f64::consts::FRAC_1_SQRT_2)) => {
                    div.inputs[0].clone()
                }
                "Mul" if const_scalar(graph, &div.inputs[0]).is_some_and(|c| near(c, std::f64::consts::FRAC_1_SQRT_2)) => {
                    div.inputs[1].clone()
                }
                _ => continue,
            };
            let erf_out = &erf.outputs[0];
            if consumers.get(erf_out.as_str()) != Some(&1) || consumers.get(div.outputs[0].as_str()) != Some(&1) {
                continue;
            }
            let Some((ai, add)) = graph.nodes.iter().enumerate().find(|(_, n)| n.op == "Add" && n.inputs.contains(erf_out))
            else {
                continue;
            };
            let other = if add.inputs[0] == *erf_out { &add.inputs[1] } else { &add.inputs[0] };
            if !const_scalar(graph, other).is_some_and(|c| near(c, 1.0)) || consumers.get(add.outputs[0].as_str()) != Some(&1) {
                continue;
            }
            let add_out = add.outputs[0].clone();
            let Some((mi, mul)) = graph.nodes.iter().enumerate().find(|(_, n)| n.op == "Mul" && n.inputs.contains(&add_out))
            else {
                continue;
            };
            let other = if mul.inputs[0] == add_out { mul.inputs[1].clone() } else { mul.inputs[0].clone() };
            let mut remove = vec![di, ei, ai, mi];
            #[allow(clippy::needless_late_init)]
            let out_name;
            if other == x {
                // (x * (1 + erf)) * 0.5
                if consumers.get(mul.outputs[0].as_str()) != Some(&1) {
                    continue;
                }
                let mul_out = mul.outputs[0].clone();
                let Some((hi, half)) = graph.nodes.iter().enumerate().find(|(_, n)| n.op == "Mul" && n.inputs.contains(&mul_out))
                else {
                    continue;
                };
                let hother = if half.inputs[0] == mul_out { &half.inputs[1] } else { &half.inputs[0] };
                if !const_scalar(graph, hother).is_some_and(|c| near(c, 0.5)) {
                    continue;
                }
                remove.push(hi);
                out_name = half.outputs[0].clone();
            } else {
                // (x * 0.5) * (1 + erf)
                let Some(&hi) = producers.get(other.as_str()) else { continue };
                let half = &graph.nodes[hi];
                if half.op != "Mul" {
                    continue;
                }
                let hx = if half.inputs[0] == x {
                    &half.inputs[1]
                } else if half.inputs[1] == x {
                    &half.inputs[0]
                } else {
                    continue;
                };
                if !const_scalar(graph, hx).is_some_and(|c| near(c, 0.5)) || consumers.get(other.as_str()) != Some(&1) {
                    continue;
                }
                remove.push(hi);
                out_name = mul.outputs[0].clone();
            }
            found = Some((remove, x, out_name, erf.name.clone()));
            break;
        }
        let Some((remove, x, out, name)) = found else { break };
        let mut gelu = Node::new("GeluErf", &format!("{name}_gelu"), &[&x], &[&out]);
        gelu.domain = String::new();
        splice(graph, &remove, gelu);
        fused += 1;
        tracing::trace!(x, out, "fused GeluErf");
    }
    if fused > 0 {
        tracing::info!(fused, "gelu fusion");
    }
    fused
}

/// The single axis a ReduceMean reduces, if it has exactly one and keeps dims.
fn mean_axis(graph: &Graph, n: &Node) -> Option<i64> {
    if n.op != "ReduceMean" || n.attr_i("keepdims", 1) == 0 || n.attr_i("noop_with_empty_axes", 0) != 0 {
        return None;
    }
    let axes = match n.attr_ints("axes") {
        Some(a) => a,
        None => graph.constants.get(n.inputs.get(1)?)?.as_i64().to_vec(),
    };
    (axes.len() == 1).then(|| axes[0])
}

/// The layer normalisation torch exports below opset 17, as nine nodes:
///
/// ```text
/// m = ReduceMean(x, -1); d = Sub(x, m); v = ReduceMean(Pow(d, 2), -1)
/// y = Mul(Div(d, Sqrt(Add(v, eps))), scale) + bias
/// ```
///
/// becomes one `LayerNormalization`, which `ops_binary` emits as `ggml_norm`.
/// Without this every `Pow` runs on the host and forces a flush, which on the
/// whisper encoder means 65 round trips through host memory per run.
pub fn fuse_layer_norm(graph: &mut Graph) -> usize {
    let mut fused = 0usize;
    loop {
        let producers = graph.producers();
        let consumers = graph.consumer_counts();
        let one = |name: &str| consumers.get(name) == Some(&1);
        let mut found: Option<(Vec<usize>, Node)> = None;

        for (pi, pow) in graph.nodes.iter().enumerate() {
            if pow.op != "Pow" || !const_scalar(graph, &pow.inputs[1]).is_some_and(|c| near(c, 2.0)) {
                continue;
            }
            // Sub(x, ReduceMean(x, axis))
            let Some(&si) = pow.input(0).and_then(|n| producers.get(n)) else { continue };
            let sub = &graph.nodes[si];
            if sub.op != "Sub" {
                continue;
            }
            let Some(&mi) = producers.get(sub.inputs[1].as_str()) else { continue };
            let Some(axis) = mean_axis(graph, &graph.nodes[mi]) else { continue };
            let x = sub.inputs[0].clone();
            if graph.nodes[mi].inputs[0] != x || !one(&sub.inputs[1]) {
                continue;
            }
            // ReduceMean(Pow, axis) -> Add(eps) -> Sqrt
            let Some((vi, var)) = graph.nodes.iter().enumerate().find(|(_, n)| n.input(0) == Some(&pow.outputs[0])) else {
                continue;
            };
            if mean_axis(graph, var) != Some(axis) || !one(&pow.outputs[0]) {
                continue;
            }
            let Some((ai, eps_add)) =
                graph.nodes.iter().enumerate().find(|(_, n)| n.op == "Add" && n.inputs.contains(&var.outputs[0]))
            else {
                continue;
            };
            if !one(&var.outputs[0]) {
                continue;
            }
            let other = if eps_add.inputs[0] == var.outputs[0] { &eps_add.inputs[1] } else { &eps_add.inputs[0] };
            let Some(eps) = const_scalar(graph, other) else { continue };
            let Some((qi, sqrt)) = graph.nodes.iter().enumerate().find(|(_, n)| n.input(0) == Some(&eps_add.outputs[0])) else {
                continue;
            };
            if sqrt.op != "Sqrt" || !one(&eps_add.outputs[0]) {
                continue;
            }
            // Div(d, sqrt) -> Mul(scale) -> optional Add(bias)
            let Some((di, div)) = graph.nodes.iter().enumerate().find(|(_, n)| {
                n.op == "Div" && n.input(0) == Some(sub.outputs[0].as_str()) && n.input(1) == Some(sqrt.outputs[0].as_str())
            }) else {
                continue;
            };
            if !one(&sqrt.outputs[0]) {
                continue;
            }
            // `d` feeds both Pow and Div and nothing else
            if consumers.get(sub.outputs[0].as_str()) != Some(&2) {
                continue;
            }
            let Some((si2, scale_mul)) =
                graph.nodes.iter().enumerate().find(|(_, n)| n.op == "Mul" && n.inputs.contains(&div.outputs[0]))
            else {
                continue;
            };
            if !one(&div.outputs[0]) {
                continue;
            }
            let scale =
                if scale_mul.inputs[0] == div.outputs[0] { scale_mul.inputs[1].clone() } else { scale_mul.inputs[0].clone() };
            if !graph.constants.contains_key(&scale) {
                continue;
            }
            let mut remove = vec![mi, si, pi, vi, ai, qi, di, si2];
            let mut inputs = vec![x, scale];
            let mut out = scale_mul.outputs[0].clone();
            if one(&scale_mul.outputs[0]) {
                if let Some((bi, bias_add)) =
                    graph.nodes.iter().enumerate().find(|(_, n)| n.op == "Add" && n.inputs.contains(&out))
                {
                    let bias = if bias_add.inputs[0] == out { bias_add.inputs[1].clone() } else { bias_add.inputs[0].clone() };
                    if graph.constants.contains_key(&bias) {
                        remove.push(bi);
                        inputs.push(bias);
                        out = bias_add.outputs[0].clone();
                    }
                }
            }
            let refs: Vec<&str> = inputs.iter().map(|s| s.as_str()).collect();
            let mut ln = Node::new("LayerNormalization", &format!("{}_layernorm", scale_mul.name), &refs, &[&out]);
            ln.domain = String::new();
            ln.set_attr_i("axis", axis);
            ln.attrs.insert("epsilon".into(), crate::ir::Attr::Float(eps as f32));
            found = Some((remove, ln));
            break;
        }
        let Some((remove, ln)) = found else { break };
        tracing::trace!(x = %ln.inputs[0], out = %ln.outputs[0], "fused LayerNormalization");
        splice(graph, &remove, ln);
        fused += 1;
    }
    if fused > 0 {
        tracing::info!(fused, "layer norm fusion");
    }
    fused
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::host::tensor::HostTensor;
    use crate::ir::{Attr, DType, ValueDesc};

    fn desc(name: &str) -> ValueDesc {
        ValueDesc { name: name.into(), dtype: DType::F32, shape: vec![] }
    }

    #[test]
    fn fuses_gelu() {
        let mut graph = Graph::default();
        graph.constants.insert("sqrt2".into(), Arc::new(HostTensor::scalar_f32(std::f32::consts::SQRT_2)));
        graph.constants.insert("one".into(), Arc::new(HostTensor::scalar_f32(1.0)));
        graph.constants.insert("half".into(), Arc::new(HostTensor::scalar_f32(0.5)));
        graph.nodes.push(Node::new("Div", "d", &["x", "sqrt2"], &["d_out"]));
        graph.nodes.push(Node::new("Erf", "e", &["d_out"], &["e_out"]));
        graph.nodes.push(Node::new("Add", "a", &["e_out", "one"], &["a_out"]));
        graph.nodes.push(Node::new("Mul", "m", &["x", "a_out"], &["m_out"]));
        graph.nodes.push(Node::new("Mul", "h", &["m_out", "half"], &["y"]));
        graph.outputs.push(desc("y"));
        assert_eq!(fuse_gelu(&mut graph), 1);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op, "GeluErf");
        assert_eq!(graph.nodes[0].inputs, vec!["x"]);
        assert_eq!(graph.nodes[0].outputs, vec!["y"]);
    }

    #[test]
    fn fuses_layer_norm() {
        let mut graph = Graph::default();
        graph.constants.insert("two".into(), Arc::new(HostTensor::scalar_f32(2.0)));
        graph.constants.insert("eps".into(), Arc::new(HostTensor::scalar_f32(1e-5)));
        graph.constants.insert("w".into(), Arc::new(HostTensor::f32(vec![2], vec![1.0, 1.0])));
        graph.constants.insert("b".into(), Arc::new(HostTensor::f32(vec![2], vec![0.0, 0.0])));
        let mut mean = Node::new("ReduceMean", "rm", &["x"], &["m"]);
        mean.attrs.insert("axes".into(), Attr::Ints(vec![-1]));
        graph.nodes.push(mean);
        graph.nodes.push(Node::new("Sub", "sub", &["x", "m"], &["d"]));
        graph.nodes.push(Node::new("Pow", "pow", &["d", "two"], &["p"]));
        let mut var = Node::new("ReduceMean", "rm1", &["p"], &["v"]);
        var.attrs.insert("axes".into(), Attr::Ints(vec![-1]));
        graph.nodes.push(var);
        graph.nodes.push(Node::new("Add", "ae", &["v", "eps"], &["ve"]));
        graph.nodes.push(Node::new("Sqrt", "sq", &["ve"], &["s"]));
        graph.nodes.push(Node::new("Div", "dv", &["d", "s"], &["n"]));
        graph.nodes.push(Node::new("Mul", "mu", &["n", "w"], &["nw"]));
        graph.nodes.push(Node::new("Add", "ab", &["nw", "b"], &["y"]));
        graph.outputs.push(desc("y"));
        assert_eq!(fuse_layer_norm(&mut graph), 1);
        assert_eq!(graph.nodes.len(), 1);
        let n = &graph.nodes[0];
        assert_eq!(n.op, "LayerNormalization");
        assert_eq!(n.inputs, vec!["x", "w", "b"]);
        assert_eq!(n.outputs, vec!["y"]);
        assert_eq!(n.attr_i("axis", 0), -1);
        assert!((n.attr_f("epsilon", 0.0) - 1e-5).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------- attention

/// Insert `new` where the last removed node stood (all of its inputs exist by then).
fn splice_at_last(graph: &mut Graph, remove: &[usize], new: Node) {
    let last = *remove.iter().max().expect("splice with no nodes to remove");
    let removed: HashSet<usize> = remove.iter().copied().collect();
    let mut new = Some(new);
    let mut nodes = Vec::with_capacity(graph.nodes.len());
    for (i, n) in graph.nodes.drain(..).enumerate() {
        if !removed.contains(&i) {
            nodes.push(n);
        }
        if i == last {
            nodes.push(new.take().expect("splice inserts once"));
        }
    }
    graph.nodes = nodes;
}

/// Drop nodes whose outputs nobody reads (left behind by a fusion), repeatedly.
pub fn prune_dead_nodes(graph: &mut Graph) -> usize {
    let mut removed = 0usize;
    loop {
        let counts = graph.consumer_counts();
        let dead: Vec<usize> = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.outputs.iter().all(|o| o.is_empty() || counts.get(o.as_str()).copied().unwrap_or(0) == 0))
            .map(|(i, _)| i)
            .collect();
        if dead.is_empty() {
            break;
        }
        let dead: HashSet<usize> = dead.into_iter().collect();
        removed += dead.len();
        let nodes = std::mem::take(&mut graph.nodes);
        graph.nodes = nodes.into_iter().enumerate().filter(|(i, _)| !dead.contains(i)).map(|(_, n)| n).collect();
    }
    if removed > 0 {
        tracing::debug!(removed, "dead nodes pruned");
    }
    removed
}

fn is_neg_inf_fill(graph: &Graph, name: &str, producers: &HashMap<&str, usize>) -> bool {
    if let Some(t) = graph.constants.get(name) {
        return t.numel() >= 1 && t.as_f64().iter().all(|&v| v <= -1e30);
    }
    match producers.get(name).map(|&i| &graph.nodes[i]) {
        Some(n) if n.op == "ConstantOfShape" => n.attr_tensor("value").map(|v| v.as_f64()[0] <= -1e30).unwrap_or(false),
        _ => false,
    }
}

/// `Softmax(scale * Q·Kᵀ [masked]) · V` into one `FusedAttention(q, k, v[, mask])`.
///
/// Anchored at every `Softmax(axis=-1)` whose only consumer is a `MatMul`
/// taking it as the left operand. Walking up from the softmax: an optional
/// `Where(cond, x, -inf)` (the mask), an optional `Div`/`Mul` by a scalar
/// (the scale), then the `MatMul(q, kᵀ)`. Q, K and V are passed in the
/// layouts the export produced (`[.., T, D]`, `[.., D, S]`, `[.., S, D]`);
/// the emitter permutes. The exporter's trailing Transpose/Reshape stay.
pub fn fuse_attention(graph: &mut Graph) -> usize {
    let mut fused = 0usize;
    loop {
        let producers = graph.producers();
        let consumers = graph.consumer_counts();
        let mut found: Option<(Vec<usize>, Node)> = None;
        for (si, sm) in graph.nodes.iter().enumerate() {
            if sm.op != "Softmax" || sm.attr_i("axis", -1) != -1 {
                continue;
            }
            let sm_out = sm.outputs[0].as_str();
            if consumers.get(sm_out) != Some(&1) {
                continue;
            }
            let Some((mi, mm2)) = graph.nodes.iter().enumerate().find(|(_, n)| n.op == "MatMul" && n.inputs[0] == sm_out) else {
                continue;
            };
            let mut remove = vec![si, mi];
            let mut x = sm.inputs[0].clone();
            let mut mask: Option<String> = None;
            let mut scale = 1.0f64;
            let mut mm1: Option<usize> = None;
            let mut ok = true;
            for _ in 0..4 {
                let Some(&pi) = producers.get(x.as_str()) else {
                    ok = false;
                    break;
                };
                if consumers.get(x.as_str()) != Some(&1) {
                    ok = false;
                    break;
                }
                let p = &graph.nodes[pi];
                match p.op.as_str() {
                    "Where" if mask.is_none() && is_neg_inf_fill(graph, &p.inputs[2], &producers) => {
                        mask = Some(p.inputs[0].clone());
                        x = p.inputs[1].clone();
                        remove.push(pi);
                    }
                    "Div" | "Mul" => {
                        let Some(c) =
                            graph.constants.get(&p.inputs[1]).filter(|t| t.numel() == 1).and_then(|t| t.scalar_f64().ok())
                        else {
                            ok = false;
                            break;
                        };
                        scale = if p.op == "Div" { scale / c } else { scale * c };
                        x = p.inputs[0].clone();
                        remove.push(pi);
                    }
                    "MatMul" => {
                        mm1 = Some(pi);
                        remove.push(pi);
                        break;
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            let (Some(mm1), true) = (mm1, ok) else { continue };
            let q = graph.nodes[mm1].inputs[0].clone();
            let k = graph.nodes[mm1].inputs[1].clone();
            let v = mm2.inputs[1].clone();
            let out = mm2.outputs[0].clone();
            let mut inputs = vec![q.as_str(), k.as_str(), v.as_str()];
            if let Some(m) = &mask {
                inputs.push(m.as_str());
            }
            let mut node = Node::new("FusedAttention", &format!("{}_fused", sm.name), &inputs, &[&out]);
            node.attrs.insert("scale".into(), crate::ir::Attr::Float(scale as f32));
            found = Some((remove, node));
            break;
        }
        let Some((remove, node)) = found else { break };
        tracing::trace!(node = %node.name, inputs = ?node.inputs, "fused attention");
        splice_at_last(graph, &remove, node);
        fused += 1;
    }
    if fused > 0 {
        let pruned = prune_dead_nodes(graph);
        tracing::info!(fused, pruned, "attention fusion");
    }
    fused
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use crate::host::tensor::HostTensor;
    use crate::ir::ValueDesc;

    #[test]
    fn fuses_masked_scaled_attention() {
        let mut g = Graph::default();
        g.constants.insert("eight".into(), std::sync::Arc::new(HostTensor::scalar_f32(8.0)));
        g.constants.insert("ninf".into(), std::sync::Arc::new(HostTensor::scalar_f32(f32::NEG_INFINITY)));
        g.nodes.push(Node::new("MatMul", "mm1", &["q", "kt"], &["s"]));
        g.nodes.push(Node::new("Div", "div", &["s", "eight"], &["sd"]));
        g.nodes.push(Node::new("Where", "where", &["mask", "sd", "ninf"], &["sm_in"]));
        let mut sm = Node::new("Softmax", "sm", &["sm_in"], &["p"]);
        sm.set_attr_i("axis", -1);
        g.nodes.push(sm);
        g.nodes.push(Node::new("MatMul", "mm2", &["p", "v"], &["y"]));
        g.outputs.push(ValueDesc { name: "y".into(), dtype: crate::ir::DType::F32, shape: vec![] });
        assert_eq!(fuse_attention(&mut g), 1);
        assert_eq!(g.nodes.len(), 1);
        let n = &g.nodes[0];
        assert_eq!(n.op, "FusedAttention");
        assert_eq!(n.inputs, vec!["q", "kt", "v", "mask"]);
        assert!((n.attr_f("scale", 0.0) - 0.125).abs() < 1e-7);
    }
}
