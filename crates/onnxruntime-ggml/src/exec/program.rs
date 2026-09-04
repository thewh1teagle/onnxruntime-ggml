//! Compile an `ir::Graph` once per session:
//!
//! 1. fold every node whose inputs are all constants (the export's shape
//!    arithmetic mostly disappears here)
//! 2. rewrite patterns ggml has a kernel for (GELU) and pre-transpose MatMul
//!    weights into the layout `ggml_mul_mat` wants (see `exec::fusion`)
//! 3. drop constants nothing reads any more
//! 4. upload the float constants to the primary backend, once

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use ggml_sys as g;

use crate::error::{Error, Result};
use crate::exec::backend::{Backend, WeightPrecision};
use crate::exec::fusion;
use crate::exec::ggml::{self, Ctx};
use crate::exec::runtime::Run;
use crate::exec::value::DeviceTensor;
use crate::host::eval;
use crate::host::eval_shape::transpose;
use crate::host::tensor::HostTensor;
use crate::ir::{DType, Graph};
use crate::logging::bytes;

pub struct Weights {
    pub ctx: Ctx,
    pub buffer: g::ggml_backend_buffer_t,
    pub tensors: HashMap<String, DeviceTensor>,
    pub nbytes: usize,
    /// How many of them are stored as F16, and how many bytes that is.
    pub n_f16: usize,
    pub f16_bytes: usize,
}

unsafe impl Send for Weights {}
unsafe impl Sync for Weights {}

impl Drop for Weights {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                g::ggml_backend_buffer_free(self.buffer);
            }
            if !self.ctx.is_null() {
                g::ggml_free(self.ctx);
            }
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CompileStats {
    pub nodes_in: usize,
    pub nodes_out: usize,
    pub folded: usize,
    pub gelu_fused: usize,
    pub layer_norms_fused: usize,
    pub weights_transposed: usize,
    pub constants: usize,
    pub weights_uploaded: usize,
    pub weight_bytes: usize,
    pub weights_f16: usize,
    pub weight_f16_bytes: usize,
    pub millis: f64,
}

pub struct Program {
    pub name: String,
    pub graph: Graph,
    pub backend: Arc<Backend>,
    pub weights: Weights,
    pub last_use: HashMap<String, usize>,
    pub stats: CompileStats,
}

impl Program {
    pub fn compile(name: &str, mut graph: Graph, backend: Arc<Backend>) -> Result<Program> {
        let started = Instant::now();
        let span = tracing::info_span!("compile", graph = name);
        let _enter = span.enter();
        let mut stats = CompileStats { nodes_in: graph.nodes.len(), ..Default::default() };
        tracing::info!(
            nodes = graph.nodes.len(),
            inputs = graph.inputs.len(),
            outputs = graph.outputs.len(),
            constants = graph.constants.len(),
            "graph imported"
        );

        stats.folded = fold_constants(&mut graph)?;
        stats.gelu_fused = fusion::fuse_gelu(&mut graph);
        stats.layer_norms_fused = fusion::fuse_layer_norm(&mut graph);
        stats.weights_transposed = pretranspose_weights(&mut graph)?;
        prune_constants(&mut graph);
        stats.constants = graph.constants.len();
        stats.nodes_out = graph.nodes.len();

        let weights = upload_weights(&graph, &backend)?;
        stats.weights_uploaded = weights.tensors.len();
        stats.weight_bytes = weights.nbytes;
        stats.weights_f16 = weights.n_f16;
        stats.weight_f16_bytes = weights.f16_bytes;
        let last_use = graph.last_use();
        stats.millis = started.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            nodes_in = stats.nodes_in,
            nodes_out = stats.nodes_out,
            folded = stats.folded,
            gelu_fused = stats.gelu_fused,
            layer_norms_fused = stats.layer_norms_fused,
            weights_transposed = stats.weights_transposed,
            weights = stats.weights_uploaded,
            weight_bytes = %bytes(stats.weight_bytes),
            weights_f16 = stats.weights_f16,
            weight_f16_bytes = %bytes(stats.weight_f16_bytes),
            ms = format!("{:.1}", stats.millis),
            "compiled"
        );
        for (op, n) in graph.op_histogram() {
            tracing::debug!(op, n, "op after compile");
        }
        Ok(Program { name: name.to_owned(), graph, backend, weights, last_use, stats })
    }

    /// Run the program on host inputs given in `graph.inputs` order; returns
    /// outputs in `graph.outputs` order.
    pub fn run(&self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        let _guard = self.backend.lock.lock().map_err(|_| Error::internal("backend lock poisoned"))?;
        let mut run = Run::new(self)?;
        run.execute(inputs)
    }
}

/// Evaluate every node whose inputs are all constants. Returns how many nodes went away.
pub fn fold_constants(graph: &mut Graph) -> Result<usize> {
    let mut folded = 0usize;
    let mut kept = Vec::with_capacity(graph.nodes.len());
    let nodes = std::mem::take(&mut graph.nodes);
    for node in nodes {
        let all_const = node.inputs.iter().all(|i| i.is_empty() || graph.constants.contains_key(i));
        if all_const && eval::supported(&node.op) && !node.outputs.iter().any(|o| graph.outputs.iter().any(|d| &d.name == o)) {
            let ins: Vec<Option<&HostTensor>> = node.inputs.iter().map(|i| graph.constants.get(i)).collect();
            match eval::eval(&node, &ins) {
                Ok(outs) => {
                    for (name, t) in node.outputs.iter().zip(outs) {
                        if !name.is_empty() {
                            tracing::trace!(node = %node, value = name, result = %t.brief(), "folded");
                            graph.constants.insert(name.clone(), t);
                        }
                    }
                    folded += 1;
                    continue;
                }
                Err(err) => {
                    tracing::debug!(node = %node, %err, "not folded");
                }
            }
        }
        kept.push(node);
    }
    graph.nodes = kept;
    tracing::info!(folded, remaining = graph.nodes.len(), "constant folding");
    Ok(folded)
}

/// `ggml_mul_mat` wants the weight as `[N, K]` row-major (ONNX Gemm transB=1).
/// MatMul weights are `[K, N]`; transpose them once here.
pub fn pretranspose_weights(graph: &mut Graph) -> Result<usize> {
    let mut n = 0usize;
    let mut new_consts: Vec<(String, HostTensor)> = Vec::new();
    for node in &mut graph.nodes {
        match node.op.as_str() {
            "MatMul" => {
                let b = node.inputs[1].clone();
                if let Some(w) = graph.constants.get(&b) {
                    if w.rank() == 2 && w.dtype().is_float() {
                        let tname = format!("{b}__T");
                        if !graph.constants.contains_key(&tname) && !new_consts.iter().any(|(k, _)| k == &tname) {
                            new_consts.push((tname.clone(), transpose(w, &[1, 0])?));
                        }
                        node.inputs[1] = tname;
                        node.set_attr_i("__b_transposed", 1);
                        n += 1;
                    }
                }
            }
            "Gemm" if node.attr_i("transB", 0) == 0 => {
                let b = node.inputs[1].clone();
                if let Some(w) = graph.constants.get(&b) {
                    if w.rank() == 2 {
                        let tname = format!("{b}__T");
                        if !graph.constants.contains_key(&tname) && !new_consts.iter().any(|(k, _)| k == &tname) {
                            new_consts.push((tname.clone(), transpose(w, &[1, 0])?));
                        }
                        node.inputs[1] = tname;
                        node.set_attr_i("transB", 1);
                        n += 1;
                    }
                }
            }
            _ => {}
        }
    }
    for (k, v) in new_consts {
        graph.constants.insert(k, v);
    }
    if n > 0 {
        tracing::info!(n, "weights pre-transposed for mul_mat");
    }
    Ok(n)
}

pub fn prune_constants(graph: &mut Graph) {
    let mut used: HashSet<&str> = HashSet::new();
    for node in &graph.nodes {
        for i in &node.inputs {
            used.insert(i.as_str());
        }
    }
    for o in &graph.outputs {
        used.insert(o.name.as_str());
    }
    let used: HashSet<String> = used.into_iter().map(|s| s.to_owned()).collect();
    let before = graph.constants.len();
    graph.constants.retain(|k, _| used.contains(k));
    tracing::debug!(before, after = graph.constants.len(), "constants pruned");
}

/// The 2-D weight matrices `ggml_mul_mat` reads as src0: MatMul operands
/// pre-transposed by `pretranspose_weights`, and Gemm B operands with
/// `transB=1`. Only these are candidates for F16 storage; a name used anywhere
/// else keeps its F32 copy, because the emitters there assume F32 data.
fn mul_mat_weights(graph: &Graph) -> HashSet<String> {
    let mut src0: HashSet<&str> = HashSet::new();
    let mut other: HashSet<&str> = HashSet::new();
    for node in &graph.nodes {
        let b = match node.op.as_str() {
            "MatMul" if node.attr_i("__b_transposed", 0) != 0 => Some(1),
            "Gemm" if node.attr_i("transB", 0) != 0 => Some(1),
            _ => None,
        };
        for (i, name) in node.inputs.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            if Some(i) == b {
                src0.insert(name);
            } else {
                other.insert(name);
            }
        }
    }
    for o in &graph.outputs {
        other.insert(o.name.as_str());
    }
    src0.difference(&other)
        .filter(|n| graph.constants.get(**n).is_some_and(|t| t.rank() == 2 && t.dtype().is_float()))
        .map(|n| (*n).to_owned())
        .collect()
}

/// Float constants of rank <= 4 go to the primary backend once. Everything
/// else stays host-only (ints are shape math; rank > 4 runs on the host).
///
/// With `weights=f16` the 2-D matmul weights are stored as F16: `ggml_mul_mat`
/// takes an F16 src0 against an F32 src1 on both Metal and CPU, which halves
/// the bytes read per matmul. Vectors (biases, norm scales) stay F32.
fn upload_weights(graph: &Graph, backend: &Backend) -> Result<Weights> {
    let candidates: Vec<(&String, &HostTensor)> =
        graph.constants.iter().filter(|(_, t)| t.dtype().is_float() && t.rank() <= ggml::MAX_RANK && t.numel() > 0).collect();
    let half = match backend.options.weights {
        WeightPrecision::F16 => mul_mat_weights(graph),
        WeightPrecision::F32 => HashSet::new(),
    };
    unsafe {
        let ctx = g::ggml_init(g::ggml_init_params {
            mem_size: g::ggml_tensor_overhead() * (candidates.len() + 1),
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        });
        if ctx.is_null() {
            return Err(Error::ggml("ggml_init for weights failed"));
        }
        let mut tensors = HashMap::new();
        for (name, t) in &candidates {
            let dtype = if half.contains(*name) { DType::F16 } else { DType::F32 };
            let d = ggml::new_tensor(ctx, dtype, &t.shape)?;
            ggml::set_name(d.t, name);
            tensors.insert((*name).clone(), d);
        }
        let buffer = if candidates.is_empty() {
            std::ptr::null_mut()
        } else {
            let b = g::ggml_backend_alloc_ctx_tensors(ctx, backend.primary);
            if b.is_null() {
                g::ggml_free(ctx);
                return Err(Error::ggml("could not allocate the weight buffer on the primary backend"));
            }
            b
        };
        let (mut nbytes, mut n_f16, mut f16_bytes) = (0usize, 0usize, 0usize);
        for (name, t) in &candidates {
            let d = tensors[*name];
            let f16 = half.contains(*name);
            let len = if f16 {
                let data = t.to_bytes(DType::F16)?;
                g::ggml_backend_tensor_set(d.t, data.as_ptr().cast(), 0, data.len());
                n_f16 += 1;
                f16_bytes += data.len();
                data.len()
            } else {
                let data = t.as_f32();
                let len = data.len() * 4;
                g::ggml_backend_tensor_set(d.t, data.as_ptr().cast(), 0, len);
                len
            };
            nbytes += len;
            tracing::trace!(name = %name, shape = ?t.shape, bytes = len, f16, "weight uploaded");
        }
        tracing::info!(
            n = tensors.len(),
            bytes = %bytes(nbytes),
            f16 = n_f16,
            f16_bytes = %bytes(f16_bytes),
            backend = %backend.primary_name,
            "weights resident"
        );
        Ok(Weights { ctx, buffer, tensors, nbytes, n_f16, f16_bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Attr, Node, ValueDesc};

    fn desc(name: &str) -> ValueDesc {
        ValueDesc { name: name.into(), dtype: DType::F32, shape: vec![] }
    }

    #[test]
    fn folds_constant_chain() {
        let mut graph = Graph::default();
        graph.constants.insert("a".into(), HostTensor::i64(vec![2], vec![2, 3]));
        graph.nodes.push(Node::new("ReduceProd", "p", &["a"], &["prod"]));
        let mut u = Node::new("Unsqueeze", "u", &["prod", "ax"], &["out"]);
        u.attrs.insert("keepdims".into(), Attr::Int(0));
        graph.constants.insert("ax".into(), HostTensor::i64(vec![1], vec![0]));
        graph.nodes.push(u);
        graph.outputs.push(desc("out"));
        // last node feeds a graph output, so it stays; the ReduceProd folds
        let folded = fold_constants(&mut graph).unwrap();
        assert_eq!(folded, 1);
        assert_eq!(graph.constants["prod"].as_i64()[0], 6);
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn transposes_matmul_weights() {
        let mut graph = Graph::default();
        graph.constants.insert("w".into(), HostTensor::f32(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]));
        graph.nodes.push(Node::new("MatMul", "mm", &["x", "w"], &["y"]));
        assert_eq!(pretranspose_weights(&mut graph).unwrap(), 1);
        assert_eq!(graph.nodes[0].inputs[1], "w__T");
        assert_eq!(graph.constants["w__T"].shape, vec![3, 2]);
        assert_eq!(graph.nodes[0].attr_i("__b_transposed", 0), 1);
    }
}
