//! One run of a compiled program.
//!
//! Nodes are walked in order. Shape and index math stays on the host; float
//! work becomes ggml ops in a graph that is built lazily and computed only
//! when something needs a value back on the host (a "flush"), or at the end.
//! Data therefore crosses the boundary only at real host/device seams, and
//! every crossing is logged.
//!
//! Placement rule, per node:
//! - forced host: shape-only ops, comparisons, int casts, rank > 4, ops
//!   without a ggml emitter
//! - all inputs `Host` (not `Staged`): host
//! - otherwise device; if the emitter declines (`Unsupported`), host

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ggml_sys as g;

use crate::error::{Error, Result};
use crate::exec::backend::GRAPH_SIZE;
use crate::exec::ggml::{self, Ctx};
use crate::exec::program::Program;
use crate::exec::value::{DeviceTensor, Value};
use crate::exec::{ops_binary, ops_nn, ops_shape};
use crate::host::eval;
use crate::host::tensor::HostTensor;
use crate::ir::{DType, Node};
use crate::logging::bytes;

/// Ops that never go to the device: their outputs are shapes, indices or booleans.
pub const HOST_ONLY: &[&str] = &[
    "Shape",
    "Range",
    "ConstantOfShape",
    "ReduceProd",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Equal",
    "And",
    "Or",
    "Not",
    "Constant",
    "NonZero",
];

/// Ops with a ggml emitter.
pub const DEVICE_OPS: &[&str] = &[
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Sqrt",
    "Exp",
    "Log",
    "Sin",
    "Cos",
    "Sigmoid",
    "Elu",
    "Relu",
    "Tanh",
    "Neg",
    "Abs",
    "Erf",
    "GeluErf",
    "Reciprocal",
    "Softmax",
    "LayerNormalization",
    "ReduceMean",
    "ReduceSum",
    "Where",
    "Cast",
    "Clip",
    "Reshape",
    "Unsqueeze",
    "Squeeze",
    "Transpose",
    "Slice",
    "Concat",
    "Gather",
    "Split",
    "Expand",
    "Identity",
    "MatMul",
    "Gemm",
    "Conv",
    "ConvTranspose",
];

pub fn device_capable(op: &str) -> bool {
    DEVICE_OPS.contains(&op)
}

/// An input as the emitters see it: its graph name (for weight lookup) and value.
#[derive(Clone)]
pub struct In {
    pub name: String,
    pub v: Value,
}

struct Upload {
    t: *mut g::ggml_tensor,
    bytes: Vec<u8>,
}

#[derive(Debug, Default, Clone)]
pub struct RunStats {
    pub host_ops: usize,
    pub device_ops: usize,
    pub fallbacks: usize,
    pub flushes: usize,
    pub uploads: usize,
    pub upload_bytes: usize,
    pub readbacks: usize,
    pub readback_bytes: usize,
    pub ggml_nodes: usize,
    pub host_ms: f64,
    pub build_ms: f64,
    pub compute_ms: f64,
}

pub struct Run<'p> {
    pub prog: &'p Program,
    pub ctx: Ctx,
    graph: *mut g::ggml_cgraph,
    values: HashMap<String, Value>,
    uploads: Vec<Upload>,
    pub stats: RunStats,
    node_index: usize,
}

impl<'p> Drop for Run<'p> {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                g::ggml_free(self.ctx);
            }
        }
    }
}

unsafe fn new_graph_ctx() -> Result<(Ctx, *mut g::ggml_cgraph)> {
    let mem = g::ggml_tensor_overhead() * GRAPH_SIZE + g::ggml_graph_overhead_custom(GRAPH_SIZE, false) + (1 << 20);
    let ctx = g::ggml_init(g::ggml_init_params { mem_size: mem, mem_buffer: std::ptr::null_mut(), no_alloc: true });
    if ctx.is_null() {
        return Err(Error::ggml("ggml_init for the run context failed"));
    }
    let graph = g::ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    if graph.is_null() {
        g::ggml_free(ctx);
        return Err(Error::ggml("ggml_new_graph_custom failed"));
    }
    Ok((ctx, graph))
}

impl<'p> Run<'p> {
    pub fn new(prog: &'p Program) -> Result<Run<'p>> {
        let (ctx, graph) = unsafe { new_graph_ctx()? };
        Ok(Run { prog, ctx, graph, values: HashMap::new(), uploads: Vec::new(), stats: RunStats::default(), node_index: 0 })
    }

    // ------------------------------------------------------------ lookups

    fn lookup(&self, name: &str) -> Option<Value> {
        if let Some(v) = self.values.get(name) {
            return Some(v.clone());
        }
        let c = self.prog.graph.constants.get(name)?;
        Some(if c.dtype().is_float() { Value::staged_of(c.clone()) } else { Value::host_of(c.clone()) })
    }

    /// A pre-uploaded weight, if this name is one.
    pub fn weight(&self, name: &str) -> Option<DeviceTensor> {
        self.prog.weights.tensors.get(name).copied()
    }

    // ------------------------------------------------------------ device side

    /// Put host data into the current graph as an input leaf.
    pub fn upload(&mut self, t: &HostTensor, name: &str) -> Result<DeviceTensor> {
        let (dtype, bytes) = match t.dtype() {
            DType::F32 | DType::F16 | DType::F64 | DType::Bool => (DType::F32, t.to_bytes(DType::F32)?),
            DType::I64 | DType::I32 | DType::I8 | DType::U8 => (DType::I32, t.to_bytes(DType::I32)?),
        };
        let d = unsafe { ggml::new_tensor(self.ctx, dtype, &t.shape)? };
        unsafe {
            ggml::set_name(d.t, &format!("in:{name}"));
            g::ggml_set_input(d.t);
        }
        self.stats.uploads += 1;
        self.stats.upload_bytes += bytes.len();
        tracing::trace!(name, shape = ?t.shape, dtype = %dtype, bytes = bytes.len(), "upload");
        self.uploads.push(Upload { t: d.t, bytes });
        Ok(d)
    }

    /// The device tensor for an input: as is, a resident weight, or an upload.
    pub fn dev(&mut self, i: &In) -> Result<DeviceTensor> {
        match &i.v {
            Value::Device(d) => Ok(*d),
            Value::Host(t) | Value::Staged(t) => {
                if let Some(w) = self.weight(&i.name) {
                    return Ok(w);
                }
                let t = t.clone();
                self.upload(&t, &i.name)
            }
        }
    }

    /// Like `dev`, but any integer or bool data is widened to f32 first.
    pub fn dev_f32(&mut self, i: &In) -> Result<DeviceTensor> {
        match &i.v {
            Value::Device(d) => Ok(*d),
            Value::Host(t) | Value::Staged(t) => {
                if let Some(w) = self.weight(&i.name) {
                    return Ok(w);
                }
                let t = if t.dtype() == DType::F32 { t.clone() } else { Arc::new(t.cast(DType::F32)) };
                self.upload(&t, &i.name)
            }
        }
    }

    /// A one-element f32 tensor on the device.
    pub fn scalar(&mut self, v: f32) -> Result<DeviceTensor> {
        self.upload(&HostTensor::f32(vec![1], vec![v]), "scalar")
    }

    /// Host data an emitter needs for shapes or indices. The runtime flushes
    /// device params before calling an emitter, so `Device` here is a bug.
    pub fn host_param<'a>(&self, i: &'a In, what: &str) -> Result<&'a HostTensor> {
        i.v.host().ok_or_else(|| Error::internal(format!("{what}: parameter '{}' is on the device", i.name)))
    }

    // ------------------------------------------------------------ flush

    /// Compute the graph built so far and bring every live device value back
    /// to the host (as `Staged`). Afterwards the graph is empty again.
    pub fn flush(&mut self, reason: &str) -> Result<()> {
        let live: Vec<(String, DeviceTensor)> =
            self.values.iter().filter_map(|(k, v)| v.device().map(|d| (k.clone(), d))).collect();
        if live.is_empty() && self.uploads.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        let sched = self.prog.backend.sched;
        let mut outs = Vec::with_capacity(live.len());
        unsafe {
            for (name, d) in &live {
                let d = ggml::contig(self.ctx, *d);
                g::ggml_set_output(d.t);
                g::ggml_build_forward_expand(self.graph, d.t);
                outs.push((name.clone(), d));
            }
            let n_nodes = g::ggml_graph_n_nodes(self.graph) as usize;
            self.stats.ggml_nodes += n_nodes;
            g::ggml_backend_sched_reset(sched);
            if !g::ggml_backend_sched_alloc_graph(sched, self.graph) {
                return Err(Error::ggml(format!("scheduler could not allocate a graph of {n_nodes} nodes ({reason})")));
            }
            let mut set = 0usize;
            for up in &self.uploads {
                if (*up.t).buffer.is_null() && (*up.t).data.is_null() {
                    tracing::trace!("upload leaf unused by any node, skipped");
                    continue;
                }
                g::ggml_backend_tensor_set(up.t, up.bytes.as_ptr().cast(), 0, up.bytes.len());
                set += 1;
            }
            let t_alloc = started.elapsed();
            let status = g::ggml_backend_sched_graph_compute(sched, self.graph);
            if status != g::ggml_status_GGML_STATUS_SUCCESS {
                return Err(Error::ggml(format!("graph compute failed with status {status} ({reason})")));
            }
            let t_compute = started.elapsed() - t_alloc;
            let mut read_bytes = 0usize;
            for (name, d) in &outs {
                let n = ggml::nelements(d.t);
                let mut data = vec![0f32; n];
                g::ggml_backend_tensor_get(d.t, data.as_mut_ptr().cast(), 0, n * 4);
                read_bytes += n * 4;
                let t = HostTensor::f32(d.shape(), data);
                tracing::trace!(name = %name, value = %t.brief(), "readback");
                self.values.insert(name.clone(), Value::staged_of(t));
            }
            g::ggml_backend_sched_reset(sched);
            g::ggml_free(self.ctx);
            self.ctx = std::ptr::null_mut();
            let (ctx, graph) = new_graph_ctx()?;
            self.ctx = ctx;
            self.graph = graph;
            self.uploads.clear();
            self.stats.flushes += 1;
            self.stats.readbacks += outs.len();
            self.stats.readback_bytes += read_bytes;
            self.stats.compute_ms += t_compute.as_secs_f64() * 1000.0;
            self.stats.build_ms += t_alloc.as_secs_f64() * 1000.0;
            tracing::debug!(
                reason,
                ggml_nodes = n_nodes,
                inputs_set = set,
                readbacks = outs.len(),
                readback = %bytes(read_bytes),
                alloc_ms = format!("{:.2}", t_alloc.as_secs_f64() * 1000.0),
                compute_ms = format!("{:.2}", t_compute.as_secs_f64() * 1000.0),
                "flush"
            );
        }
        Ok(())
    }

    /// Make sure the named inputs are host-readable, flushing if any is on the device.
    fn ensure_host(&mut self, ins: &mut [Option<In>], which: &[usize], reason: &str) -> Result<()> {
        let need = which.iter().any(|&k| ins.get(k).and_then(|i| i.as_ref()).is_some_and(|i| i.v.is_device()));
        if need {
            self.flush(reason)?;
            for i in ins.iter_mut().flatten() {
                if i.v.is_device() {
                    i.v =
                        self.values.get(&i.name).cloned().ok_or_else(|| Error::internal(format!("{} lost in flush", i.name)))?;
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ execution

    pub fn execute(&mut self, inputs: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        let started = Instant::now();
        let graph = &self.prog.graph;
        if inputs.len() != graph.inputs.len() {
            return Err(Error::shape(format!("{} inputs given, graph has {}", inputs.len(), graph.inputs.len())));
        }
        for (desc, t) in graph.inputs.iter().zip(inputs) {
            let v = if t.dtype().is_float() && t.rank() > 0 { Value::staged_of(t) } else { Value::host_of(t) };
            tracing::trace!(input = %desc.name, value = %v.brief(), "graph input");
            self.values.insert(desc.name.clone(), v);
        }

        let n = graph.nodes.len();
        for i in 0..n {
            self.node_index = i;
            let node = &self.prog.graph.nodes[i];
            let mut ins: Vec<Option<In>> = Vec::with_capacity(node.inputs.len());
            for name in &node.inputs {
                if name.is_empty() {
                    ins.push(None);
                } else {
                    let v = self.lookup(name).ok_or_else(|| Error::model(format!("{node}: input '{name}' has no producer")))?;
                    ins.push(Some(In { name: name.clone(), v }));
                }
            }
            let outs = self.run_node(node, ins)?;
            for (name, v) in node.outputs.iter().zip(outs) {
                if !name.is_empty() {
                    if self.prog.backend.options.dump {
                        tracing::trace!(node = %node, output = %name, value = %v.brief(), "value");
                    }
                    self.values.insert(name.clone(), v);
                }
            }
            // free what nothing later reads
            for name in &node.inputs {
                if self.prog.last_use.get(name).copied() == Some(i) {
                    self.values.remove(name);
                }
            }
        }

        // outputs
        if self.prog.graph.outputs.iter().any(|d| self.values.get(&d.name).is_some_and(|v| v.is_device())) {
            self.flush("graph outputs")?;
        }
        let mut outputs = Vec::with_capacity(self.prog.graph.outputs.len());
        for desc in &self.prog.graph.outputs {
            let v = self.lookup(&desc.name).ok_or_else(|| Error::model(format!("output '{}' was never produced", desc.name)))?;
            let t = v.host().ok_or_else(|| Error::internal(format!("output '{}' still on device", desc.name)))?;
            let t = if t.dtype() == desc.dtype { t.clone() } else { t.cast(desc.dtype) };
            tracing::trace!(output = %desc.name, value = %t.brief(), "graph output");
            outputs.push(t);
        }
        let total = started.elapsed().as_secs_f64() * 1000.0;
        let s = &self.stats;
        tracing::debug!(
            host_ops = s.host_ops,
            device_ops = s.device_ops,
            fallbacks = s.fallbacks,
            flushes = s.flushes,
            uploads = s.uploads,
            upload = %bytes(s.upload_bytes),
            readbacks = s.readbacks,
            readback = %bytes(s.readback_bytes),
            ggml_nodes = s.ggml_nodes,
            host_ms = format!("{:.2}", s.host_ms),
            alloc_ms = format!("{:.2}", s.build_ms),
            compute_ms = format!("{:.2}", s.compute_ms),
            total_ms = format!("{:.2}", total),
            "run"
        );
        Ok(outputs)
    }

    fn run_node(&mut self, node: &Node, mut ins: Vec<Option<In>>) -> Result<Vec<Value>> {
        let all_host = ins.iter().flatten().all(|i| i.v.is_host());
        let forced = self.forced_host(node, &ins);
        let place = if forced || all_host { "host" } else { "device" };
        tracing::trace!(
            node = %node,
            place,
            forced,
            inputs = ?ins.iter().map(|i| i.as_ref().map(|i| i.v.brief()).unwrap_or_else(|| "-".into())).collect::<Vec<_>>(),
            "node"
        );
        if place == "device" {
            let params = param_indices(node);
            self.ensure_host(&mut ins, &params, &format!("{node} needs host params"))?;
            match self.emit(node, &ins) {
                Ok(outs) => {
                    self.stats.device_ops += 1;
                    return Ok(outs);
                }
                Err(Error::Unsupported(msg)) => {
                    tracing::debug!(node = %node, %msg, "device emitter declined, using host");
                    self.stats.fallbacks += 1;
                }
                Err(e) => return Err(e),
            }
        }
        // Shape reads metadata only, never element data: answer it from the
        // value's shape so a device input does not force a flush + readback.
        if node.op == "Shape" {
            let x = ins.first().and_then(|i| i.as_ref()).ok_or_else(|| Error::model(format!("{node}: input 0 missing")))?;
            let shape = x.v.shape();
            let rank = shape.len();
            let norm = |axis: i64, default: usize| -> usize {
                let r = rank as i64;
                let a = if axis < 0 { axis + r } else { axis };
                if a < 0 {
                    0
                } else if a > r {
                    default.min(rank)
                } else {
                    a as usize
                }
            };
            let start = norm(node.attr_i("start", 0), 0);
            let end = norm(node.attr_i("end", rank as i64), rank);
            let dims: Vec<i64> = shape[start..end.max(start)].iter().map(|&d| d as i64).collect();
            self.stats.host_ops += 1;
            return Ok(vec![Value::host_of(HostTensor::i64(vec![dims.len()], dims))]);
        }
        // host path
        let all: Vec<usize> = (0..ins.len()).collect();
        self.ensure_host(&mut ins, &all, &format!("{node} on host"))?;
        let started = Instant::now();
        let refs: Vec<Option<&HostTensor>> = ins.iter().map(|i| i.as_ref().and_then(|i| i.v.host())).collect();
        let outs = eval::eval(node, &refs).map_err(|e| Error::Model(format!("{node}: {e}")))?;
        self.stats.host_ops += 1;
        self.stats.host_ms += started.elapsed().as_secs_f64() * 1000.0;
        let any_staged = ins.iter().flatten().any(|i| i.v.is_staged());
        Ok(outs
            .into_iter()
            .map(|t| if t.dtype().is_float() && any_staged && t.rank() > 0 { Value::staged_of(t) } else { Value::host_of(t) })
            .collect())
    }

    fn forced_host(&self, node: &Node, ins: &[Option<In>]) -> bool {
        let op = node.op.as_str();
        if HOST_ONLY.contains(&op) || !device_capable(op) {
            return true;
        }
        if op == "Cast" && !DType::from_onnx(node.attr_i("to", 1) as i32).map(|d| d.is_float()).unwrap_or(false) {
            return true;
        }
        if ins.iter().flatten().any(|i| i.v.rank() > ggml::MAX_RANK) {
            return true;
        }
        predicted_rank(node, ins) > ggml::MAX_RANK
    }

    fn emit(&mut self, node: &Node, ins: &[Option<In>]) -> Result<Vec<Value>> {
        match node.op.as_str() {
            "Reshape" | "Unsqueeze" | "Squeeze" | "Transpose" | "Slice" | "Concat" | "Gather" | "Split" | "Expand"
            | "Identity" => ops_shape::emit(self, node, ins),
            "MatMul" | "Gemm" | "Conv" | "ConvTranspose" => ops_nn::emit(self, node, ins),
            _ => ops_binary::emit(self, node, ins),
        }
    }
}

/// Which inputs an op reads as host parameters (shapes, axes, indices).
pub fn param_indices(node: &Node) -> Vec<usize> {
    match node.op.as_str() {
        "Reshape" | "Unsqueeze" | "Squeeze" | "Expand" | "Split" => vec![1],
        "Slice" => vec![1, 2, 3, 4],
        "Gather" => vec![1],
        "Where" => vec![0],
        "Clip" => vec![1, 2],
        "ReduceMean" | "ReduceSum" => vec![1],
        _ => vec![],
    }
}

/// Output rank before running the op, from host params where the op reshapes.
pub fn predicted_rank(node: &Node, ins: &[Option<In>]) -> usize {
    let rank0 = ins.first().and_then(|i| i.as_ref()).map(|i| i.v.rank()).unwrap_or(0);
    let param = |k: usize| ins.get(k).and_then(|i| i.as_ref()).and_then(|i| i.v.host().map(|t| t.numel()));
    match node.op.as_str() {
        "Reshape" | "Expand" => param(1).unwrap_or(rank0),
        "Unsqueeze" => rank0 + param(1).or_else(|| node.attr_ints("axes").map(|a| a.len())).unwrap_or(0),
        "Squeeze" => rank0.saturating_sub(param(1).or_else(|| node.attr_ints("axes").map(|a| a.len())).unwrap_or(0)),
        "Gather" => {
            let ri = ins.get(1).and_then(|i| i.as_ref()).map(|i| i.v.rank()).unwrap_or(0);
            (rank0 + ri).saturating_sub(1)
        }
        "MatMul" => ins.iter().flatten().map(|i| i.v.rank()).max().unwrap_or(0),
        _ => ins.iter().flatten().map(|i| i.v.rank()).max().unwrap_or(0),
    }
}
