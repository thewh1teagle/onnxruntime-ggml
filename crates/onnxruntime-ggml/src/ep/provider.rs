//! `OrtEp`: one per session. Claims the graph in `GetCapability`, compiles it
//! in `Compile`, and owns the ggml backend the compiled programs run on.
//!
//! Claiming policy: all or nothing. A subgraph is claimed only when every op
//! in it is supported, because each boundary between this provider and
//! onnxruntime's CPU provider copies tensors every run. `partial=1` overrides
//! that for experiments.

use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::{Arc, Mutex};

use ort_ep_sys::*;

use crate::ep::compute::ComputeInfo;
use crate::ep::options;
use crate::error::{Error, Result};
use crate::exec::backend::{Backend, Options};
use crate::exec::runtime::device_capable;
use crate::exec::Program;
use crate::host::eval;
use crate::ort::api::{cstr, guard, log};
use crate::{ep_call, ort_call, EP_VERSION};

#[repr(C)]
pub struct Provider {
    base: OrtEp,
    pub name: CString,
    compat: CString,
    pub logger: *const OrtLogger,
    pub options: Options,
    pub backend: Arc<Backend>,
    pub programs: Mutex<Vec<Arc<Program>>>,
}

/// Ops that exist only to be fused away at compile time (`exec::fusion`).
pub const FUSED_ONLY: &[&str] = &["DynamicQuantizeLinear", "MatMulInteger"];

pub fn op_supported(op: &str, domain: &str) -> bool {
    (domain.is_empty() || domain == "ai.onnx") && (eval::supported(op) || device_capable(op) || FUSED_ONLY.contains(&op))
}

impl Provider {
    /// # Safety
    /// `session_options` and `logger` are the pointers onnxruntime passed to `CreateEp`.
    pub unsafe fn create(name: &str, session_options: *const OrtSessionOptions, logger: *const OrtLogger) -> Result<*mut OrtEp> {
        let opts = options::from_session(name, session_options);
        let backend = Arc::new(Backend::new(opts.clone())?);
        log(logger, ORT_LOGGING_LEVEL_INFO, &format!("onnxruntime-ggml {EP_VERSION}: primary backend {}", backend.primary_name));
        let mut base: OrtEp = std::mem::zeroed();
        base.ort_version_supported = ORT_API_VERSION;
        base.GetName = Some(get_name);
        base.GetCapability = Some(get_capability);
        base.Compile = Some(compile);
        base.ReleaseNodeComputeInfos = Some(release_node_compute_infos);
        base.GetCompiledModelCompatibilityInfo = Some(compat_info);
        base.CreateAllocator = Some(create_allocator);
        base.CreateSyncStreamForDevice = Some(create_sync_stream);
        base.Sync = Some(sync);
        base.GetDefaultMemoryDevice = Some(default_memory_device);
        base.GetWeightlessSupport = Some(weightless_support);
        let compat = format!("{name};version={EP_VERSION};ort_api_version={ORT_API_VERSION};backend={}", backend.primary_name);
        let p = Box::new(Provider {
            base,
            name: CString::new(name).unwrap(),
            compat: CString::new(compat).unwrap(),
            logger,
            options: opts,
            backend,
            programs: Mutex::new(Vec::new()),
        });
        Ok(Box::into_raw(p) as *mut OrtEp)
    }

    /// # Safety
    /// `p` came from `create`.
    pub unsafe fn release(p: *mut OrtEp) {
        if !p.is_null() {
            drop(Box::from_raw(p as *mut Provider));
        }
    }

    unsafe fn from_ptr<'a>(p: *const OrtEp) -> &'a Provider {
        &*(p as *const Provider)
    }
}

unsafe extern "C" fn get_name(this: *const OrtEp) -> *const c_char {
    Provider::from_ptr(this).name.as_ptr()
}

struct NodeInfo {
    ptr: *const OrtNode,
    op: String,
    domain: String,
}

unsafe fn list_nodes(graph: *const OrtGraph) -> Result<Vec<NodeInfo>> {
    let mut n = 0usize;
    ort_call!(Graph_GetNumNodes(graph, &mut n))?;
    let mut nodes: Vec<*const OrtNode> = vec![std::ptr::null(); n];
    if n > 0 {
        ort_call!(Graph_GetNodes(graph, nodes.as_mut_ptr(), n))?;
    }
    let mut out = Vec::with_capacity(n);
    for ptr in nodes {
        let mut p = std::ptr::null();
        ort_call!(Node_GetOperatorType(ptr, &mut p))?;
        let op = cstr(p);
        ort_call!(Node_GetDomain(ptr, &mut p))?;
        let domain = cstr(p);
        out.push(NodeInfo { ptr, op, domain });
    }
    Ok(out)
}

unsafe extern "C" fn get_capability(
    this: *mut OrtEp,
    graph: *const OrtGraph,
    info: *mut OrtEpGraphSupportInfo,
) -> *mut OrtStatus {
    guard("GetCapability", || {
        let provider = Provider::from_ptr(this);
        let span = tracing::info_span!("capability");
        let _e = span.enter();
        let nodes = list_nodes(graph)?;
        if nodes.is_empty() {
            tracing::info!("empty graph, nothing to claim");
            return Ok(());
        }
        let mut supported: Vec<*const OrtNode> = Vec::with_capacity(nodes.len());
        let mut unsupported: HashMap<String, usize> = HashMap::new();
        let mut hist: HashMap<String, usize> = HashMap::new();
        for n in &nodes {
            *hist.entry(n.op.clone()).or_default() += 1;
            if op_supported(&n.op, &n.domain) {
                supported.push(n.ptr);
            } else {
                let key = if n.domain.is_empty() { n.op.clone() } else { format!("{}::{}", n.domain, n.op) };
                *unsupported.entry(key).or_default() += 1;
            }
        }
        let mut hist: Vec<(String, usize)> = hist.into_iter().collect();
        hist.sort_by_key(|e| std::cmp::Reverse(e.1));
        tracing::debug!(ops = ?hist, "op histogram");
        if !unsupported.is_empty() {
            let mut list: Vec<(String, usize)> = unsupported.into_iter().collect();
            list.sort_by_key(|e| std::cmp::Reverse(e.1));
            let msg = format!(
                "{} of {} nodes use ops this provider lacks: {}",
                nodes.len() - supported.len(),
                nodes.len(),
                list.iter().map(|(k, v)| format!("{k} x{v}")).collect::<Vec<_>>().join(", ")
            );
            if provider.options.partial {
                tracing::warn!(%msg, "claiming the supported nodes only (partial=1); expect copies at every boundary");
                log(provider.logger, ORT_LOGGING_LEVEL_WARNING, &format!("onnxruntime-ggml: {msg}; partial claim"));
            } else {
                tracing::warn!(%msg, "claiming nothing; the CPU provider runs this graph");
                log(
                    provider.logger,
                    ORT_LOGGING_LEVEL_WARNING,
                    &format!("onnxruntime-ggml: {msg}; claiming nothing (set partial=1 to override)"),
                );
                return Ok(());
            }
        }
        if supported.is_empty() {
            return Ok(());
        }
        let mut fusion = OrtNodeFusionOptions { ort_version_supported: ORT_API_VERSION, drop_constant_initializers: true };
        ep_call!(EpGraphSupportInfo_AddNodesToFuse(info, supported.as_ptr(), supported.len(), &mut fusion))?;
        tracing::info!(claimed = supported.len(), total = nodes.len(), "nodes claimed for fusion");
        log(
            provider.logger,
            ORT_LOGGING_LEVEL_INFO,
            &format!("onnxruntime-ggml: claimed {} of {} nodes", supported.len(), nodes.len()),
        );
        Ok(())
    })
}

unsafe extern "C" fn compile(
    this: *mut OrtEp,
    graphs: *mut *const OrtGraph,
    fused_nodes: *mut *const OrtNode,
    count: usize,
    node_compute_infos: *mut *mut OrtNodeComputeInfo,
    ep_context_nodes: *mut *mut OrtNode,
) -> *mut OrtStatus {
    guard("Compile", || {
        let provider = Provider::from_ptr(this);
        for i in 0..count {
            let graph = *graphs.add(i);
            let fused = *fused_nodes.add(i);
            let mut p = std::ptr::null();
            ort_call!(Node_GetName(fused, &mut p))?;
            let fused_name = cstr(p);
            tracing::info!(index = i, count, fused = %fused_name, "compiling subgraph");
            let ir = crate::ort::graph::import(graph)?;
            let program = Arc::new(Program::compile(&fused_name, ir, provider.backend.clone())?);
            provider.programs.lock().map_err(|_| Error::internal("programs lock poisoned"))?.push(program.clone());
            *node_compute_infos.add(i) = ComputeInfo::create(program, fused_name);
            if !ep_context_nodes.is_null() {
                *ep_context_nodes.add(i) = std::ptr::null_mut();
            }
        }
        Ok(())
    })
}

unsafe extern "C" fn release_node_compute_infos(_this: *mut OrtEp, infos: *mut *mut OrtNodeComputeInfo, n: usize) {
    for i in 0..n {
        ComputeInfo::release(*infos.add(i));
    }
    tracing::debug!(n, "compute infos released");
}

unsafe extern "C" fn compat_info(this: *mut OrtEp, _graph: *const OrtGraph) -> *const c_char {
    Provider::from_ptr(this).compat.as_ptr()
}

unsafe extern "C" fn create_allocator(
    _this: *mut OrtEp,
    _mi: *const OrtMemoryInfo,
    allocator: *mut *mut OrtAllocator,
) -> *mut OrtStatus {
    *allocator = std::ptr::null_mut();
    std::ptr::null_mut()
}

unsafe extern "C" fn create_sync_stream(
    _this: *mut OrtEp,
    _dev: *const OrtMemoryDevice,
    stream: *mut *mut OrtSyncStreamImpl,
) -> *mut OrtStatus {
    *stream = std::ptr::null_mut();
    std::ptr::null_mut()
}

unsafe extern "C" fn sync(_this: *mut OrtEp) -> *mut OrtStatus {
    std::ptr::null_mut()
}

unsafe extern "C" fn default_memory_device(_this: *const OrtEp, device: *mut *const OrtMemoryDevice) -> *mut OrtStatus {
    *device = std::ptr::null();
    std::ptr::null_mut()
}

unsafe extern "C" fn weightless_support(_this: *const OrtEp, support: *mut OrtWeightlessSupport) -> *mut OrtStatus {
    *support = OrtWeightlessSupport_NONE;
    std::ptr::null_mut()
}
