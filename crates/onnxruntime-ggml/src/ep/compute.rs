//! `OrtNodeComputeInfo`: the callable onnxruntime invokes for a fused node.
//! Reads the node's inputs from the kernel context, runs the program, writes
//! the outputs back.

use std::os::raw::c_void;
use std::sync::Arc;
use std::time::Instant;

use ort_ep_sys::*;

use crate::error::{Error, Result};
use crate::exec::Program;
use crate::ort::api::guard;
use crate::ort::kernel;

#[repr(C)]
pub struct ComputeInfo {
    base: OrtNodeComputeInfo,
    program: Arc<Program>,
    name: String,
    runs: std::sync::atomic::AtomicU64,
}

impl ComputeInfo {
    pub fn create(program: Arc<Program>, name: String) -> *mut OrtNodeComputeInfo {
        let mut base: OrtNodeComputeInfo = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.CreateState = Some(create_state);
        base.Compute = Some(compute);
        base.ReleaseState = Some(release_state);
        Box::into_raw(Box::new(ComputeInfo { base, program, name, runs: Default::default() })) as *mut OrtNodeComputeInfo
    }

    /// # Safety
    /// `p` came from `create`.
    pub unsafe fn release(p: *mut OrtNodeComputeInfo) {
        if !p.is_null() {
            drop(Box::from_raw(p as *mut ComputeInfo));
        }
    }

    unsafe fn from_ptr<'a>(p: *const OrtNodeComputeInfo) -> &'a ComputeInfo {
        &*(p as *const ComputeInfo)
    }
}

unsafe extern "C" fn create_state(
    this: *mut OrtNodeComputeInfo,
    _ctx: *mut OrtNodeComputeContext,
    state: *mut *mut c_void,
) -> *mut OrtStatus {
    let info = ComputeInfo::from_ptr(this);
    tracing::debug!(node = %info.name, "compute state created");
    *state = this as *mut c_void;
    std::ptr::null_mut()
}

unsafe extern "C" fn compute(
    this: *mut OrtNodeComputeInfo,
    _state: *mut c_void,
    kernel_ctx: *mut OrtKernelContext,
) -> *mut OrtStatus {
    guard("Compute", || {
        let info = ComputeInfo::from_ptr(this);
        let run_no = info.runs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let span = tracing::debug_span!("compute", node = %info.name, run = run_no);
        let _e = span.enter();
        let started = Instant::now();
        let inputs = kernel::read_inputs(kernel_ctx)?;
        let t_in = started.elapsed();
        let outputs = info.program.run(inputs)?;
        let t_run = started.elapsed() - t_in;
        let n_out = kernel::output_count(kernel_ctx)?;
        if n_out != outputs.len() {
            return Err(Error::internal(format!("program produced {} outputs, node has {n_out}", outputs.len())));
        }
        for (i, (t, desc)) in outputs.iter().zip(info.program.graph.outputs.iter()).enumerate() {
            kernel::write_output(kernel_ctx, i, t, desc.dtype)?;
        }
        let total = started.elapsed();
        tracing::debug!(
            read_ms = format!("{:.2}", t_in.as_secs_f64() * 1000.0),
            run_ms = format!("{:.2}", t_run.as_secs_f64() * 1000.0),
            write_ms = format!("{:.2}", (total - t_in - t_run).as_secs_f64() * 1000.0),
            total_ms = format!("{:.2}", total.as_secs_f64() * 1000.0),
            "compute done"
        );
        Ok(())
    })
}

unsafe extern "C" fn release_state(_this: *mut OrtNodeComputeInfo, _state: *mut c_void) {}

#[allow(dead_code)]
fn _result_used() -> Result<()> {
    Ok(())
}
