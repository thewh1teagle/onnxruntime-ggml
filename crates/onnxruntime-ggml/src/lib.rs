//! onnxruntime-ggml: an onnxruntime plugin execution provider that runs ONNX
//! graphs on ggml's Metal, Vulkan and CPU kernels.
//!
//! onnxruntime loads this shared library, calls [`CreateEpFactories`], and from
//! then on talks to the provider only through the C function tables in
//! `ep::factory` and `ep::provider`. Everything above those tables is ordinary
//! Rust:
//!
//! - `ort`   : thin safe wrappers over the onnxruntime C API table
//! - `ir`    : the graph the provider compiles (nodes, attributes, constants)
//! - `host`  : an interpreter for ONNX ops on host tensors (shape math, folding)
//! - `exec`  : the compiler (fold, rewrite, upload weights) and the per-run
//!             executor that emits ggml graphs and moves data across
//!
//! Every stage traces with `tracing`; set `ORT_GGML_LOG=debug` (or `trace`)
//! to follow a session from capability query to the last readback.

pub mod ep;
pub mod error;
pub mod exec;
pub mod host;
pub mod ir;
pub mod logging;
pub mod ort;

use std::ffi::CStr;
use std::os::raw::c_char;

use ort_ep_sys::{OrtApiBase, OrtEpFactory, OrtLogger, OrtStatus};

/// The provider's registration name, as passed to
/// `register_execution_provider_library` by default.
pub const EP_NAME: &str = "ggml";
pub const EP_VENDOR: &str = "onnxruntime-ggml";
pub const EP_VENDOR_ID: u32 = 0x66_67; // "gg"
pub const EP_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Entry point onnxruntime looks up after `dlopen`.
///
/// # Safety
/// Called by onnxruntime with valid pointers; `factories` has room for `max_factories`.
#[no_mangle]
pub unsafe extern "C" fn CreateEpFactories(
    registration_name: *const c_char,
    ort_api_base: *const OrtApiBase,
    default_logger: *const OrtLogger,
    factories: *mut *mut OrtEpFactory,
    max_factories: usize,
    num_factories: *mut usize,
) -> *mut OrtStatus {
    logging::init();
    let name = if registration_name.is_null() {
        EP_NAME.to_owned()
    } else {
        CStr::from_ptr(registration_name).to_string_lossy().into_owned()
    };
    tracing::info!(
        registration_name = %name,
        version = EP_VERSION,
        ort_api_version = ort_ep_sys::ORT_API_VERSION,
        "CreateEpFactories"
    );
    if let Err(err) = ort::api::init(ort_api_base) {
        tracing::error!(%err, "onnxruntime API table unavailable");
        // No API table means no CreateStatus either; a null return would tell
        // onnxruntime we succeeded, so report failure through num_factories.
        if !num_factories.is_null() {
            *num_factories = 0;
        }
        return std::ptr::null_mut();
    }
    if max_factories < 1 {
        return ort::api::status(ort_ep_sys::ORT_INVALID_ARGUMENT, "need room for at least one factory");
    }
    let factory = ep::factory::Factory::create(name, default_logger);
    *factories = factory;
    *num_factories = 1;
    std::ptr::null_mut()
}

/// Counterpart of [`CreateEpFactories`].
///
/// # Safety
/// `factory` must come from `CreateEpFactories`.
#[no_mangle]
pub unsafe extern "C" fn ReleaseEpFactory(factory: *mut OrtEpFactory) -> *mut OrtStatus {
    tracing::info!("ReleaseEpFactory");
    ep::factory::Factory::release(factory);
    std::ptr::null_mut()
}
