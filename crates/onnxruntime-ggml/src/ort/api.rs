//! The onnxruntime API table, captured once in `CreateEpFactories`, and the
//! helpers every other module uses to call into it and to turn a Rust
//! `Error` into an `OrtStatus`.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::OnceLock;

use ort_ep_sys::{OrtApi, OrtApiBase, OrtEpApi, OrtErrorCode, OrtLogger, OrtLoggingLevel, OrtStatus};

use crate::error::{Error, Result};

struct Table {
    api: *const OrtApi,
    ep: *const OrtEpApi,
}

unsafe impl Send for Table {}
unsafe impl Sync for Table {}

static TABLE: OnceLock<Table> = OnceLock::new();

/// Capture the API for `ORT_API_VERSION`. Safe to call more than once.
///
/// # Safety
/// `base` must be the pointer onnxruntime passed to `CreateEpFactories`.
pub unsafe fn init(base: *const OrtApiBase) -> Result<()> {
    if base.is_null() {
        return Err(Error::Ort("null OrtApiBase".into()));
    }
    let get_api = (*base).GetApi.ok_or_else(|| Error::Ort("OrtApiBase::GetApi missing".into()))?;
    let api = get_api(ort_ep_sys::ORT_API_VERSION);
    if api.is_null() {
        let version = (*base)
            .GetVersionString
            .map(|f| CStr::from_ptr(f()).to_string_lossy().into_owned())
            .unwrap_or_default();
        return Err(Error::Ort(format!(
            "onnxruntime {version} does not provide API version {}; this provider needs onnxruntime >= 1.29",
            ort_ep_sys::ORT_API_VERSION
        )));
    }
    let get_ep = (*api).GetEpApi.ok_or_else(|| Error::Ort("OrtApi::GetEpApi missing".into()))?;
    let ep = get_ep();
    if ep.is_null() {
        return Err(Error::Ort("OrtEpApi unavailable".into()));
    }
    let _ = TABLE.set(Table { api, ep });
    if let Some(f) = (*base).GetVersionString {
        tracing::info!(onnxruntime = %CStr::from_ptr(f()).to_string_lossy(), "api table captured");
    }
    Ok(())
}

pub fn api() -> &'static OrtApi {
    unsafe { &*TABLE.get().expect("onnxruntime API not initialised").api }
}

pub fn ep_api() -> &'static OrtEpApi {
    unsafe { &*TABLE.get().expect("onnxruntime EP API not initialised").ep }
}

/// Call a status-returning `OrtApi` function and convert the result.
#[macro_export]
macro_rules! ort_call {
    ($name:ident ( $($arg:expr),* $(,)? )) => {{
        let f = $crate::ort::api::api()
            .$name
            .ok_or_else(|| $crate::error::Error::Ort(format!("OrtApi::{} is not available", stringify!($name))))?;
        $crate::ort::api::check(f($($arg),*), stringify!($name))
    }};
}

/// Same for the `OrtEpApi` table.
#[macro_export]
macro_rules! ep_call {
    ($name:ident ( $($arg:expr),* $(,)? )) => {{
        let f = $crate::ort::api::ep_api()
            .$name
            .ok_or_else(|| $crate::error::Error::Ort(format!("OrtEpApi::{} is not available", stringify!($name))))?;
        $crate::ort::api::check(f($($arg),*), stringify!($name))
    }};
}

/// Turn an `OrtStatus` into a `Result`, releasing the status.
///
/// # Safety
/// `status` is null or a status onnxruntime returned.
pub unsafe fn check(status: *mut OrtStatus, what: &str) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }
    let api = api();
    let msg = api
        .GetErrorMessage
        .map(|f| CStr::from_ptr(f(status)).to_string_lossy().into_owned())
        .unwrap_or_default();
    let code = api.GetErrorCode.map(|f| f(status) as i32).unwrap_or(-1);
    if let Some(release) = api.ReleaseStatus {
        release(status);
    }
    Err(Error::Ort(format!("{what}: {msg} (code {code})")))
}

/// Build an `OrtStatus` for onnxruntime.
///
/// # Safety
/// The API must be initialised.
pub unsafe fn status(code: OrtErrorCode, msg: &str) -> *mut OrtStatus {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    match api().CreateStatus {
        Some(f) => f(code, c.as_ptr()),
        None => std::ptr::null_mut(),
    }
}

pub fn to_status(err: &Error) -> *mut OrtStatus {
    tracing::error!(%err, "returning error status to onnxruntime");
    unsafe { status(err.ort_code(), &err.to_string()) }
}

/// Run a closure at the C boundary and hand any error to onnxruntime.
pub fn guard<F: FnOnce() -> Result<()>>(what: &str, f: F) -> *mut OrtStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(e)) => to_status(&e),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into());
            let err = Error::Internal(format!("{what} panicked: {msg}"));
            to_status(&err)
        }
    }
}

/// # Safety
/// `p` is null or a NUL-terminated string.
pub unsafe fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// Send a line to the session logger too, so it shows up next to
/// onnxruntime's own messages when a user enabled verbose logging.
///
/// # Safety
/// `logger` is null or a valid `OrtLogger`.
pub unsafe fn log(logger: *const OrtLogger, level: OrtLoggingLevel, msg: &str) {
    if logger.is_null() {
        return;
    }
    let Some(f) = api().Logger_LogMessage else { return };
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    let st = f(logger, level, c.as_ptr(), c"onnxruntime-ggml".as_ptr(), 0, c"ep".as_ptr());
    if !st.is_null() {
        if let Some(release) = api().ReleaseStatus {
            release(st);
        }
    }
}
