//! Provider options. Precedence, lowest to highest: defaults, environment
//! (`ORT_GGML_*`), session config entries `ep.<name>.<key>` (what
//! `add_provider_for_devices(devices, {key: value})` writes).
//!
//! Keys: `device` (auto|gpu|cpu), `threads` (int), `partial` (0|1), `dump` (0|1).

use ort_ep_sys::OrtSessionOptions;

use crate::exec::backend::Options;
use crate::ort::api::api;

pub const KEYS: &[&str] = &["device", "threads", "partial", "dump"];

/// # Safety
/// `session_options` is the pointer onnxruntime passed to `CreateEp` (valid for that call only).
pub unsafe fn from_session(ep_name: &str, session_options: *const OrtSessionOptions) -> Options {
    let mut opts = Options::from_env();
    if session_options.is_null() {
        return opts;
    }
    let (Some(has), Some(get)) = (api().HasSessionConfigEntry, api().GetSessionConfigEntry) else { return opts };
    for key in KEYS {
        let full = format!("ep.{ep_name}.{key}");
        let c = std::ffi::CString::new(full.clone()).unwrap();
        let mut present = 0i32;
        let st = has(session_options, c.as_ptr(), &mut present);
        if !st.is_null() {
            if let Some(release) = api().ReleaseStatus {
                release(st);
            }
            continue;
        }
        if present == 0 {
            continue;
        }
        let mut size = 0usize;
        let st = get(session_options, c.as_ptr(), std::ptr::null_mut(), &mut size);
        if !st.is_null() {
            if let Some(release) = api().ReleaseStatus {
                release(st);
            }
        }
        if size == 0 {
            continue;
        }
        let mut buf = vec![0u8; size];
        let st = get(session_options, c.as_ptr(), buf.as_mut_ptr().cast(), &mut size);
        if !st.is_null() {
            if let Some(release) = api().ReleaseStatus {
                release(st);
            }
            continue;
        }
        let value = String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string();
        tracing::info!(key = %full, %value, "session option");
        opts.apply(key, Some(&value));
    }
    tracing::info!(?opts, "provider options");
    opts
}
