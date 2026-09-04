//! Unsafe, generated bindings for the pinned ggml runtime.
//!
//! Safe crates wrap this API rather than exposing its pointers directly.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, improper_ctypes, clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(target_arch = "x86_64")]
mod cpu_variant;
