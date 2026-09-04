//! Unsafe, generated bindings for the onnxruntime C API (core + plugin EP API).
//!
//! `ORT_API_VERSION` is the header version these bindings were generated from;
//! the factory reports it in `ort_version_supported`.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, improper_ctypes, clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
