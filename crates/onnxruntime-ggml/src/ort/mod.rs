//! Thin safe wrappers over the onnxruntime C API table.
//!
//! - `api`    : the global `OrtApi` / `OrtEpApi` pointers, status conversion, logging
//! - `graph`  : `OrtGraph` -> `ir::Graph`
//! - `kernel` : reading inputs from and writing outputs to an `OrtKernelContext`

pub mod api;
pub mod graph;
pub mod kernel;
