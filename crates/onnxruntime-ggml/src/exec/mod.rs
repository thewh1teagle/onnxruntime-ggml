//! The compiler and executor.
//!
//! - `backend` : ggml backends (Metal/Vulkan + CPU) and a scheduler
//! - `fold`    : logical shapes of any rank on 4-dimensional ggml tensors
//! - `ggml`    : safe helpers over ggml tensors (shape convention, views, cont)
//! - `program` : compile an `ir::Graph`: fold constants, rewrite patterns,
//!               upload weights once
//! - `runtime` : run a compiled program: walk nodes, keep shape math on the
//!               host, emit ggml ops for the rest, move data only at boundaries
//! - `ops_*`   : the ONNX -> ggml op emitters
//! - `value`   : what a runtime value is (host, staged-for-device, or device)

pub mod backend;
pub mod fold;
pub mod ggml;
pub mod ops_binary;
pub mod ops_nn;
pub mod ops_shape;
pub mod program;
pub mod runtime;
pub mod value;

pub use program::Program;
