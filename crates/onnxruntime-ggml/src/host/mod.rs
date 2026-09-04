//! Host-side tensors and an interpreter for ONNX ops on them.
//!
//! This is where shape arithmetic, index math and constant folding happen. It
//! also carries float tensors that are too awkward for ggml (rank above 4, or
//! ops ggml lacks), at host speed. Every op is written for clarity over speed:
//! the hot path of a model runs on ggml, this runs on scalars and caches.

pub mod broadcast;
pub mod eval;
pub mod eval_math;
pub mod eval_nn;
pub mod eval_shape;
pub mod tensor;

pub use tensor::{Data, HostTensor};
