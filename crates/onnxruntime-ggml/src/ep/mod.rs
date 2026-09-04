//! The onnxruntime plugin execution provider: three C function tables and
//! the Rust state behind them.
//!
//! - `factory`  : `OrtEpFactory`, one per loaded library; says which devices
//!                the provider supports (the CPU device: memory is host memory)
//! - `provider` : `OrtEp`, one per session; claims nodes and compiles them
//! - `compute`  : `OrtNodeComputeInfo`, one per compiled subgraph; runs it
//! - `options`  : provider options from session config and environment

pub mod compute;
pub mod factory;
pub mod options;
pub mod provider;
