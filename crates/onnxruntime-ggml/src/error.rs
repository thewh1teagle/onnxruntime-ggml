//! One error type for the whole provider. At the C boundary it becomes an
//! `OrtStatus` (see `ort::api::to_status`); everywhere else it is a `Result`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("onnxruntime: {0}")]
    Ort(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("shape: {0}")]
    Shape(String),
    #[error("ggml: {0}")]
    Ggml(String),
    #[error("model: {0}")]
    Model(String),
    #[error("internal: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
    pub fn shape(msg: impl Into<String>) -> Self {
        Error::Shape(msg.into())
    }
    pub fn ggml(msg: impl Into<String>) -> Self {
        Error::Ggml(msg.into())
    }
    pub fn model(msg: impl Into<String>) -> Self {
        Error::Model(msg.into())
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        Error::Internal(msg.into())
    }

    /// The onnxruntime error code this maps to.
    pub fn ort_code(&self) -> ort_ep_sys::OrtErrorCode {
        match self {
            Error::Ort(_) => ort_ep_sys::ORT_EP_FAIL,
            Error::Unsupported(_) => ort_ep_sys::ORT_NOT_IMPLEMENTED,
            Error::Shape(_) => ort_ep_sys::ORT_INVALID_ARGUMENT,
            Error::Ggml(_) => ort_ep_sys::ORT_EP_FAIL,
            Error::Model(_) => ort_ep_sys::ORT_INVALID_GRAPH,
            Error::Internal(_) => ort_ep_sys::ORT_FAIL,
        }
    }
}

/// Shorthand for the many `Err(Error::Shape(format!(...)))` sites.
#[macro_export]
macro_rules! bail_shape {
    ($($arg:tt)*) => { return Err($crate::error::Error::Shape(format!($($arg)*))) };
}
#[macro_export]
macro_rules! bail_unsupported {
    ($($arg:tt)*) => { return Err($crate::error::Error::Unsupported(format!($($arg)*))) };
}
