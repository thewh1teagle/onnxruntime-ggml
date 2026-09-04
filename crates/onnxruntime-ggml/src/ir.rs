//! The provider's own view of an ONNX graph: what `ort::graph` imports from the
//! `OrtGraph`, what `exec::program` folds and rewrites, and what
//! `exec::runtime` walks at run time.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::host::tensor::HostTensor;

/// Tensor element types the provider understands. Everything else is rejected
/// at capability time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    F64,
    I64,
    I32,
    I8,
    U8,
    Bool,
}

impl DType {
    pub fn from_onnx(code: i32) -> Result<DType> {
        use ort_ep_sys::*;
        // bindgen makes C enums i32 on MSVC and u32 elsewhere; compare in the enum's own type.
        Ok(match code as ONNXTensorElementDataType {
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => DType::F32,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => DType::F16,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => DType::F64,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => DType::I64,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => DType::I32,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => DType::I8,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => DType::U8,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => DType::Bool,
            other => return Err(Error::unsupported(format!("tensor element type {other}"))),
        })
    }

    pub fn to_onnx(self) -> ort_ep_sys::ONNXTensorElementDataType {
        use ort_ep_sys::*;
        match self {
            DType::F32 => ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            DType::F16 => ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            DType::F64 => ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            DType::I64 => ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            DType::I32 => ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            DType::I8 => ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
            DType::U8 => ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            DType::Bool => ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        }
    }

    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
            DType::F64 | DType::I64 => 8,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::F64)
    }

    pub fn is_int(self) -> bool {
        matches!(self, DType::I64 | DType::I32 | DType::I8 | DType::U8)
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::F64 => "f64",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::I8 => "i8",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, Debug)]
pub enum Attr {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    Str(String),
    Strs(Vec<String>),
    Tensor(HostTensor),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub op: String,
    pub domain: String,
    /// Empty string marks an omitted optional input.
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, Attr>,
}

impl Node {
    pub fn new(op: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            name: name.to_owned(),
            op: op.to_owned(),
            domain: String::new(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attrs: HashMap::new(),
        }
    }

    pub fn input(&self, i: usize) -> Option<&str> {
        self.inputs.get(i).map(|s| s.as_str()).filter(|s| !s.is_empty())
    }

    pub fn attr_i(&self, name: &str, default: i64) -> i64 {
        match self.attrs.get(name) {
            Some(Attr::Int(v)) => *v,
            Some(Attr::Ints(v)) if v.len() == 1 => v[0],
            _ => default,
        }
    }

    pub fn attr_ints(&self, name: &str) -> Option<Vec<i64>> {
        match self.attrs.get(name) {
            Some(Attr::Ints(v)) => Some(v.clone()),
            Some(Attr::Int(v)) => Some(vec![*v]),
            _ => None,
        }
    }

    pub fn attr_f(&self, name: &str, default: f32) -> f32 {
        match self.attrs.get(name) {
            Some(Attr::Float(v)) => *v,
            Some(Attr::Int(v)) => *v as f32,
            _ => default,
        }
    }

    pub fn attr_str(&self, name: &str) -> Option<&str> {
        match self.attrs.get(name) {
            Some(Attr::Str(v)) => Some(v.as_str()),
            _ => None,
        }
    }

    pub fn attr_tensor(&self, name: &str) -> Option<&HostTensor> {
        match self.attrs.get(name) {
            Some(Attr::Tensor(t)) => Some(t),
            _ => None,
        }
    }

    pub fn set_attr_i(&mut self, name: &str, v: i64) {
        self.attrs.insert(name.to_owned(), Attr::Int(v));
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.op, self.name)
    }
}

/// A graph input or output: name plus whatever the model declared about it.
/// `None` dims are symbolic.
#[derive(Clone, Debug)]
pub struct ValueDesc {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<Option<i64>>,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub name: String,
    pub inputs: Vec<ValueDesc>,
    pub outputs: Vec<ValueDesc>,
    /// Initializers, plus anything constant folding produced.
    pub constants: HashMap<String, HostTensor>,
    /// Topologically ordered.
    pub nodes: Vec<Node>,
}

impl Graph {
    /// Which node produces each value.
    pub fn producers(&self) -> HashMap<&str, usize> {
        let mut map = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            for out in &node.outputs {
                if !out.is_empty() {
                    map.insert(out.as_str(), i);
                }
            }
        }
        map
    }

    /// How many nodes (plus graph outputs) read each value.
    pub fn consumer_counts(&self) -> HashMap<&str, usize> {
        let mut map: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            for inp in &node.inputs {
                if !inp.is_empty() {
                    *map.entry(inp.as_str()).or_default() += 1;
                }
            }
        }
        for out in &self.outputs {
            *map.entry(out.name.as_str()).or_default() += 1;
        }
        map
    }

    /// Index of the last node that reads each value; graph outputs get `usize::MAX`.
    pub fn last_use(&self) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            for inp in &node.inputs {
                if !inp.is_empty() {
                    map.insert(inp.clone(), i);
                }
            }
        }
        for out in &self.outputs {
            map.insert(out.name.clone(), usize::MAX);
        }
        map
    }

    pub fn op_histogram(&self) -> Vec<(String, usize)> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            *counts.entry(node.op.as_str()).or_default() += 1;
        }
        let mut list: Vec<(String, usize)> = counts.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
        list.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        list
    }
}
