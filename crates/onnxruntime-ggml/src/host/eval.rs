//! Dispatch for the host interpreter. `eval` runs one node on host tensors and
//! returns its outputs; `supported` says whether an op has a host
//! implementation at all (the runtime asks before routing).

use crate::error::{Error, Result};
use crate::host::tensor::HostTensor;
use crate::ir::Node;

use super::{eval_math, eval_nn, eval_shape};

/// Ops with a host implementation.
pub const HOST_OPS: &[&str] = &[
    "Shape",
    "Reshape",
    "Unsqueeze",
    "Squeeze",
    "Transpose",
    "Concat",
    "Slice",
    "Gather",
    "Split",
    "Range",
    "ConstantOfShape",
    "Expand",
    "Cast",
    "Where",
    "Constant",
    "Identity",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Neg",
    "Abs",
    "Sqrt",
    "Exp",
    "Log",
    "Sin",
    "Cos",
    "Tanh",
    "Sigmoid",
    "Elu",
    "Relu",
    "Erf",
    "Reciprocal",
    "Floor",
    "Ceil",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Equal",
    "And",
    "Or",
    "Not",
    "ReduceMean",
    "ReduceSum",
    "ReduceProd",
    "ReduceMax",
    "ReduceMin",
    "Softmax",
    "MatMul",
    "Gemm",
    "LayerNormalization",
    "Conv",
    "ConvTranspose",
    "FusedAttention",
    "GeluErf",
    "Clip",
    "Max",
    "Min",
];

pub fn supported(op: &str) -> bool {
    HOST_OPS.contains(&op)
}

/// Run `node` on host inputs (`None` for omitted optional inputs).
pub fn eval(node: &Node, inputs: &[Option<&HostTensor>]) -> Result<Vec<HostTensor>> {
    let op = node.op.as_str();
    let outs = match op {
        "Shape" | "Reshape" | "Unsqueeze" | "Squeeze" | "Transpose" | "Concat" | "Slice" | "Gather" | "Split" | "Range"
        | "ConstantOfShape" | "Expand" | "Cast" | "Where" | "Constant" | "Identity" => eval_shape::eval(node, inputs)?,
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Neg" | "Abs" | "Sqrt" | "Exp" | "Log" | "Sin" | "Cos" | "Tanh" | "Sigmoid"
        | "Elu" | "Relu" | "Erf" | "Reciprocal" | "Floor" | "Ceil" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual"
        | "Equal" | "And" | "Or" | "Not" | "ReduceMean" | "ReduceSum" | "ReduceProd" | "ReduceMax" | "ReduceMin" | "Softmax"
        | "GeluErf" | "Clip" | "Max" | "Min" => eval_math::eval(node, inputs)?,
        "MatMul" | "Gemm" | "LayerNormalization" | "Conv" | "ConvTranspose" | "FusedAttention" => eval_nn::eval(node, inputs)?,
        other => return Err(Error::unsupported(format!("host op {other}"))),
    };
    tracing::trace!(node = %node, outputs = ?outs.iter().map(|t| t.brief()).collect::<Vec<_>>(), "host eval");
    Ok(outs)
}

/// The input at `i`, or an error naming the node.
pub fn need<'a>(node: &Node, inputs: &[Option<&'a HostTensor>], i: usize) -> Result<&'a HostTensor> {
    inputs.get(i).copied().flatten().ok_or_else(|| Error::model(format!("{node}: missing input {i}")))
}

/// Normalise a possibly negative axis against `rank`.
pub fn norm_axis(axis: i64, rank: usize) -> Result<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r.max(1) {
        return Err(Error::shape(format!("axis {axis} out of range for rank {rank}")));
    }
    Ok(a as usize)
}
