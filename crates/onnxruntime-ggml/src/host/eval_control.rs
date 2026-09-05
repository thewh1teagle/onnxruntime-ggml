//! Standard ONNX sequence values and structured control flow.
//!
//! Values are reference counted: inserting a tensor into a sequence copies
//! references rather than the tensor payload. Nested graphs resolve lexical
//! captures against their parent environment.
use crate::error::{Error, Result};
use crate::host::{eval, eval_random::Streams, HostTensor};
use crate::ir::{Attr, Graph, Node};
use std::collections::HashMap;
use std::sync::Arc;

pub const OPS: &[&str] =
    &["If", "Loop", "SequenceEmpty", "SequenceAt", "SequenceInsert", "SequenceLength", "SplitToSequence", "ConcatFromSequence"];

#[derive(Clone, Debug)]
pub enum FlowValue {
    Tensor(Arc<HostTensor>),
    Sequence(Arc<Vec<Arc<HostTensor>>>),
}

impl FlowValue {
    pub fn tensor(&self) -> Result<&HostTensor> {
        match self {
            Self::Tensor(v) => Ok(v),
            Self::Sequence(_) => Err(Error::model("expected tensor, got sequence")),
        }
    }
    pub fn sequence(&self) -> Result<&Vec<Arc<HostTensor>>> {
        match self {
            Self::Sequence(v) => Ok(v),
            Self::Tensor(_) => Err(Error::model("expected sequence, got tensor")),
        }
    }
    fn of(t: HostTensor) -> Self {
        Self::Tensor(Arc::new(t))
    }
}

fn graph<'a>(n: &'a Node, name: &str) -> Result<&'a Graph> {
    match n.attrs.get(name) {
        Some(Attr::Graph(g)) => Ok(g),
        _ => Err(Error::model(format!("{n}: missing graph {name}"))),
    }
}

pub fn run_graph(
    g: &Graph,
    parent: &HashMap<String, FlowValue>,
    inputs: Vec<FlowValue>,
    streams: &mut Streams,
) -> Result<Vec<FlowValue>> {
    if inputs.len() != g.inputs.len() {
        return Err(Error::shape("control-flow graph input count"));
    }
    let mut env = parent.clone();
    env.extend(g.constants.iter().map(|(name, t)| (name.clone(), FlowValue::Tensor(t.clone()))));
    env.extend(g.inputs.iter().zip(inputs).map(|(d, v)| (d.name.clone(), v)));
    for n in &g.nodes {
        let values = if n.op == "Identity" {
            vec![env
                .get(n.input(0).ok_or_else(|| Error::model("Identity input"))?)
                .cloned()
                .ok_or_else(|| Error::model("Identity missing value"))?]
        } else if OPS.contains(&n.op.as_str()) {
            eval(n, &env, streams)?
        } else {
            let mut inputs = Vec::new();
            for name in &n.inputs {
                inputs.push(if name.is_empty() {
                    None
                } else {
                    Some(env.get(name).ok_or_else(|| Error::model(format!("{n}: missing {name}")))?.tensor()?)
                });
            }
            let outputs = if super::eval_random::OPS.contains(&n.op.as_str()) {
                streams.eval(n, &inputs)?
            } else {
                eval::eval(n, &inputs)?
            };
            outputs.into_iter().map(FlowValue::of).collect()
        };
        for (name, value) in n.outputs.iter().zip(values) {
            if !name.is_empty() {
                env.insert(name.clone(), value);
            }
        }
    }
    g.outputs
        .iter()
        .map(|d| env.get(&d.name).cloned().ok_or_else(|| Error::model(format!("missing control-flow output {}", d.name))))
        .collect()
}

pub fn eval(n: &Node, env: &HashMap<String, FlowValue>, streams: &mut Streams) -> Result<Vec<FlowValue>> {
    let input = |i| -> Result<&FlowValue> {
        let name = n.input(i).ok_or_else(|| Error::model(format!("{n}: missing input {i}")))?;
        env.get(name).ok_or_else(|| Error::model(format!("{n}: no value for {name}")))
    };
    tracing::trace!(node = %n, "control-flow enter");
    let out = match n.op.as_str() {
        "If" => {
            let cond = input(0)?.tensor()?.scalar_i64()? != 0;
            let branch = if cond { "then_branch" } else { "else_branch" };
            tracing::trace!(node = %n, branch, "control-flow branch");
            return run_graph(graph(n, branch)?, env, vec![], streams);
        }
        "Loop" => {
            let body = graph(n, "body")?;
            let max = if n.input(0).is_some() { input(0)?.tensor()?.scalar_i64()?.max(0) as u64 } else { u64::MAX };
            let mut cond = if n.input(1).is_some() { input(1)?.tensor()?.scalar_i64()? != 0 } else { true };
            let carried = n.inputs.len().saturating_sub(2);
            if body.inputs.len() != carried + 2 || body.outputs.len() < carried + 1 {
                return Err(Error::model("Loop body signature"));
            }
            let mut state: Vec<FlowValue> = (2..n.inputs.len()).map(input).map(|v| v.cloned()).collect::<Result<_>>()?;
            let mut scans = vec![Vec::<Arc<HostTensor>>::new(); body.outputs.len() - carried - 1];
            let mut iteration = 0u64;
            while iteration < max && cond {
                let args = [
                    vec![
                        FlowValue::of(HostTensor::const_i64(iteration as i64)),
                        FlowValue::of(HostTensor::bool(vec![], vec![cond])),
                    ],
                    state,
                ]
                .concat();
                let values = run_graph(body, env, args, streams)?;
                cond = values[0].tensor()?.scalar_i64()? != 0;
                state = values[1..1 + carried].to_vec();
                for (scan, value) in scans.iter_mut().zip(&values[1 + carried..]) {
                    scan.push(Arc::new(value.tensor()?.clone()));
                }
                iteration += 1;
            }
            tracing::debug!(node = %n, iteration, scans = scans.len(), "control-flow loop completed");
            for (i, scan) in scans.iter().enumerate() {
                let tensor = if scan.is_empty() {
                    let desc = &body.outputs[carried + 1 + i];
                    let mut shape = vec![0];
                    for dim in &desc.shape {
                        shape
                            .push(dim.ok_or_else(|| Error::unsupported("zero-iteration Loop with symbolic scan dimensions"))?
                                as usize);
                    }
                    HostTensor::zeros(desc.dtype, shape)
                } else {
                    concatenate(scan, 0, true)?
                };
                state.push(FlowValue::of(tensor));
            }
            return Ok(state);
        }
        "SequenceEmpty" => FlowValue::Sequence(Arc::new(vec![])),
        "SequenceLength" => FlowValue::of(HostTensor::const_i64(input(0)?.sequence()?.len() as i64)),
        "SequenceAt" => {
            let sequence = input(0)?.sequence()?;
            let mut index = input(1)?.tensor()?.scalar_i64()?;
            if index < 0 {
                index += sequence.len() as i64;
            }
            if index < 0 || index as usize >= sequence.len() {
                return Err(Error::shape("SequenceAt index"));
            }
            FlowValue::Tensor(sequence[index as usize].clone())
        }
        "SequenceInsert" => {
            let mut sequence = input(0)?.sequence()?.clone();
            let value = input(1)?.tensor()?;
            if sequence.first().is_some_and(|t| t.dtype() != value.dtype()) {
                return Err(Error::model("SequenceInsert element type"));
            }
            let mut index = if n.input(2).is_some() { input(2)?.tensor()?.scalar_i64()? } else { sequence.len() as i64 };
            if index < 0 {
                index += sequence.len() as i64;
            }
            if index < 0 || index as usize > sequence.len() {
                return Err(Error::shape("SequenceInsert index"));
            }
            sequence.insert(index as usize, Arc::new(value.clone()));
            FlowValue::Sequence(Arc::new(sequence))
        }
        "SplitToSequence" => {
            let x = input(0)?.tensor()?;
            let axis = eval::norm_axis(n.attr_i("axis", 0), x.rank())?;
            let size = x.shape[axis];
            let supplied = n.input(1).is_some();
            let split = if supplied { input(1)?.tensor()?.as_i64().into_owned() } else { vec![1] };
            let lengths: Vec<usize> = if !supplied || input(1)?.tensor()?.rank() == 0 {
                let block = *split.first().ok_or_else(|| Error::shape("SplitToSequence split"))?;
                if block <= 0 {
                    return Err(Error::shape("SplitToSequence nonpositive split"));
                }
                (0..size).step_by(block as usize).map(|start| (size - start).min(block as usize)).collect()
            } else {
                if split.iter().any(|&v| v < 0) || split.iter().sum::<i64>() != size as i64 {
                    return Err(Error::shape("SplitToSequence sizes"));
                }
                split.iter().map(|&v| v as usize).collect()
            };
            let keepdims = supplied || n.attr_i("keepdims", 1) != 0;
            let mut pieces = Vec::new();
            let mut offset = 0;
            for len in lengths {
                let mut node = Node::new("Slice", "sequence_slice", &[], &["out"]);
                node.set_attr_i("axis", axis as i64);
                let starts = HostTensor::i64(vec![1], vec![offset as i64]);
                let ends = HostTensor::i64(vec![1], vec![(offset + len) as i64]);
                let axes = HostTensor::i64(vec![1], vec![axis as i64]);
                let mut piece = eval::eval(&node, &[Some(x), Some(&starts), Some(&ends), Some(&axes)])?.remove(0);
                if !keepdims {
                    piece.shape.remove(axis);
                }
                pieces.push(Arc::new(piece));
                offset += len;
            }
            FlowValue::Sequence(Arc::new(pieces))
        }
        "ConcatFromSequence" => {
            let sequence = input(0)?.sequence()?;
            FlowValue::of(concatenate(sequence, n.attr_i("axis", 0), n.attr_i("new_axis", 0) != 0)?)
        }
        _ => return Err(Error::unsupported(&n.op)),
    };
    Ok(vec![out])
}

fn concatenate(sequence: &[Arc<HostTensor>], axis: i64, new_axis: bool) -> Result<HostTensor> {
    let first = sequence.first().ok_or_else(|| Error::shape("cannot concatenate an empty sequence"))?;
    let axis = eval::norm_axis(axis, first.rank() + usize::from(new_axis))?;
    let values = if new_axis {
        sequence
            .iter()
            .map(|v| {
                let mut t = (**v).clone();
                t.shape.insert(axis, 1);
                t
            })
            .collect::<Vec<_>>()
    } else {
        sequence.iter().map(|v| (**v).clone()).collect()
    };
    let mut n = Node::new("Concat", "sequence_concat", &[], &["out"]);
    n.set_attr_i("axis", axis as i64);
    eval::eval(&n, &values.iter().map(Some).collect::<Vec<_>>()).map(|mut v| v.remove(0))
}
