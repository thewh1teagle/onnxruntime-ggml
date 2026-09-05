//! Bridge sequence/control-flow values to the provider's host interpreter.
use super::*;
use crate::host::eval_control::{self, FlowValue};

impl Run<'_> {
    pub(super) fn run_control(&mut self, node: &Node) -> Result<()> {
        let started = Instant::now();
        let names: Vec<String> = node.inputs.iter().filter(|n| !n.is_empty()).cloned().chain(node.captures()).collect();
        if names.iter().any(|name| self.lookup(name).is_some_and(|v| v.is_device())) {
            self.flush(&format!("{node} control-flow inputs"))?;
        }
        let mut env = HashMap::new();
        for name in names {
            let v = if let Some(sequence) = self.sequences.get(&name) {
                FlowValue::Sequence(sequence.clone())
            } else {
                let v = self.lookup(&name).ok_or_else(|| Error::model(format!("{node}: missing capture {name}")))?;
                match v {
                    Value::Host(t) | Value::Staged(t) => FlowValue::Tensor(t),
                    Value::Device(_) => return Err(Error::internal("control input remains on device")),
                }
            };
            env.insert(name, v);
        }
        let outputs = if node.op == "Identity" {
            vec![env[&node.inputs[0]].clone()]
        } else {
            let mut streams = self.prog.random.lock().map_err(|_| Error::internal("random stream lock"))?;
            eval_control::eval(node, &env, &mut streams)?
        };
        for (name, value) in node.outputs.iter().zip(outputs) {
            if name.is_empty() {
                continue;
            }
            match value {
                FlowValue::Tensor(t) => {
                    self.values.insert(name.clone(), if t.dtype().is_float() { Value::Staged(t) } else { Value::Host(t) });
                }
                FlowValue::Sequence(s) => {
                    self.sequences.insert(name.clone(), s);
                }
            }
        }
        self.stats.host_ops += 1;
        self.stats.host_ms += started.elapsed().as_secs_f64() * 1000.;
        tracing::debug!(node = %node, ms = started.elapsed().as_secs_f64() * 1000., "control-flow node completed");
        Ok(())
    }
}
