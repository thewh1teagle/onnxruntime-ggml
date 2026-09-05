//! Graph submission and host/device transfers for a run.
use super::*;

impl Run<'_> {
    // ------------------------------------------------------------ flush

    /// Compute the graph built so far and bring every live device value back
    /// to the host (as `Staged`). Afterwards the graph is empty again.
    pub fn flush(&mut self, reason: &str) -> Result<()> {
        let live: Vec<(String, DeviceTensor)> =
            self.values.iter().filter_map(|(k, v)| v.device().map(|d| (k.clone(), d))).collect();
        if live.is_empty() && self.uploads.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        let sched = self.prog.backend.sched;
        let mut outs = Vec::with_capacity(live.len());
        unsafe {
            for (name, d) in &live {
                let d = ggml::contig(self.ctx, *d);
                g::ggml_set_output(d.t);
                g::ggml_build_forward_expand(self.graph, d.t);
                outs.push((name.clone(), d));
            }
            let n_nodes = g::ggml_graph_n_nodes(self.graph) as usize;
            self.stats.ggml_nodes += n_nodes;
            if tracing::enabled!(tracing::Level::DEBUG) {
                let mut hist: HashMap<String, usize> = HashMap::new();
                for i in 0..n_nodes as i32 {
                    let node = g::ggml_graph_node(self.graph, i);
                    let op = std::ffi::CStr::from_ptr(g::ggml_op_desc(node)).to_string_lossy().into_owned();
                    *hist.entry(op).or_default() += 1;
                }
                let mut list: Vec<(String, usize)> = hist.into_iter().collect();
                list.sort_by_key(|e| std::cmp::Reverse(e.1));
                tracing::debug!(ops = ?list, "ggml graph ops");
            }
            let single = !self.prog.galloc.0.is_null();
            if single {
                if !g::ggml_gallocr_alloc_graph(self.prog.galloc.0, self.graph) {
                    return Err(Error::ggml(format!("could not allocate a graph of {n_nodes} nodes ({reason})")));
                }
            } else {
                g::ggml_backend_sched_reset(sched);
                if !g::ggml_backend_sched_alloc_graph(sched, self.graph) {
                    return Err(Error::ggml(format!("scheduler could not allocate a graph of {n_nodes} nodes ({reason})")));
                }
            }
            let mut set = 0usize;
            for up in &self.uploads {
                if (*up.t).buffer.is_null() && (*up.t).data.is_null() {
                    tracing::trace!("upload leaf unused by any node, skipped");
                    continue;
                }
                match &up.src {
                    UploadSrc::Owned(bytes) => g::ggml_backend_tensor_set(up.t, bytes.as_ptr().cast(), 0, bytes.len()),
                    UploadSrc::Borrowed(ptr, len) => g::ggml_backend_tensor_set(up.t, (*ptr).cast(), 0, *len),
                }
                set += 1;
            }
            let t_alloc = started.elapsed();
            let status = if single {
                g::ggml_backend_graph_compute(self.prog.device, self.graph)
            } else if self.prog.backend.options.profile {
                crate::exec::profile::compute_profiled(sched, self.graph, reason)
            } else {
                g::ggml_backend_sched_graph_compute(sched, self.graph)
            };
            if status != g::ggml_status_GGML_STATUS_SUCCESS {
                return Err(Error::ggml(format!("graph compute failed with status {status} ({reason})")));
            }
            let t_compute = started.elapsed() - t_alloc;
            if !single {
                if tracing::enabled!(tracing::Level::DEBUG) {
                    let mut placement: HashMap<String, usize> = HashMap::new();
                    for i in 0..n_nodes as i32 {
                        let node = g::ggml_graph_node(self.graph, i);
                        let backend = g::ggml_backend_sched_get_tensor_backend(sched, node);
                        if !backend.is_null() {
                            let device = std::ffi::CStr::from_ptr(g::ggml_backend_name(backend)).to_string_lossy();
                            let op = std::ffi::CStr::from_ptr(g::ggml_op_desc(node)).to_string_lossy();
                            *placement.entry(format!("{device}:{op}")).or_default() += 1;
                        }
                    }
                    tracing::debug!(?placement, "ggml backend placement");
                }
                tracing::debug!(
                    splits = g::ggml_backend_sched_get_n_splits(sched),
                    copies = g::ggml_backend_sched_get_n_copies(sched),
                    "scheduler"
                );
            }
            let t_read0 = Instant::now();
            let mut read_bytes = 0usize;
            for (name, d) in &outs {
                let n = ggml::nelements(d.t);
                let mut data = vec![0f32; n];
                g::ggml_backend_tensor_get(d.t, data.as_mut_ptr().cast(), 0, n * 4);
                read_bytes += n * 4;
                let t = HostTensor::f32(d.shape(), data);
                tracing::trace!(name = %name, value = %t.brief(), "readback");
                self.values.insert(name.clone(), Value::staged_of(t));
            }
            let t_read = t_read0.elapsed();
            let t_reset0 = Instant::now();
            g::ggml_backend_sched_reset(sched);
            // Reuse the context's arena: freeing it would mean another ~12 MiB
            // allocation for the next graph, per flush.
            g::ggml_reset(self.ctx);
            self.graph = g::ggml_new_graph_custom(self.ctx, GRAPH_SIZE, false);
            if self.graph.is_null() {
                return Err(Error::ggml("ggml_new_graph_custom failed after a flush"));
            }
            self.uploads.clear();
            self.uploaded.clear();
            self.stats.flushes += 1;
            self.stats.readbacks += outs.len();
            self.stats.readback_bytes += read_bytes;
            self.stats.compute_ms += t_compute.as_secs_f64() * 1000.0;
            self.stats.build_ms += t_alloc.as_secs_f64() * 1000.0;
            tracing::debug!(
                reason,
                ggml_nodes = n_nodes,
                inputs_set = set,
                readbacks = outs.len(),
                readback = %bytes(read_bytes),
                alloc_ms = format!("{:.2}", t_alloc.as_secs_f64() * 1000.0),
                compute_ms = format!("{:.2}", t_compute.as_secs_f64() * 1000.0),
                read_ms = format!("{:.2}", t_read.as_secs_f64() * 1000.0),
                reset_ms = format!("{:.2}", t_reset0.elapsed().as_secs_f64() * 1000.0),
                "flush"
            );
        }
        Ok(())
    }

    /// Make sure the named inputs are host-readable, flushing if any is on the device.
    pub(super) fn ensure_host(&mut self, ins: &mut [Option<In>], which: &[usize], reason: &str) -> Result<()> {
        let need = which.iter().any(|&k| ins.get(k).and_then(|i| i.as_ref()).is_some_and(|i| i.v.is_device()));
        if need {
            self.flush(reason)?;
            for i in ins.iter_mut().flatten() {
                if i.v.is_device() {
                    i.v =
                        self.values.get(&i.name).cloned().ok_or_else(|| Error::internal(format!("{} lost in flush", i.name)))?;
                }
            }
        }
        Ok(())
    }
}
