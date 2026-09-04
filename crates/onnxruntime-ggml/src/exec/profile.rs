//! Per-kernel timing through `ggml_backend_sched_set_eval_callback`.
//!
//! With a callback installed the scheduler runs the graph one node at a time
//! and synchronises around each, so the numbers are exact per kernel but the
//! whole graph is slower than in production. `profile=1` (or
//! `ORT_GGML_PROFILE=1`) turns it on; the summary is logged at `info` per
//! flush, aggregated by op and by op+shape, sorted by time.

use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::time::Instant;

use ggml_sys as g;

struct State {
    started: Option<(Instant, String)>,
    by_op: HashMap<String, (f64, usize)>,
    by_shape: HashMap<String, (f64, usize)>,
}

unsafe extern "C" fn callback(t: *mut g::ggml_tensor, ask: bool, user: *mut c_void) -> bool {
    let st = &mut *(user as *mut State);
    if ask {
        // asked whether we want this node observed: yes for every node; timing starts now
        let op = CStr::from_ptr(g::ggml_op_desc(t)).to_string_lossy().into_owned();
        let ne = (*t).ne;
        let key = format!("{op} ne=[{},{},{},{}]", ne[0], ne[1], ne[2], ne[3]);
        st.started = Some((Instant::now(), key));
        return true;
    }
    if let Some((t0, key)) = st.started.take() {
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let op = key.split(' ').next().unwrap_or("?").to_owned();
        let e = st.by_op.entry(op).or_insert((0.0, 0));
        e.0 += ms;
        e.1 += 1;
        let e = st.by_shape.entry(key).or_insert((0.0, 0));
        e.0 += ms;
        e.1 += 1;
    }
    true
}

/// # Safety
/// `sched` and `graph` are valid and allocated.
pub unsafe fn compute_profiled(sched: g::ggml_backend_sched_t, graph: *mut g::ggml_cgraph, reason: &str) -> g::ggml_status {
    let mut st = State { started: None, by_op: HashMap::new(), by_shape: HashMap::new() };
    g::ggml_backend_sched_set_eval_callback(sched, Some(callback), &mut st as *mut State as *mut c_void);
    let t0 = Instant::now();
    let status = g::ggml_backend_sched_graph_compute(sched, graph);
    let total = t0.elapsed().as_secs_f64() * 1000.0;
    g::ggml_backend_sched_set_eval_callback(sched, None, std::ptr::null_mut());
    let mut ops: Vec<(String, f64, usize)> = st.by_op.into_iter().map(|(k, (ms, n))| (k, ms, n)).collect();
    ops.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let ops_fmt: Vec<String> = ops.iter().map(|(k, ms, n)| format!("{k} {ms:.2}ms x{n}")).collect();
    let mut shapes: Vec<(String, f64, usize)> = st.by_shape.into_iter().map(|(k, (ms, n))| (k, ms, n)).collect();
    shapes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let shapes_fmt: Vec<String> = shapes.iter().take(12).map(|(k, ms, n)| format!("{k} {ms:.2}ms x{n}")).collect();
    tracing::info!(reason, total_ms = format!("{total:.2}"), by_op = ?ops_fmt, "profile");
    tracing::info!(top_shapes = ?shapes_fmt, "profile");
    status
}
