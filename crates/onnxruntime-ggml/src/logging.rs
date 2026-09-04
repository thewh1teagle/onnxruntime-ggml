//! Tracing setup. The provider lives inside someone else's process, so the
//! subscriber is installed once, writes to stderr, and is controlled only by
//! `ORT_GGML_LOG` (a `tracing_subscriber::EnvFilter` directive, default `info`).
//!
//! Levels, so a reader knows what to expect:
//! - `info`  : one line per session-level event (factory, devices, compile, claim)
//! - `debug` : per-run summaries (op counts, flushes, bytes moved, timings)
//! - `trace` : per-node placement and shapes, per-upload and per-readback

use std::sync::Once;

use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_env("ORT_GGML_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .with_target(false)
            .with_ansi(false)
            .compact()
            .try_init();
        tracing::debug!("tracing initialised (ORT_GGML_LOG)");
    });
}

/// Human-sized byte counts for log lines.
pub fn bytes(n: usize) -> String {
    const K: f64 = 1024.0;
    let n = n as f64;
    if n < K {
        format!("{n:.0} B")
    } else if n < K * K {
        format!("{:.1} KiB", n / K)
    } else if n < K * K * K {
        format!("{:.1} MiB", n / K / K)
    } else {
        format!("{:.2} GiB", n / K / K / K)
    }
}
