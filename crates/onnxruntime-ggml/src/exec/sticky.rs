//! Graph inputs that stay resident on the device between runs.
//!
//! A decode step of whisper hands the provider the encoder cross-attention KV
//! caches as ordinary graph inputs: 61 MiB of f32 that is bit-identical at
//! every step. Uploading them per run costs more than the step itself.
//!
//! This module keeps one device tensor per graph input, in a buffer owned by
//! the `Program` (not by the per-run graph), and re-uses it when the host bytes
//! look unchanged. "Look unchanged" is a *fingerprint*: the byte length, the
//! shape, and a hash of the first and last 64 elements plus a stride of samples
//! across the tensor. That is not a full comparison — a change that touches
//! none of the sampled elements would be missed — but hashing every byte would
//! cost what the upload costs. Set the `sticky=0` option to switch the whole
//! mechanism off if a model needs the guarantee.
//!
//! Only float graph inputs above `MIN_BYTES` are candidates: small inputs are
//! not worth a fingerprint, and non-inputs (activations) differ every run.

use std::collections::{HashMap, HashSet};

use ggml_sys as g;

use crate::error::{Error, Result};
use crate::exec::backend::Backend;
use crate::exec::ggml::{self, Ctx};
use crate::exec::value::DeviceTensor;
use crate::host::tensor::HostTensor;
use crate::ir::DType;

/// Below this an upload is cheap enough that residency is not worth it.
pub const MIN_BYTES: usize = 256 * 1024;

/// How many elements the fingerprint samples besides the ends.
const SAMPLES: usize = 512;
const EDGE: usize = 64;

#[derive(Debug, Default, Clone)]
pub struct StickyStats {
    pub hits: usize,
    pub misses: usize,
    /// Bytes not uploaded because a resident tensor was re-used.
    pub bytes_saved: usize,
}

struct Entry {
    ctx: Ctx,
    buffer: g::ggml_backend_buffer_t,
    d: DeviceTensor,
    fp: u64,
    nbytes: usize,
}

impl Drop for Entry {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                g::ggml_backend_buffer_free(self.buffer);
            }
            if !self.ctx.is_null() {
                g::ggml_free(self.ctx);
            }
        }
    }
}

/// The resident graph inputs of one program.
#[derive(Default)]
pub struct Sticky {
    entries: HashMap<String, Entry>,
    /// Inputs whose content changed at least once: residency only costs a
    /// device buffer allocation for them, so they go back to per-run uploads.
    volatile: HashSet<String>,
    pub stats: StickyStats,
}

unsafe impl Send for Sticky {}
unsafe impl Sync for Sticky {}

impl Sticky {
    /// Is this host tensor worth keeping resident?
    pub fn eligible(t: &HostTensor) -> bool {
        t.dtype() == DType::F32 && t.rank() > 0 && t.rank() <= ggml::MAX_RANK && t.numel() * 4 >= MIN_BYTES
    }

    /// The resident tensor for `name`, or `None` when this input has proved to
    /// change from run to run and is better uploaded into the run's own graph.
    pub fn get(&mut self, backend: &Backend, name: &str, t: &HostTensor) -> Result<Option<DeviceTensor>> {
        if self.volatile.contains(name) {
            return Ok(None);
        }
        let data = t.as_f32();
        let fp = fingerprint(&data);
        let nbytes = data.len() * 4;
        if let Some(e) = self.entries.get(name) {
            if e.nbytes == nbytes && e.d.shape() == t.shape && e.fp == fp {
                self.stats.hits += 1;
                self.stats.bytes_saved += nbytes;
                tracing::trace!(name, bytes = nbytes, "sticky hit");
                return Ok(Some(e.d));
            }
            // It changed once; it will change again. Give the buffer back and
            // let the ordinary per-graph upload path have it from now on.
            self.entries.remove(name);
            self.volatile.insert(name.to_owned());
            self.stats.misses += 1;
            tracing::debug!(name, bytes = nbytes, "sticky input changed, not kept resident");
            return Ok(None);
        }
        let e = unsafe { alloc(backend, name, t, &data, fp)? };
        let d = e.d;
        self.entries.insert(name.to_owned(), e);
        self.stats.misses += 1;
        tracing::trace!(name, bytes = nbytes, "sticky allocate");
        Ok(Some(d))
    }
}

unsafe fn alloc(backend: &Backend, name: &str, t: &HostTensor, data: &[f32], fp: u64) -> Result<Entry> {
    let ctx = g::ggml_init(g::ggml_init_params {
        mem_size: g::ggml_tensor_overhead() * 2,
        mem_buffer: std::ptr::null_mut(),
        no_alloc: true,
    });
    if ctx.is_null() {
        return Err(Error::ggml("ggml_init for a sticky input failed"));
    }
    let d = match ggml::new_tensor(ctx, DType::F32, &t.shape) {
        Ok(d) => d,
        Err(e) => {
            g::ggml_free(ctx);
            return Err(e);
        }
    };
    ggml::set_name(d.t, &format!("sticky:{name}"));
    let buffer = g::ggml_backend_alloc_ctx_tensors(ctx, backend.primary);
    if buffer.is_null() {
        g::ggml_free(ctx);
        return Err(Error::ggml("could not allocate a sticky input buffer"));
    }
    let nbytes = data.len() * 4;
    g::ggml_backend_tensor_set(d.t, data.as_ptr().cast(), 0, nbytes);
    Ok(Entry { ctx, buffer, d, fp, nbytes })
}

/// FNV-1a over the length, the two ends and a stride of samples. Cheap enough
/// to run on every input of every step; see the module comment for the risk.
fn fingerprint(data: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u32| {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    };
    eat(data.len() as u32);
    let n = data.len();
    let ends = EDGE.min(n);
    for &v in &data[..ends] {
        eat(v.to_bits());
    }
    for &v in &data[n - ends..] {
        eat(v.to_bits());
    }
    let step = (n / SAMPLES).max(1);
    let mut i = 0usize;
    while i < n {
        eat(data[i].to_bits());
        i += step;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_sees_edges_and_samples() {
        let a: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
        let mut b = a.clone();
        assert_eq!(fingerprint(&a), fingerprint(&b));
        b[0] = -1.0;
        assert_ne!(fingerprint(&a), fingerprint(&b));
        let mut c = a.clone();
        c[9_999] = -1.0;
        assert_ne!(fingerprint(&a), fingerprint(&c));
        let mut d = a.clone();
        // a run longer than the sampling stride is always caught
        d[5_000..5_100].fill(-1.0);
        assert_ne!(fingerprint(&a), fingerprint(&d));
        assert_ne!(fingerprint(&a), fingerprint(&a[..a.len() - 1]));
    }
}
