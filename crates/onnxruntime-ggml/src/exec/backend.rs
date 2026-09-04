//! ggml backend selection and the scheduler that spreads a graph over them.
//!
//! The preferred device (Metal on macOS, Vulkan elsewhere) comes first; the CPU
//! backend always comes last so any op the GPU kernel set lacks still runs,
//! inside ggml, without an onnxruntime partition boundary.

use std::ffi::CStr;
use std::sync::Mutex;

use ggml_sys as g;

use crate::error::{Error, Result};

/// Graph capacity per scheduler. pocket-tts is ~2300 ONNX nodes; each becomes
/// one to three ggml nodes.
pub const GRAPH_SIZE: usize = 32768;

/// How resident matmul weights are stored on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPrecision {
    F32,
    /// Half the memory traffic, and the type Metal's matmul kernels want. Only
    /// the 2-D matmul weights are converted; biases and norms stay f32.
    F16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// GPU if any is present, else CPU.
    Auto,
    Gpu,
    Cpu,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub device: Device,
    pub threads: i32,
    /// Claim a subgraph even when some ops are unsupported (onnxruntime then
    /// runs the rest on its CPU provider, with copies at every boundary).
    pub partial: bool,
    /// Log every value at trace level, including data heads. Slow.
    pub dump: bool,
    /// Add the ACCEL backends (BLAS on macOS) to the scheduler. They only take
    /// matmuls, and the split costs more than they save on this workload.
    pub accel: bool,
    /// Storage type for the 2-D weight matrices `ggml_mul_mat` reads as src0.
    pub weights: WeightPrecision,
    /// Keep large float graph inputs resident on the device between runs and
    /// re-upload only when a fingerprint of the host bytes changed
    /// (`exec::sticky`).
    pub sticky: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            device: Device::Auto,
            threads: default_threads(),
            partial: false,
            dump: false,
            accel: false,
            weights: WeightPrecision::F16,
            sticky: true,
        }
    }
}

fn default_threads() -> i32 {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (n / 2).clamp(1, 8) as i32
}

impl Options {
    /// Environment overrides: ORT_GGML_DEVICE, ORT_GGML_THREADS, ORT_GGML_PARTIAL,
    /// ORT_GGML_DUMP, ORT_GGML_ACCEL, ORT_GGML_WEIGHTS, ORT_GGML_STICKY.
    pub fn from_env() -> Options {
        let mut o = Options::default();
        o.apply("device", std::env::var("ORT_GGML_DEVICE").ok().as_deref());
        o.apply("threads", std::env::var("ORT_GGML_THREADS").ok().as_deref());
        o.apply("partial", std::env::var("ORT_GGML_PARTIAL").ok().as_deref());
        o.apply("dump", std::env::var("ORT_GGML_DUMP").ok().as_deref());
        o.apply("accel", std::env::var("ORT_GGML_ACCEL").ok().as_deref());
        o.apply("weights", std::env::var("ORT_GGML_WEIGHTS").ok().as_deref());
        o.apply("sticky", std::env::var("ORT_GGML_STICKY").ok().as_deref());
        o
    }

    /// Apply one option by name; unknown names are ignored with a warning.
    pub fn apply(&mut self, key: &str, value: Option<&str>) {
        let Some(value) = value else { return };
        let v = value.trim().to_ascii_lowercase();
        match key {
            "device" => {
                self.device = match v.as_str() {
                    "gpu" | "metal" | "vulkan" => Device::Gpu,
                    "cpu" => Device::Cpu,
                    "auto" | "" => Device::Auto,
                    other => {
                        tracing::warn!(value = other, "unknown device option, using auto");
                        Device::Auto
                    }
                }
            }
            "threads" => match v.parse::<i32>() {
                Ok(n) if n > 0 => self.threads = n,
                _ => tracing::warn!(value = %v, "bad threads option"),
            },
            "partial" => self.partial = matches!(v.as_str(), "1" | "true" | "yes"),
            "dump" => self.dump = matches!(v.as_str(), "1" | "true" | "yes"),
            "accel" => self.accel = matches!(v.as_str(), "1" | "true" | "yes"),
            "sticky" => self.sticky = matches!(v.as_str(), "1" | "true" | "yes"),
            "weights" => {
                self.weights = match v.as_str() {
                    "f16" | "fp16" | "half" | "" => WeightPrecision::F16,
                    "f32" | "fp32" | "float" => WeightPrecision::F32,
                    other => {
                        tracing::warn!(value = other, "unknown weights option, using f16");
                        WeightPrecision::F16
                    }
                }
            }
            other => tracing::warn!(key = other, "unknown option"),
        }
    }
}

pub struct Backend {
    pub backends: Vec<g::ggml_backend_t>,
    pub sched: g::ggml_backend_sched_t,
    /// Where weights live and where compute is preferred.
    pub primary: g::ggml_backend_t,
    pub primary_name: String,
    pub gpu: bool,
    pub options: Options,
    /// ggml's scheduler is not re-entrant; runs are serialised here.
    pub lock: Mutex<()>,
}

unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

fn load_backends_once() {
    static LOAD: std::sync::Once = std::sync::Once::new();
    LOAD.call_once(|| unsafe { g::ggml_backend_load_all() });
}

unsafe fn dev_name(dev: g::ggml_backend_dev_t) -> String {
    CStr::from_ptr(g::ggml_backend_dev_name(dev)).to_string_lossy().into_owned()
}

unsafe fn dev_desc(dev: g::ggml_backend_dev_t) -> String {
    CStr::from_ptr(g::ggml_backend_dev_description(dev)).to_string_lossy().into_owned()
}

/// Set the thread count through the registry, which works whether the CPU
/// backend is linked statically or loaded as a module.
unsafe fn set_threads(backend: g::ggml_backend_t, n: i32) {
    let dev = g::ggml_backend_get_device(backend);
    if dev.is_null() {
        return;
    }
    let reg = g::ggml_backend_dev_backend_reg(dev);
    if reg.is_null() {
        return;
    }
    let addr = g::ggml_backend_reg_get_proc_address(reg, c"ggml_backend_set_n_threads".as_ptr());
    if !addr.is_null() {
        let f: g::ggml_backend_set_n_threads_t = std::mem::transmute(addr);
        if let Some(f) = f {
            f(backend, n);
            tracing::debug!(threads = n, "cpu backend threads set");
        }
    }
}

impl Backend {
    pub fn new(options: Options) -> Result<Backend> {
        unsafe {
            load_backends_once();
            let n = g::ggml_backend_dev_count();
            let mut list = Vec::new();
            for i in 0..n {
                let dev = g::ggml_backend_dev_get(i);
                let ty = g::ggml_backend_dev_type(dev);
                tracing::info!(index = i, name = %dev_name(dev), description = %dev_desc(dev), type_ = ty, "ggml device");
                list.push((dev, ty));
            }
            let mut backends = Vec::new();
            let mut gpu = false;
            if options.device != Device::Cpu {
                for &(dev, ty) in &list {
                    if ty == g::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                        || ty == g::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
                    {
                        let b = g::ggml_backend_dev_init(dev, std::ptr::null());
                        if b.is_null() {
                            tracing::warn!(name = %dev_name(dev), "gpu backend failed to initialise");
                            continue;
                        }
                        tracing::info!(name = %dev_name(dev), description = %dev_desc(dev), "using gpu backend");
                        backends.push(b);
                        gpu = true;
                        break;
                    }
                }
                if !gpu && options.device == Device::Gpu {
                    return Err(Error::ggml("device=gpu requested but no gpu backend is available"));
                }
            }
            // ACCEL devices (BLAS on macOS) sit between the GPU and the CPU.
            for &(dev, ty) in list.iter().filter(|_| options.accel) {
                if ty == g::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_ACCEL {
                    let b = g::ggml_backend_dev_init(dev, std::ptr::null());
                    if !b.is_null() {
                        tracing::info!(name = %dev_name(dev), "using accel backend");
                        backends.push(b);
                    }
                }
            }
            let cpu = g::ggml_backend_init_by_type(g::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU, std::ptr::null());
            if cpu.is_null() {
                return Err(Error::ggml("no cpu backend"));
            }
            set_threads(cpu, options.threads);
            backends.push(cpu);
            let primary = backends[0];
            let primary_name = CStr::from_ptr(g::ggml_backend_name(primary)).to_string_lossy().into_owned();
            let sched = g::ggml_backend_sched_new(
                backends.as_mut_ptr(),
                std::ptr::null_mut(),
                backends.len() as i32,
                GRAPH_SIZE,
                false,
                true,
            );
            if sched.is_null() {
                return Err(Error::ggml("ggml_backend_sched_new failed"));
            }
            tracing::info!(primary = %primary_name, n_backends = backends.len(), gpu, threads = options.threads, "backend ready");
            Ok(Backend { backends, sched, primary, primary_name, gpu, options, lock: Mutex::new(()) })
        }
    }

    pub fn cpu(&self) -> g::ggml_backend_t {
        *self.backends.last().unwrap()
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            if !self.sched.is_null() {
                g::ggml_backend_sched_free(self.sched);
            }
            for &b in &self.backends {
                g::ggml_backend_free(b);
            }
        }
        tracing::debug!("backend released");
    }
}
