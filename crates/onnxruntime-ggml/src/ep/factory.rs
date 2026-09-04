//! `OrtEpFactory`: tells onnxruntime which hardware devices the provider can
//! serve and creates an `OrtEp` per session.
//!
//! The provider registers on the CPU hardware device and declares no device
//! memory: onnxruntime hands it inputs in host memory and takes outputs back
//! the same way. Where ggml actually computes (Metal, Vulkan, CPU) is the
//! provider's own business and invisible to onnxruntime.

use std::ffi::CString;
use std::os::raw::c_char;

use ort_ep_sys::*;

use crate::ep::provider::Provider;
use crate::error::Result;
use crate::ort::api::{api, ep_api, guard};
use crate::{ep_call, EP_VENDOR, EP_VENDOR_ID, EP_VERSION};

#[repr(C)]
pub struct Factory {
    base: OrtEpFactory,
    pub name: CString,
    vendor: CString,
    version: CString,
    pub logger: *const OrtLogger,
}

impl Factory {
    /// Allocate a factory and hand its `OrtEpFactory` header to onnxruntime.
    pub fn create(name: String, logger: *const OrtLogger) -> *mut OrtEpFactory {
        let mut base: OrtEpFactory = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.GetName = Some(get_name);
        base.GetVendor = Some(get_vendor);
        base.GetVendorId = Some(get_vendor_id);
        base.GetVersion = Some(get_version);
        base.GetSupportedDevices = Some(get_supported_devices);
        base.CreateEp = Some(create_ep);
        base.ReleaseEp = Some(release_ep);
        base.ValidateCompiledModelCompatibilityInfo = Some(validate_compat);
        base.CreateAllocator = Some(create_allocator);
        base.ReleaseAllocator = Some(release_allocator);
        base.CreateDataTransfer = Some(create_data_transfer);
        base.IsStreamAware = Some(is_stream_aware);
        base.CreateSyncStreamForDevice = Some(create_sync_stream);
        base.GetHardwareDeviceIncompatibilityDetails = Some(incompatibility_details);
        base.CreateExternalResourceImporterForDevice = Some(create_importer);
        base.GetNumCustomOpDomains = Some(num_custom_op_domains);
        base.GetCustomOpDomains = Some(custom_op_domains);
        let factory = Box::new(Factory {
            base,
            name: CString::new(name).unwrap_or_else(|_| c"ggml".to_owned()),
            vendor: CString::new(EP_VENDOR).unwrap(),
            version: CString::new(EP_VERSION).unwrap(),
            logger,
        });
        Box::into_raw(factory) as *mut OrtEpFactory
    }

    /// # Safety
    /// `p` came from `create`.
    pub unsafe fn release(p: *mut OrtEpFactory) {
        if !p.is_null() {
            drop(Box::from_raw(p as *mut Factory));
        }
    }

    unsafe fn from_ptr<'a>(p: *const OrtEpFactory) -> &'a Factory {
        &*(p as *const Factory)
    }
}

unsafe extern "C" fn get_name(this: *const OrtEpFactory) -> *const c_char {
    Factory::from_ptr(this).name.as_ptr()
}

unsafe extern "C" fn get_vendor(this: *const OrtEpFactory) -> *const c_char {
    Factory::from_ptr(this).vendor.as_ptr()
}

unsafe extern "C" fn get_vendor_id(_this: *const OrtEpFactory) -> u32 {
    EP_VENDOR_ID
}

unsafe extern "C" fn get_version(this: *const OrtEpFactory) -> *const c_char {
    Factory::from_ptr(this).version.as_ptr()
}

unsafe extern "C" fn get_supported_devices(
    this: *mut OrtEpFactory,
    devices: *const *const OrtHardwareDevice,
    num_devices: usize,
    ep_devices: *mut *mut OrtEpDevice,
    max_ep_devices: usize,
    num_ep_devices: *mut usize,
) -> *mut OrtStatus {
    guard("GetSupportedDevices", || {
        *num_ep_devices = 0;
        let api = api();
        let type_of = api.HardwareDevice_Type.ok_or_else(|| crate::error::Error::Ort("HardwareDevice_Type missing".into()))?;
        for i in 0..num_devices {
            if *num_ep_devices >= max_ep_devices {
                break;
            }
            let device = *devices.add(i);
            let ty = type_of(device);
            let vendor = api.HardwareDevice_Vendor.map(|f| crate::ort::api::cstr(f(device))).unwrap_or_default();
            tracing::debug!(index = i, type_ = ty, vendor = %vendor, "hardware device offered");
            if ty != OrtHardwareDeviceType_CPU {
                continue;
            }
            let mut metadata: *mut OrtKeyValuePairs = std::ptr::null_mut();
            if let (Some(create), Some(add)) = (api.CreateKeyValuePairs, api.AddKeyValuePair) {
                create(&mut metadata);
                let v = CString::new(EP_VERSION).unwrap();
                add(metadata, c"version".as_ptr(), v.as_ptr());
                add(metadata, c"backend".as_ptr(), c"ggml".as_ptr());
            }
            let mut ep_device: *mut OrtEpDevice = std::ptr::null_mut();
            let result: Result<()> = (|| ep_call!(CreateEpDevice(this, device, metadata, std::ptr::null(), &mut ep_device)))();
            if let Some(release) = api.ReleaseKeyValuePairs {
                if !metadata.is_null() {
                    release(metadata);
                }
            }
            result?;
            *ep_devices.add(*num_ep_devices) = ep_device;
            *num_ep_devices += 1;
            tracing::info!(name = %Factory::from_ptr(this).name.to_string_lossy(), "registered on the CPU device");
        }
        Ok(())
    })
}

unsafe extern "C" fn create_ep(
    this: *mut OrtEpFactory,
    _devices: *const *const OrtHardwareDevice,
    _ep_metadata: *const *const OrtKeyValuePairs,
    num_devices: usize,
    session_options: *const OrtSessionOptions,
    logger: *const OrtLogger,
    ep: *mut *mut OrtEp,
) -> *mut OrtStatus {
    guard("CreateEp", || {
        *ep = std::ptr::null_mut();
        let factory = Factory::from_ptr(this);
        let name = factory.name.to_string_lossy().into_owned();
        tracing::info!(ep = %name, num_devices, "CreateEp");
        let provider = Provider::create(&name, session_options, logger)?;
        *ep = provider;
        Ok(())
    })
}

unsafe extern "C" fn release_ep(_this: *mut OrtEpFactory, ep: *mut OrtEp) {
    tracing::info!("ReleaseEp");
    Provider::release(ep);
}

unsafe extern "C" fn validate_compat(
    _this: *mut OrtEpFactory,
    _devices: *const *const OrtHardwareDevice,
    _num_devices: usize,
    _info: *const c_char,
    out: *mut OrtCompiledModelCompatibility,
) -> *mut OrtStatus {
    // No EPContext support yet: compiled models from this provider do not exist.
    if !out.is_null() {
        *out = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
    std::ptr::null_mut()
}

unsafe extern "C" fn create_allocator(
    _this: *mut OrtEpFactory,
    _memory_info: *const OrtMemoryInfo,
    _options: *const OrtKeyValuePairs,
    allocator: *mut *mut OrtAllocator,
) -> *mut OrtStatus {
    // Host memory only: onnxruntime's own CPU allocator serves the provider.
    *allocator = std::ptr::null_mut();
    std::ptr::null_mut()
}

unsafe extern "C" fn release_allocator(_this: *mut OrtEpFactory, _allocator: *mut OrtAllocator) {}

unsafe extern "C" fn create_data_transfer(_this: *mut OrtEpFactory, dt: *mut *mut OrtDataTransferImpl) -> *mut OrtStatus {
    *dt = std::ptr::null_mut();
    std::ptr::null_mut()
}

unsafe extern "C" fn is_stream_aware(_this: *const OrtEpFactory) -> bool {
    false
}

unsafe extern "C" fn create_sync_stream(
    _this: *mut OrtEpFactory,
    _device: *const OrtMemoryDevice,
    _options: *const OrtKeyValuePairs,
    stream: *mut *mut OrtSyncStreamImpl,
) -> *mut OrtStatus {
    *stream = std::ptr::null_mut();
    std::ptr::null_mut()
}

unsafe extern "C" fn incompatibility_details(
    _this: *mut OrtEpFactory,
    _hw: *const OrtHardwareDevice,
    _details: *mut OrtDeviceEpIncompatibilityDetails,
) -> *mut OrtStatus {
    std::ptr::null_mut()
}

unsafe extern "C" fn create_importer(
    _this: *mut OrtEpFactory,
    _device: *const OrtEpDevice,
    out: *mut *mut OrtExternalResourceImporterImpl,
) -> *mut OrtStatus {
    if !out.is_null() {
        *out = std::ptr::null_mut();
    }
    std::ptr::null_mut()
}

unsafe extern "C" fn num_custom_op_domains(_this: *mut OrtEpFactory, n: *mut usize) -> *mut OrtStatus {
    *n = 0;
    std::ptr::null_mut()
}

unsafe extern "C" fn custom_op_domains(_this: *mut OrtEpFactory, _domains: *mut *mut OrtCustomOpDomain, _n: usize) -> *mut OrtStatus {
    std::ptr::null_mut()
}

#[allow(dead_code)]
fn _ep_api_used() -> &'static OrtEpApi {
    ep_api()
}
