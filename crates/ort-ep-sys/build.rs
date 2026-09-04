//! Generates bindings for the onnxruntime C API. Nothing is linked: a plugin
//! execution provider receives the whole API as a table of function pointers
//! (`OrtApiBase` -> `OrtApi` -> `OrtEpApi`) when onnxruntime loads it.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.parent().and_then(Path::parent).unwrap();
    let include_dir = root.join("libs/ort/include");
    let wrapper = manifest_dir.join("wrapper.h");
    for path in [&wrapper, &include_dir] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    if !include_dir.join("onnxruntime_c_api.h").exists() {
        panic!("missing onnxruntime headers under libs/ort/include; run `chore fetch-ort-headers`");
    }

    let mut bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_type("Ort.*")
        .allowlist_type("ONNX.*")
        .allowlist_var("ORT_.*")
        .allowlist_var("kOrt.*")
        .allowlist_function("OrtGetApiBase")
        .prepend_enum_name(false)
        .generate_comments(false)
        .derive_default(true)
        .derive_debug(false);
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        if let Ok(output) = std::process::Command::new("xcrun").args(["--show-sdk-path"]).output() {
            let sdk = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if output.status.success() && !sdk.is_empty() {
                bindings = bindings.clang_arg("-isysroot").clang_arg(sdk);
            }
        }
    }
    bindings
        .generate()
        .expect("failed to generate onnxruntime bindings")
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .expect("failed to write onnxruntime bindings");
}
