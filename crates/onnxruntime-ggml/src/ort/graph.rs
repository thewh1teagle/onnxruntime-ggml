//! Import an `OrtGraph` (the view onnxruntime hands `Compile`) into `ir::Graph`.

use std::collections::HashMap;

use ort_ep_sys::*;

use crate::error::{Error, Result};
use crate::host::tensor::HostTensor;
use crate::ir::{Attr, DType, Graph, Node, ValueDesc};
use crate::ort::api::cstr;
use crate::ort_call;

/// # Safety
/// `graph` is a valid `OrtGraph`.
pub unsafe fn import(graph: *const OrtGraph) -> Result<Graph> {
    let mut out = Graph::default();
    let mut name: *const std::os::raw::c_char = std::ptr::null();
    ort_call!(Graph_GetName(graph, &mut name))?;
    out.name = cstr(name);

    for vi in graph_values(graph, true)? {
        out.inputs.push(value_desc(vi)?);
    }
    for vi in graph_values(graph, false)? {
        out.outputs.push(value_desc(vi)?);
    }

    let mut n_init = 0usize;
    ort_call!(Graph_GetNumInitializers(graph, &mut n_init))?;
    let mut inits: Vec<*const OrtValueInfo> = vec![std::ptr::null(); n_init];
    if n_init > 0 {
        ort_call!(Graph_GetInitializers(graph, inits.as_mut_ptr(), n_init))?;
    }
    let mut init_bytes = 0usize;
    for vi in inits {
        let mut cname = std::ptr::null();
        ort_call!(GetValueInfoName(vi, &mut cname))?;
        let name = cstr(cname);
        let mut value: *const OrtValue = std::ptr::null();
        ort_call!(ValueInfo_GetInitializerValue(vi, &mut value))?;
        let t = read_value(value)?;
        init_bytes += t.nbytes();
        tracing::trace!(name = %name, tensor = %t.brief(), "initializer");
        out.constants.insert(name, t);
    }

    let mut n_nodes = 0usize;
    ort_call!(Graph_GetNumNodes(graph, &mut n_nodes))?;
    let mut nodes: Vec<*const OrtNode> = vec![std::ptr::null(); n_nodes];
    if n_nodes > 0 {
        ort_call!(Graph_GetNodes(graph, nodes.as_mut_ptr(), n_nodes))?;
    }
    for node in nodes {
        out.nodes.push(import_node(node)?);
    }
    tracing::info!(
        graph = %out.name,
        nodes = out.nodes.len(),
        initializers = out.constants.len(),
        initializer_bytes = %crate::logging::bytes(init_bytes),
        inputs = ?out.inputs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
        outputs = ?out.outputs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
        "graph imported"
    );
    Ok(out)
}

unsafe fn graph_values(graph: *const OrtGraph, inputs: bool) -> Result<Vec<*const OrtValueInfo>> {
    let mut n = 0usize;
    if inputs {
        ort_call!(Graph_GetNumInputs(graph, &mut n))?;
    } else {
        ort_call!(Graph_GetNumOutputs(graph, &mut n))?;
    }
    let mut list: Vec<*const OrtValueInfo> = vec![std::ptr::null(); n];
    if n > 0 {
        if inputs {
            ort_call!(Graph_GetInputs(graph, list.as_mut_ptr(), n))?;
        } else {
            ort_call!(Graph_GetOutputs(graph, list.as_mut_ptr(), n))?;
        }
    }
    Ok(list)
}

unsafe fn value_desc(vi: *const OrtValueInfo) -> Result<ValueDesc> {
    let mut cname = std::ptr::null();
    ort_call!(GetValueInfoName(vi, &mut cname))?;
    let name = cstr(cname);
    let mut type_info: *const OrtTypeInfo = std::ptr::null();
    ort_call!(GetValueInfoTypeInfo(vi, &mut type_info))?;
    let mut tensor_info: *const OrtTensorTypeAndShapeInfo = std::ptr::null();
    ort_call!(CastTypeInfoToTensorInfo(type_info, &mut tensor_info))?;
    if tensor_info.is_null() {
        return Err(Error::unsupported(format!("value '{name}' is not a tensor")));
    }
    let mut elem: ONNXTensorElementDataType = 0;
    ort_call!(GetTensorElementType(tensor_info, &mut elem))?;
    let dtype = DType::from_onnx(elem as i32)?;
    let mut rank = 0usize;
    ort_call!(GetDimensionsCount(tensor_info, &mut rank))?;
    let mut dims = vec![0i64; rank];
    if rank > 0 {
        ort_call!(GetDimensions(tensor_info, dims.as_mut_ptr(), rank))?;
    }
    let shape = dims.into_iter().map(|d| if d < 0 { None } else { Some(d) }).collect();
    Ok(ValueDesc { name, dtype, shape })
}

/// Copy a CPU `OrtValue` tensor into a `HostTensor`.
///
/// # Safety
/// `value` is a valid tensor `OrtValue` in host memory.
pub unsafe fn read_value(value: *const OrtValue) -> Result<HostTensor> {
    let mut info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
    ort_call!(GetTensorTypeAndShape(value, &mut info))?;
    let result = (|| {
        let mut elem: ONNXTensorElementDataType = 0;
        ort_call!(GetTensorElementType(info, &mut elem))?;
        let dtype = DType::from_onnx(elem as i32)?;
        let mut rank = 0usize;
        ort_call!(GetDimensionsCount(info, &mut rank))?;
        let mut dims = vec![0i64; rank];
        if rank > 0 {
            ort_call!(GetDimensions(info, dims.as_mut_ptr(), rank))?;
        }
        let shape: Vec<usize> = dims.iter().map(|&d| d.max(0) as usize).collect();
        let numel: usize = shape.iter().product();
        let mut data: *mut std::os::raw::c_void = std::ptr::null_mut();
        ort_call!(GetTensorMutableData(value as *mut OrtValue, &mut data))?;
        let nbytes = numel * dtype.size();
        let bytes: &[u8] = if nbytes == 0 { &[] } else { std::slice::from_raw_parts(data as *const u8, nbytes) };
        HostTensor::from_bytes(dtype, shape, bytes)
    })();
    if let Some(release) = crate::ort::api::api().ReleaseTensorTypeAndShapeInfo {
        release(info);
    }
    result
}

unsafe fn names_of(list: &[*const OrtValueInfo]) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(list.len());
    for &vi in list {
        if vi.is_null() {
            out.push(String::new());
            continue;
        }
        let mut cname = std::ptr::null();
        ort_call!(GetValueInfoName(vi, &mut cname))?;
        out.push(cstr(cname));
    }
    Ok(out)
}

unsafe fn import_node(node: *const OrtNode) -> Result<Node> {
    let mut p = std::ptr::null();
    ort_call!(Node_GetName(node, &mut p))?;
    let name = cstr(p);
    ort_call!(Node_GetOperatorType(node, &mut p))?;
    let op = cstr(p);
    ort_call!(Node_GetDomain(node, &mut p))?;
    let domain = cstr(p);

    let mut n_in = 0usize;
    ort_call!(Node_GetNumInputs(node, &mut n_in))?;
    let mut ins: Vec<*const OrtValueInfo> = vec![std::ptr::null(); n_in];
    if n_in > 0 {
        ort_call!(Node_GetInputs(node, ins.as_mut_ptr(), n_in))?;
    }
    let mut n_out = 0usize;
    ort_call!(Node_GetNumOutputs(node, &mut n_out))?;
    let mut outs: Vec<*const OrtValueInfo> = vec![std::ptr::null(); n_out];
    if n_out > 0 {
        ort_call!(Node_GetOutputs(node, outs.as_mut_ptr(), n_out))?;
    }

    let mut n_attr = 0usize;
    ort_call!(Node_GetNumAttributes(node, &mut n_attr))?;
    let mut attrs_p: Vec<*const OrtOpAttr> = vec![std::ptr::null(); n_attr];
    if n_attr > 0 {
        ort_call!(Node_GetAttributes(node, attrs_p.as_mut_ptr(), n_attr))?;
    }
    let mut attrs = HashMap::new();
    for a in attrs_p {
        let mut cname = std::ptr::null();
        ort_call!(OpAttr_GetName(a, &mut cname))?;
        let aname = cstr(cname);
        match read_attr(a)? {
            Some(v) => {
                attrs.insert(aname, v);
            }
            None => tracing::debug!(node = %name, attr = %aname, "attribute type skipped"),
        }
    }
    Ok(Node { name, op, domain, inputs: names_of(&ins)?, outputs: names_of(&outs)?, attrs })
}

/// Size query then read. `ReadOpAttr` counts `len` and `out` in bytes for every type.
unsafe fn read_sized(attr: *const OrtOpAttr, ty: OrtOpAttrType, _elem: usize) -> Result<Vec<u8>> {
    let api = crate::ort::api::api();
    let f = api.ReadOpAttr.ok_or_else(|| Error::Ort("ReadOpAttr missing".into()))?;
    let mut needed = 0usize;
    let st = f(attr, ty, std::ptr::null_mut(), 0, &mut needed);
    if !st.is_null() {
        if let Some(release) = api.ReleaseStatus {
            release(st);
        }
    }
    let mut buf = vec![0u8; needed];
    if needed > 0 {
        let mut got = 0usize;
        let st = f(attr, ty, buf.as_mut_ptr().cast(), needed, &mut got);
        crate::ort::api::check(st, "ReadOpAttr")?;
    }
    Ok(buf)
}

unsafe fn read_attr(attr: *const OrtOpAttr) -> Result<Option<Attr>> {
    let mut ty: OrtOpAttrType = 0;
    ort_call!(OpAttr_GetType(attr, &mut ty))?;
    Ok(match ty {
        ORT_OP_ATTR_INT => {
            let mut v: i64 = 0;
            let mut out = 0usize;
            ort_call!(ReadOpAttr(attr, ty, (&mut v as *mut i64).cast(), std::mem::size_of::<i64>(), &mut out))?;
            Some(Attr::Int(v))
        }
        ORT_OP_ATTR_INTS => {
            let bytes = read_sized(attr, ty, 8)?;
            Some(Attr::Ints(bytes.chunks_exact(8).map(|c| i64::from_ne_bytes(c.try_into().unwrap())).collect()))
        }
        ORT_OP_ATTR_FLOAT => {
            let mut v: f32 = 0.0;
            let mut out = 0usize;
            ort_call!(ReadOpAttr(attr, ty, (&mut v as *mut f32).cast(), std::mem::size_of::<f32>(), &mut out))?;
            Some(Attr::Float(v))
        }
        ORT_OP_ATTR_FLOATS => {
            let bytes = read_sized(attr, ty, 4)?;
            Some(Attr::Floats(bytes.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect()))
        }
        ORT_OP_ATTR_STRING => {
            let bytes = read_sized(attr, ty, 1)?;
            let s = String::from_utf8_lossy(&bytes).trim_end_matches('\0').to_string();
            Some(Attr::Str(s))
        }
        ORT_OP_ATTR_TENSOR => {
            let mut value: *mut OrtValue = std::ptr::null_mut();
            ort_call!(OpAttr_GetTensorAttributeAsOrtValue(attr, &mut value))?;
            if value.is_null() {
                None
            } else {
                let t = read_value(value);
                if let Some(release) = crate::ort::api::api().ReleaseValue {
                    release(value);
                }
                Some(Attr::Tensor(t?))
            }
        }
        ORT_OP_ATTR_GRAPH => return Err(Error::unsupported("subgraph attributes (Loop/If/Scan)")),
        _ => None,
    })
}
