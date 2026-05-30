use std::collections::HashMap;

use js_sys::{ArrayBuffer, Uint8Array};
use safetensors::{Dtype, SafeTensorError};
use wasm_bindgen::prelude::*;
use web_rwkv::runtime::loader::{Reader, ReaderTensor};

#[wasm_bindgen(js_name = Tensor)]
#[derive(Debug, Clone)]
pub struct JsTensor {
    name: String,
    shape: Vec<usize>,
    buffer: ArrayBuffer,
}

#[wasm_bindgen(js_class = Tensor)]
impl JsTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(name: String, shape: &[usize], buffer: ArrayBuffer) -> Self {
        let shape = shape.to_vec();
        Self {
            name,
            shape,
            buffer,
        }
    }
}

#[wasm_bindgen]
pub struct TensorReader {
    names: Vec<String>,
    tensors: HashMap<String, JsTensor>,
}

#[wasm_bindgen]
impl TensorReader {
    #[wasm_bindgen(constructor)]
    pub fn new(tensors: Vec<JsTensor>) -> Self {
        let names = tensors.iter().map(|x| x.name.clone()).collect();
        let tensors = tensors.into_iter().map(|x| (x.name.clone(), x)).collect();
        Self { names, tensors }
    }
}

impl Reader for TensorReader {
    fn names(&self) -> Vec<&str> {
        self.names.iter().map(AsRef::as_ref).collect()
    }

    fn contains(&self, name: &str) -> bool {
        self.names.contains(&name.to_string())
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>, SafeTensorError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or(SafeTensorError::TensorNotFound(name.to_string()))?;
        Ok(tensor.shape.clone())
    }

    fn tensor(&'_ self, name: &str) -> Result<ReaderTensor<'_>, SafeTensorError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or(SafeTensorError::TensorNotFound(name.to_string()))?;

        let array = Uint8Array::new(&tensor.buffer);
        let data = array.to_vec();

        Ok((Dtype::F16, tensor.shape.clone(), data.into()))
        // TensorView::new(Dtype::F16, shape.clone(), data)
    }
}
