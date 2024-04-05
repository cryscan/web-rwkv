use anyhow::Result;
use half::f16;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::{impl_deserialize_seed, num::Scalar};

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelVersion {
    V4,
    V5,
    V6,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
pub enum ModelError {
    #[error("invalid model version")]
    InvalidVersion,
}

#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelInfo {
    pub version: ModelVersion,
    pub num_layer: usize,
    pub num_emb: usize,
    pub num_hidden: usize,
    pub num_vocab: usize,
    pub num_head: usize,
}

impl ModelInfo {
    pub const BUFFER_SIZE: usize = 256 << 20;
    pub const STORAGE_BUFFER_BINDING_SIZE: usize = 128 << 20;
}

impl_deserialize_seed!(ModelInfo);

#[wasm_bindgen]
impl ModelInfo {
    /// The required storage buffer size, not including head.
    pub fn max_non_head_buffer_size(&self) -> usize {
        self.num_emb * self.num_hidden * f16::size()
    }

    /// The head and embed's size.
    pub fn head_buffer_size(&self) -> usize {
        self.num_emb * self.num_vocab * f16::size()
    }
}

/// Quantization of a layer.
#[wasm_bindgen]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Quant {
    /// No quantization.
    #[default]
    None,
    /// Use `Int8` quantization.
    Int8,
    /// Use `NF4` quantization.
    NF4,
}

/// Device to put the model's embed tensor.
#[wasm_bindgen]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedDevice {
    #[default]
    Cpu,
    Gpu,
}
