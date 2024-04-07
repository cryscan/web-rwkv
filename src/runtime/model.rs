use std::{collections::HashMap, future::Future};

use anyhow::Result;
use half::f16;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wasm_bindgen::prelude::wasm_bindgen;

use super::loader::{Lora, Reader};
use crate::{
    context::{Context, ContextBuilder},
    impl_deserialize_seed,
    num::Scalar,
};

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

pub trait Build<T> {
    fn build(self) -> impl Future<Output = Result<T>>;
}

pub struct ModelBuilder<R: Reader> {
    pub context: Context,
    pub model: R,
    pub lora: Vec<Lora<R>>,
    pub quant: HashMap<usize, Quant>,
    pub embed_device: EmbedDevice,
}

impl<R: Reader> ModelBuilder<R> {
    pub fn new(context: &Context, model: R) -> Self {
        Self {
            context: context.clone(),
            model,
            lora: vec![],
            quant: Default::default(),
            embed_device: Default::default(),
        }
    }

    pub fn with_quant(mut self, value: HashMap<usize, Quant>) -> Self {
        self.quant = value;
        self
    }

    pub fn add_lora(mut self, value: Lora<R>) -> Self {
        self.lora.push(value);
        self
    }

    pub fn with_embed_device(mut self, value: EmbedDevice) -> Self {
        self.embed_device = value;
        self
    }
}

impl ContextBuilder {
    /// Compute the limits automatically based on given model build info.
    pub fn with_auto_limits(mut self, info: &ModelInfo) -> Self {
        self.limits.max_buffer_size = ModelInfo::BUFFER_SIZE
            .max(info.max_non_head_buffer_size())
            .max(info.head_buffer_size()) as u64;
        self.limits.max_storage_buffer_binding_size = ModelInfo::STORAGE_BUFFER_BINDING_SIZE
            .max(info.max_non_head_buffer_size())
            .max(info.head_buffer_size())
            as u32;
        self
    }
}
