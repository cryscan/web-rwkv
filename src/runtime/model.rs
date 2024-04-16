use std::{collections::HashMap, future::Future};

use anyhow::Result;
use futures::future::BoxFuture;
use half::f16;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wasm_bindgen::prelude::wasm_bindgen;

use super::loader::{Lora, Reader};
use crate::{
    context::{Context, ContextBuilder},
    impl_deserialize_seed,
    num::Scalar,
    tensor::{TensorCpu, TensorError, TensorGpuView},
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
    pub time_mix_adapter_size: usize,
    pub time_decay_adapter_size: usize,
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

pub trait State {
    /// Batch number of this state.
    fn num_batch(&self) -> usize;
    /// Initialize a one-batch state on CPU.
    fn init(&self) -> TensorCpu<f32>;
    /// The part of the state that is used in an `att` layer.
    fn att(&self, layer: usize) -> Result<TensorGpuView<f32>, TensorError>;
    /// The part of the state that is used in an `ffn` layer.
    fn ffn(&self, layer: usize) -> Result<TensorGpuView<f32>, TensorError>;
    /// Load a batch of the state from CPU to GPU.
    fn load(&self, batch: usize, tensor: TensorCpu<f32>) -> Result<(), TensorError>;
    /// Read back a batch of the state from GPU to CPU.
    fn back(&self, batch: usize) -> BoxFuture<Result<TensorCpu<f32>, TensorError>>;
    /// Get an embed vector from a backed state.
    fn embed(&self, layer: usize, backed: TensorCpu<f32>) -> Result<TensorCpu<f32>, TensorError>;
}

pub trait ModelRuntime {
    fn info(&self) -> ModelInfo;
    fn state(&self) -> impl State + Send + Sync + 'static;
    fn model(&self) -> impl Serialize + Send + Sync + 'static;
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
    pub num_batch: usize,
}

impl<R: Reader> ModelBuilder<R> {
    pub fn new(context: &Context, model: R) -> Self {
        Self {
            context: context.clone(),
            model,
            lora: vec![],
            quant: Default::default(),
            embed_device: Default::default(),
            num_batch: 1,
        }
    }

    pub fn with_quant(mut self, value: HashMap<usize, Quant>) -> Self {
        self.quant = value;
        self
    }

    pub fn with_embed_device(mut self, value: EmbedDevice) -> Self {
        self.embed_device = value;
        self
    }

    pub fn with_num_batch(mut self, value: usize) -> Self {
        assert_ne!(value, 0, "`num_batch` must not be 0");
        self.num_batch = value;
        self
    }

    pub fn add_lora(mut self, value: Lora<R>) -> Self {
        self.lora.push(value);
        self
    }
}

pub trait ContextAutoLimits {
    /// Compute the limits automatically based on given model build info.
    fn with_auto_limits(self, info: &ModelInfo) -> Self;
}

impl ContextAutoLimits for ContextBuilder {
    fn with_auto_limits(mut self, info: &ModelInfo) -> Self {
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
