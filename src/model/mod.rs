use std::{collections::HashMap, future::Future};

use anyhow::Result;
use half::f16;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wasm_bindgen::prelude::wasm_bindgen;

use self::{
    loader::{Loader, Lora, Reader},
    run::ModelRun,
    softmax::ModelSoftmax,
};
use crate::{
    context::{Context, ContextBuilder},
    impl_deserialize_seed,
    num::Scalar,
    tensor::TensorError,
};

pub mod loader;
pub mod run;
pub mod softmax;
pub mod v4;
pub mod v5;
pub mod v6;

pub const RESCALE_LAYER: usize = 6;
pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;

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
    #[error("no viable chunk size found")]
    NoViableChunkSize,
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

/// Input of one inference slot.
#[derive(Debug, Default, Clone)]
pub struct ModelInput {
    pub tokens: Vec<u16>,
    pub ty: OutputType,
}

/// Output distribution of one inference slot.
#[derive(Debug, Default, Clone)]
pub enum ModelOutput {
    /// This slot is empty.
    #[default]
    None,
    /// Only the prediction of the last token.
    Last(Vec<f32>),
    /// Predictions of all input tokens.
    Full(Vec<Vec<f32>>),
}

impl ModelOutput {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub fn is_some(&self) -> bool {
        !self.is_none()
    }

    pub fn concat(self, other: Self) -> Self {
        match (self, other) {
            (Self::None, y) => y,
            (x, Self::None) => x,
            (Self::Last(x), Self::Last(y)) => Self::Full(vec![x, y]),
            (Self::Last(x), Self::Full(y)) => Self::Full([vec![x], y].concat()),
            (Self::Full(x), Self::Last(y)) => Self::Full([x, vec![y]].concat()),
            (Self::Full(x), Self::Full(y)) => Self::Full([x, y].concat()),
        }
    }
}

#[wasm_bindgen]
#[derive(Debug, Default, Clone, Copy)]
pub enum OutputType {
    /// Only the prediction of the last token.
    #[default]
    Last,
    /// Predictions of all input tokens.
    Full,
}

pub trait Build<T> {
    type Error;

    fn build(self) -> Result<T, Self::Error>;
}

pub trait BuildFuture<T> {
    type Error;

    fn build(self) -> impl Future<Output = Result<T, Self::Error>>;
}

pub trait BackedState: Serialize + for<'a> Deserialize<'a> {
    fn num_batch(&self) -> usize;
    fn num_layer(&self) -> usize;

    /// Extract the embedding from a given layer of the state.
    fn embed(&self, batch: usize, layer: usize) -> Vec<f32>;
}

pub trait ModelState {
    type BackedState: BackedState;

    fn num_batch(&self) -> usize;

    /// Load the state from host. Their shapes must match.
    fn load(&self, backed: &Self::BackedState) -> Result<(), TensorError>;
    /// Load one batch from host. The batch size the backed state should be 1.
    fn load_batch(&self, backed: &Self::BackedState, batch: usize) -> Result<(), TensorError>;
    /// Back the entire device state to host.
    fn back(&self) -> impl Future<Output = Self::BackedState>;
    /// Back one batch of the device state to host.
    fn back_batch(
        &self,
        batch: usize,
    ) -> impl Future<Output = Result<Self::BackedState, TensorError>>;
    /// Copy one device state to another. Their shapes must match.
    fn blit(&self, other: &Self) -> Result<(), TensorError>;
    /// Copy one batch from the source state to another.
    fn blit_batch(
        &self,
        other: &Self,
        from_batch: usize,
        to_batch: usize,
    ) -> Result<(), TensorError>;
}

pub trait ModelBase {
    fn context(&self) -> &Context;
    fn info(&self) -> &ModelInfo;
}

pub trait Model: ModelBase + ModelSoftmax + ModelRun {}

impl<M> Model for M where M: ModelBase + ModelSoftmax + ModelRun {}

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

#[wasm_bindgen]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedDevice {
    #[default]
    Cpu,
    Gpu,
}

pub struct ModelBuilder<R: Reader> {
    context: Context,
    model: R,
    lora: Vec<Lora<R>>,
    quant: HashMap<usize, Quant>,
    embed_device: EmbedDevice,
    turbo: bool,
    token_chunk_size: usize,
}

struct PreparedModelBuilder<R: Reader> {
    context: Context,
    info: ModelInfo,
    loader: Loader<R>,
    quant: HashMap<usize, Quant>,
    embed_device: EmbedDevice,
    turbo: bool,
    token_chunk_size: usize,
}

impl<R: Reader> ModelBuilder<R> {
    pub fn new(context: &Context, model: R) -> Self {
        Self {
            context: context.clone(),
            model,
            lora: vec![],
            quant: Default::default(),
            turbo: false,
            embed_device: Default::default(),
            token_chunk_size: 32,
        }
    }

    fn prepare(self) -> Result<PreparedModelBuilder<R>> {
        let ModelBuilder {
            context,
            model,
            lora,
            quant,
            embed_device,
            turbo,
            token_chunk_size,
        } = self;

        let info = Loader::info(&model)?;
        let loader = Loader {
            context: context.clone(),
            model,
            lora,
        };

        let token_chunk_size = token_chunk_size
            .max(MIN_TOKEN_CHUNK_SIZE)
            .next_power_of_two();

        Ok(PreparedModelBuilder {
            context,
            info,
            loader,
            quant,
            embed_device,
            turbo,
            token_chunk_size,
        })
    }

    pub fn quant(mut self, value: HashMap<usize, Quant>) -> Self {
        self.quant = value;
        self
    }

    pub fn lora(mut self, value: Lora<R>) -> Self {
        self.lora.push(value);
        self
    }

    pub fn embed_device(mut self, value: EmbedDevice) -> Self {
        self.embed_device = value;
        self
    }

    pub fn turbo(mut self, value: bool) -> Self {
        self.turbo = value;
        self
    }

    pub fn token_chunk_size(mut self, value: usize) -> Self {
        self.token_chunk_size = value;
        self
    }
}

/// Create a model state.
/// - `num_batch`: The maximum number of runtime slots.
/// - `chunk_size`: Internally, the state is split into chunks of layers, since there is a size limit on one GPU buffer (128 MB).
///
/// If there is only one batch, it is recommended to set `chunk_size` to `info.num_layers()`.
pub struct StateBuilder {
    context: Context,
    info: ModelInfo,
    num_batch: usize,
    chunk_size: usize,
}

impl StateBuilder {
    pub fn new(context: &Context, info: &ModelInfo) -> Self {
        Self {
            context: context.clone(),
            info: info.clone(),
            num_batch: 1,
            chunk_size: info.num_layer,
        }
    }

    pub fn with_num_batch(mut self, value: usize) -> Self {
        self.num_batch = value;
        self
    }

    pub fn with_chunk_size(mut self, value: usize) -> Self {
        self.chunk_size = value;
        self
    }
}

pub trait ContextAutoLimits {
    /// Compute the limits automatically based on given model build info.
    fn auto_limits(self, info: &ModelInfo) -> Self;
}

impl ContextAutoLimits for ContextBuilder {
    fn auto_limits(mut self, info: &ModelInfo) -> Self {
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
