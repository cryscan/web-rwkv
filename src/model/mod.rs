use std::{collections::HashMap, convert::Infallible, future::Future};

use anyhow::Result;
use half::f16;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::wasm_bindgen;

use self::{
    loader::{Loader, Lora, Reader},
    run::ModelRun,
    softmax::ModelSoftmax,
};
use crate::{context::Context, num::Scalar, tensor::TensorError};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelError {
    InvalidVersion,
    NoViableChunkSize,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::InvalidVersion => write!(f, "invalid model version"),
            ModelError::NoViableChunkSize => write!(f, "no viable chunk size found"),
        }
    }
}

impl std::error::Error for ModelError {}

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

pub trait FromBuilder: Sized {
    type Builder<'a>;
    type Error;

    fn from_builder(
        builder: Self::Builder<'_>,
    ) -> impl Future<Output = Result<Self, Self::Error>> + Send;
}

pub trait BackedState:
    Serialize
    + for<'a> Deserialize<'a>
    + for<'a> FromBuilder<Builder<'a> = StateBuilder, Error = Infallible>
{
    fn num_batch(&self) -> usize;
    fn num_layer(&self) -> usize;

    /// Extract the embedding from a given layer of the state.
    fn embed(&self, batch: usize, layer: usize) -> Vec<f32>;
}

pub trait ModelState: for<'a> FromBuilder<Builder<'a> = StateBuilder, Error = Infallible> {
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

pub trait Model:
    ModelBase
    + ModelSoftmax
    + ModelRun
    + for<'a> FromBuilder<Builder<'a> = ModelBuilder<'a>, Error = anyhow::Error>
{
}

impl<M> Model for M where
    M: ModelBase
        + ModelSoftmax
        + ModelRun
        + for<'a> FromBuilder<Builder<'a> = ModelBuilder<'a>, Error = anyhow::Error>
{
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

#[wasm_bindgen]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedDevice {
    #[default]
    Cpu,
    Gpu,
}

pub struct ModelBuilder<'a> {
    context: Context,
    model: &'a dyn Reader,
    lora: Vec<Lora<'a>>,
    quant: HashMap<usize, Quant>,
    embed_device: EmbedDevice,
    turbo: bool,
    token_chunk_size: usize,
}

struct PreparedModelBuilder<'a> {
    context: Context,
    info: ModelInfo,
    loader: Loader<'a>,
    quant: HashMap<usize, Quant>,
    embed_device: EmbedDevice,
    turbo: bool,
    token_chunk_size: usize,
}

impl<'a> ModelBuilder<'a> {
    pub fn new(context: &Context, model: &'a dyn Reader) -> Self {
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

    async fn prepare(self) -> Result<PreparedModelBuilder<'a>> {
        let ModelBuilder {
            context,
            model,
            lora,
            quant,
            embed_device,
            turbo,
            token_chunk_size,
        } = self;

        let info = Loader::info(model).await?;
        let loader = Loader::new(&context, model, lora);

        let token_chunk_size = token_chunk_size
            .max(MIN_TOKEN_CHUNK_SIZE)
            .next_power_of_two();
        log::info!("token chunk size: {token_chunk_size}");

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

    pub fn with_quant(mut self, value: HashMap<usize, Quant>) -> Self {
        self.quant = value;
        self
    }

    pub fn add_lora(mut self, value: Lora<'a>) -> Self {
        self.lora.push(value);
        self
    }

    pub fn with_embed_device(mut self, value: EmbedDevice) -> Self {
        self.embed_device = value;
        self
    }

    pub fn with_turbo(mut self, value: bool) -> Self {
        self.turbo = value;
        self
    }

    pub fn with_token_chunk_size(mut self, value: usize) -> Self {
        self.token_chunk_size = value;
        self
    }

    pub async fn build<M>(self) -> Result<M>
    where
        M: ModelBase + FromBuilder<Builder<'a> = Self, Error = anyhow::Error>,
    {
        M::from_builder(self).await
    }
}

/// Create a model state.
/// - `num_batch`: The maximum number of runtime slots.
/// - `chunk_size`: Internally, the state is split into chunks of layers, since there is a size limit on one GPU buffer (128 MB).
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

    pub async fn build<S: ModelState>(self) -> S {
        S::from_builder(self).await.expect("build model state")
    }

    pub async fn build_backed<B: BackedState>(self) -> B {
        B::from_builder(self).await.expect("build backed state")
    }
}
