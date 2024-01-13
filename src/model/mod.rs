use std::{collections::HashMap, convert::Infallible, future::Future};

use anyhow::Result;
use half::f16;
use regex::Regex;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::{Deref, DerefMut};

use self::{loader::Loader, run::ModelRun, softmax::ModelSoftmax};
use crate::{context::Context, num::Scalar, tensor::TensorError};

pub mod loader;
pub mod run;
pub mod softmax;
pub mod v4;
pub mod v5;
pub mod v6;

pub const RESCALE_LAYER: usize = 6;
pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelVersion {
    V4,
    V5,
    V6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelError {
    InvalidVersion,
    NoViableChunkSize,
    BatchSize(usize, usize),
    BatchOutOfRange { batch: usize, max: usize },
    EmptyInput,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::InvalidVersion => write!(f, "invalid model version"),
            ModelError::NoViableChunkSize => write!(f, "no viable chunk size found"),
            ModelError::BatchSize(lhs, rhs) => write!(f, "input batch size {lhs} not match {rhs}"),
            ModelError::BatchOutOfRange { batch, max } => {
                write!(f, "batch {batch} out of range of max {max}")
            }
            ModelError::EmptyInput => write!(f, "input is empty"),
        }
    }
}

impl std::error::Error for ModelError {}

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
    /// Computes the required storage buffer size, not including head.
    pub fn max_non_head_buffer_size(&self) -> usize {
        (self.num_emb * self.num_hidden * f16::size()).max(256 << 20)
    }

    /// Computes the required storage buffer size, including head.
    pub fn max_buffer_size(&self) -> usize {
        (self.num_emb * self.num_vocab * f16::size()).max(256 << 20)
    }
}

pub trait FromBuilder: Sized {
    type Builder<'a>;
    type Error;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error>;
}

pub trait BackedState:
    Send + for<'a> FromBuilder<Builder<'a> = StateBuilder, Error = Infallible>
{
    fn max_batch(&self) -> usize;
    fn num_layer(&self) -> usize;

    /// Extract the embedding from a given layer of the state.
    fn embed(&self, batch: usize, layer: usize) -> Vec<f32>;
}

pub trait ModelState:
    Sync + for<'a> FromBuilder<Builder<'a> = StateBuilder, Error = Infallible>
{
    type BackedState: BackedState;

    fn max_batch(&self) -> usize;

    /// Load the state from host. Their shapes must match.
    fn load(&self, backed: &Self::BackedState) -> Result<()>;
    /// Load one batch from host. The batch size the backed state should be 1.
    fn load_batch(&self, backed: &Self::BackedState, batch: usize) -> Result<()>;
    /// Back the entire device state to host.
    fn back(&self) -> impl Future<Output = Self::BackedState> + Send;
    /// Back one batch of the device state to host.
    fn back_batch(&self, batch: usize) -> impl Future<Output = Result<Self::BackedState>> + Send;
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
    type ModelTensor;

    fn context(&self) -> &Context;
    fn info(&self) -> &ModelInfo;

    fn tensor(&self) -> &Self::ModelTensor;
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

/// A LoRA that adds to the model when loading.
#[derive(Debug, Clone)]
pub struct Lora {
    /// Binary safetensors LoRA content.
    pub data: Vec<u8>,
    /// A list of LoRA blend patterns.
    /// A blend pattern is a regex that matches the name of multiple tensors, and a blend factor.
    /// When applying the patterns, they are applied in order.
    pub blend: LoraBlend,
}

/// A list of LoRA blend patterns.
#[derive(Debug, Clone, Deref, DerefMut)]
pub struct LoraBlend(pub Vec<LoraBlendPattern>);

impl LoraBlend {
    /// Build a blend pattern that matches all tensors.
    pub fn full(alpha: f32) -> Self {
        let pattern = LoraBlendPattern::new(r".+", alpha).expect("default blend pattern");
        Self(vec![pattern])
    }
}

impl Default for LoraBlend {
    fn default() -> Self {
        Self::full(1.0)
    }
}

/// A blend pattern is a regex that matches the name of multiple tensors, and a blend factor.
#[derive(Debug, Clone)]
pub struct LoraBlendPattern {
    /// A regex pattern that matches tensors in the model.
    pattern: Regex,
    /// The blend factor.
    alpha: f32,
}

impl LoraBlendPattern {
    #[inline]
    pub fn new(pattern: &str, alpha: f32) -> Result<Self> {
        Ok(Self {
            pattern: Regex::new(pattern)?,
            alpha,
        })
    }

    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedDevice {
    #[default]
    Cpu,
    Gpu,
}

pub struct ModelBuilder<'a> {
    context: Context,
    data: &'a [u8],
    lora: Vec<Lora>,
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
    pub fn new(context: &Context, data: &'a [u8]) -> Self {
        Self {
            context: context.clone(),
            data,
            lora: vec![],
            quant: Default::default(),
            turbo: false,
            embed_device: Default::default(),
            token_chunk_size: 32,
        }
    }

    fn prepare(self) -> Result<PreparedModelBuilder<'a>> {
        let ModelBuilder {
            context,
            data,
            lora,
            quant,
            embed_device,
            turbo,
            token_chunk_size,
        } = self;

        let loader = Loader::new(&context, data, lora)?;
        let info = Loader::info(data)?;

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

    pub fn with_quant(mut self, quant: HashMap<usize, Quant>) -> Self {
        self.quant = quant;
        self
    }

    pub fn add_lora(mut self, lora: Lora) -> Self {
        self.lora.push(lora);
        self
    }

    pub fn with_embed_device(mut self, embed_device: EmbedDevice) -> Self {
        self.embed_device = embed_device;
        self
    }

    pub fn with_turbo(mut self, turbo: bool) -> Self {
        self.turbo = turbo;
        self
    }

    pub fn with_token_chunk_size(mut self, token_chunk_size: usize) -> Self {
        self.token_chunk_size = token_chunk_size;
        self
    }

    pub fn build<M>(self) -> Result<M>
    where
        M: ModelBase + FromBuilder<Builder<'a> = Self, Error = anyhow::Error>,
    {
        M::from_builder(self)
    }
}

/// Create a model state.
/// - `max_batch`: The maximum number of runtime slots.
/// - `chunk_size`: Internally, the state is split into chunks of layers, since there is a size limit on one GPU buffer (128 MB).
/// If there is only one batch, it is recommended to set `chunk_size` to `info.num_layers()`.
pub struct StateBuilder {
    context: Context,
    info: ModelInfo,
    max_batch: usize,
    chunk_size: usize,
}

impl StateBuilder {
    pub fn new(context: &Context, info: &ModelInfo) -> Self {
        Self {
            context: context.clone(),
            info: info.clone(),
            max_batch: 1,
            chunk_size: info.num_layer,
        }
    }

    pub fn with_max_batch(mut self, max_batch: usize) -> Self {
        self.max_batch = max_batch;
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    pub fn build<S: ModelState>(self) -> S {
        S::from_builder(self).expect("build model state")
    }

    pub fn build_backed<B: BackedState>(self) -> B {
        B::from_builder(self).expect("build backed state")
    }
}
