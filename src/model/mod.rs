use std::{collections::HashMap, convert::Infallible};

use anyhow::Result;
use async_trait::async_trait;
use half::f16;
use regex::Regex;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::{Deref, DerefMut};

use crate::{
    context::Context,
    num::Scalar,
    tensor::{shape::Shape, TensorError},
};

use self::loader::Loader;

pub mod loader;
pub mod matrix;
pub mod run;
pub mod softmax;
pub mod v4;
pub mod v5;
pub mod v6;

pub const RESCALE_LAYER: usize = 6;

pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;
pub const HEAD_CHUNK_SIZES: [usize; 8] = [
    0x4000, 0x3000, 0x2000, 0x1800, 0x1600, 0x1400, 0x1200, 0x1000,
];

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
    /// Computes the required storage buffer size.
    pub fn max_buffer_size(&self) -> usize {
        (self.num_emb * self.num_hidden * f16::size()).max(128 << 20)
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

#[async_trait]
pub trait ModelState:
    Sync + for<'a> FromBuilder<Builder<'a> = StateBuilder, Error = Infallible>
{
    type BackedState: BackedState;

    fn context(&self) -> &Context;
    fn max_batch(&self) -> usize;

    /// Load the state from host. Their shapes must match.
    fn load(&self, backed: &Self::BackedState) -> Result<()>;
    /// Load one batch from host. The batch size the backed state should be 1.
    fn load_batch(&self, backed: &Self::BackedState, batch: usize) -> Result<()>;
    /// Back the entire device state to host.
    async fn back(&self) -> Self::BackedState;
    /// Back one batch of the device state to host.
    async fn back_batch(&self, batch: usize) -> Result<Self::BackedState>;
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

pub trait ModelBase: Sync {
    type ModelState: ModelState;

    fn context(&self) -> &Context;
    fn info(&self) -> &ModelInfo;

    fn token_chunk_size(&self) -> usize;
    fn head_chunk_size(&self) -> usize;

    fn head_shape(&self, num_batch: usize) -> Shape;
}

pub trait Model:
    ModelBase
    + softmax::ModelSoftmax
    + run::ModelRun
    + for<'a> FromBuilder<Builder<'a> = ModelBuilder<'a>, Error = anyhow::Error>
{
}

impl<S: ModelState, M> Model for M where
    M: ModelBase<ModelState = S>
        + softmax::ModelSoftmax
        + run::ModelRun
        + for<'a> FromBuilder<Builder<'a> = ModelBuilder<'a>, Error = anyhow::Error>
{
}

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

#[derive(Debug, Clone)]
pub struct Lora {
    pub data: Vec<u8>,
    pub blend: LoraBlend,
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct LoraBlend(pub Vec<LoraBlendPattern>);

impl LoraBlend {
    pub fn full(alpha: f32) -> Self {
        let pattern = LoraBlendPattern::new(r"blocks\.[0-9]+\.([0-9a-zA-Z\.\_]+)", alpha)
            .expect("default blend pattern");
        Self(vec![pattern])
    }
}

impl Default for LoraBlend {
    fn default() -> Self {
        Self::full(1.0)
    }
}

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

pub struct ModelBuilder<'a> {
    context: Context,
    data: &'a [u8],
    lora: Vec<Lora>,
    quant: HashMap<usize, Quant>,
    turbo: bool,
    token_chunk_size: usize,
}

struct PreparedModelBuilder<'a> {
    context: Context,
    info: ModelInfo,
    loader: Loader<'a>,
    quant: HashMap<usize, Quant>,
    turbo: bool,
    rescale: bool,
    token_chunk_size: usize,
    head_chunk_size: usize,
}

impl<'a> ModelBuilder<'a> {
    pub fn new(context: &Context, data: &'a [u8]) -> Self {
        Self {
            context: context.clone(),
            data,
            lora: vec![],
            quant: Default::default(),
            turbo: false,
            token_chunk_size: 32,
        }
    }

    fn prepare(self) -> Result<PreparedModelBuilder<'a>> {
        let ModelBuilder {
            context,
            data,
            lora,
            quant,
            turbo,
            token_chunk_size,
        } = self;

        let loader = Loader::new(&context, data, lora)?;
        let info = Loader::info(data)?;

        let token_chunk_size = token_chunk_size
            .max(MIN_TOKEN_CHUNK_SIZE)
            .next_power_of_two();
        log::info!("token chunk size: {token_chunk_size}");

        let max_chunk_size = context.device.limits().max_storage_buffer_binding_size as usize;
        let head_chunk_size = HEAD_CHUNK_SIZES
            .into_iter()
            .find(|&x| info.num_emb * x * f16::size() <= max_chunk_size)
            .ok_or(ModelError::NoViableChunkSize)?;
        log::info!("head chunk size: {head_chunk_size}");

        let rescale = turbo || quant.iter().any(|(_, quant)| matches!(quant, Quant::NF4));

        Ok(PreparedModelBuilder {
            context,
            info,
            loader,
            quant,
            turbo,
            rescale,
            token_chunk_size,
            head_chunk_size,
        })
    }

    pub fn with_quant(self, quant: HashMap<usize, Quant>) -> Self {
        Self { quant, ..self }
    }

    pub fn add_lora(mut self, lora: Lora) -> Self {
        self.lora.push(lora);
        self
    }

    pub fn with_turbo(self, turbo: bool) -> Self {
        Self { turbo, ..self }
    }

    pub fn with_token_chunk_size(self, token_chunk_size: usize) -> Self {
        Self {
            token_chunk_size,
            ..self
        }
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

    pub fn with_max_batch(self, value: usize) -> Self {
        Self {
            max_batch: value,
            ..self
        }
    }

    pub fn with_chunk_size(self, value: usize) -> Self {
        Self {
            chunk_size: value,
            ..self
        }
    }

    pub fn build<S>(self) -> S
    where
        S: ModelState,
    {
        S::from_builder(self).expect("build model state")
    }

    pub fn build_backed<B: BackedState>(self) -> B {
        B::from_builder(self).expect("build backed state")
    }
}
