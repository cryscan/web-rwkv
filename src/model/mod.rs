use anyhow::Result;

use crate::tensor::TensorError;

pub mod v4;
pub mod v5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelVersion {
    V4,
    V5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelError {
    BatchSize(usize, usize),
    BatchOutOfRange { batch: usize, max: usize },
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::BatchSize(lhs, rhs) => {
                write!(f, "input batch size {lhs} not match {rhs}")
            }
            ModelError::BatchOutOfRange { batch, max } => {
                write!(f, "batch {batch} out of range of max {max}")
            }
        }
    }
}

impl std::error::Error for ModelError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelInfo {
    pub version: ModelVersion,
    pub num_layers: usize,
    pub num_emb: usize,
    pub num_hidden: usize,
    pub num_vocab: usize,
    pub head_size: usize,
}

pub trait BackedStateExt: Sized {
    fn max_batch(&self) -> usize;
}

pub trait ModelStateExt {
    type BackedState: BackedStateExt;

    fn max_batch(&self) -> usize;

    /// Load the state from host. Their shapes must match.
    fn load(&self, backed: &Self::BackedState) -> Result<()>;
    /// Load one batch from host. The shape of the backed state should be of one batch.
    fn load_batch(&self, backed: &Self::BackedState, batch: usize) -> Result<()>;
    /// Back the entire device state to host.
    fn back(&self) -> Self::BackedState;
    /// Back one batch of the device state to host.
    fn back_batch(&self, batch: usize) -> Result<Self::BackedState>;
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

pub trait ModelExt {
    type ModelState: ModelStateExt;

    fn info(&self) -> &ModelInfo;

    /// Softmax of the input tensors.
    fn softmax(&self, input: Vec<Option<Vec<f32>>>) -> Result<Vec<Option<Vec<f32>>>>;

    /// Run the model for a batch of tokens as input.
    /// The length of `tokens` must match the number of batches in `state`.
    /// `tokens` may have slots with no tokens, for which `run` won't compute that batch and will return an empty vector in that corresponding slot.
    fn run(
        &self,
        tokens: &mut Vec<Vec<u16>>,
        state: &Self::ModelState,
    ) -> Result<Vec<Option<Vec<f32>>>>;
}
