use half::f16;
use serde::{Deserialize, Serialize};

use super::{JobInfo, JobInput};
use crate::tensor::{TensorCpu, TensorInit};

pub mod rnn;
pub mod vision;

pub use rnn::{
    Rnn, RnnChunk, RnnChunkBatch, RnnInfo, RnnInfoBatch, RnnInput, RnnInputBatch, RnnIter,
    RnnOption, RnnOutput, RnnOutputBatch, RnnRedirect,
};

pub trait Infer: Send + Sync + 'static {
    type Info: JobInfo;
    type Input: JobInput;
    type Output: Send + Sync + 'static;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Token {
    Token(u32),
    Embed(TensorCpu<f16>),
}

impl Default for Token {
    fn default() -> Self {
        Self::Token(0)
    }
}

impl From<u32> for Token {
    fn from(value: u32) -> Self {
        Self::Token(value)
    }
}

impl From<Vec<f16>> for Token {
    fn from(value: Vec<f16>) -> Self {
        Self::Embed(TensorCpu::from_data_1d(value))
    }
}

impl From<Vec<f32>> for Token {
    fn from(value: Vec<f32>) -> Self {
        let value: Vec<_> = value.into_iter().map(f16::from_f32).collect();
        Self::Embed(TensorCpu::from_data_1d(value))
    }
}
