use half::f16;
use serde::{Deserialize, Serialize};

use super::{JobInfo, JobInput};

pub mod encoder;
pub mod rnn;

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
    Token(u16),
    Embed(Vec<f16>),
}

impl Default for Token {
    fn default() -> Self {
        Self::Token(0)
    }
}

impl From<u16> for Token {
    fn from(value: u16) -> Self {
        Self::Token(value)
    }
}

impl From<Vec<f16>> for Token {
    fn from(value: Vec<f16>) -> Self {
        Self::Embed(value)
    }
}

impl From<Vec<f32>> for Token {
    fn from(value: Vec<f32>) -> Self {
        let value = value.into_iter().map(f16::from_f32).collect();
        Self::Embed(value)
    }
}
