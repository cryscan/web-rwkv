use half::f16;
use serde::{Deserialize, Serialize};

use super::{JobInfo, JobInput};

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

pub trait IntoTokens {
    fn into_tokens(self) -> Vec<Token>;
}

impl IntoTokens for Vec<u16> {
    fn into_tokens(self) -> Vec<Token> {
        self.into_iter().map(Token::Token).collect()
    }
}
