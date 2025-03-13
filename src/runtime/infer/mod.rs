use super::{JobInfo, JobInput};

pub mod rnn;

pub use rnn::{
    Rnn, RnnChunk, RnnChunkBatch, RnnInfo, RnnInfoBatch, RnnInput, RnnInputBatch, RnnIter,
    RnnOption, RnnOutput, RnnOutputBatch, RnnRedirect,
};

pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;

pub trait Infer: Send + Sync + 'static {
    type Info: JobInfo;
    type Input: JobInput;
    type Output: Send + Sync + 'static;
}
