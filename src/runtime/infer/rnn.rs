use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::{Deref, DerefMut};

use crate::{
    num::Float,
    runtime::{JobInfo, JobInput},
    tensor::{kind::ReadWrite, ops::TensorOp, TensorCpu, TensorError, TensorGpu, TensorShape},
};

pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rnn;

impl super::Infer for Rnn {
    type Info = RnnInfo;
    type Input = RnnInput;
    type Output = RnnOutput;
}

#[derive(Debug, Clone, PartialEq, Eq, Deref, DerefMut)]
pub struct RnnInfo(pub Vec<RnnInfoBatch>);

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RnnInfoBatch {
    pub len: usize,
    pub option: Option<RnnOption>,
}

impl RnnInfo {
    #[inline]
    pub fn num_token(&self) -> usize {
        self.0.iter().map(|x| x.len).sum()
    }

    #[inline]
    pub fn num_batch(&self) -> usize {
        self.0.len()
    }

    pub fn redirect(&self) -> RnnRedirect {
        let mut headers = vec![];
        let mut inputs = vec![(0, 0); self.num_batch()];
        let mut outputs = vec![(0, 0); self.num_batch()];
        let mut p_in = 0;
        let mut p_out = 0;
        for (batch, info) in self.0.iter().enumerate() {
            let len = info.len;
            match &info.option {
                None => {
                    inputs[batch] = (p_in, p_in + len);
                    outputs[batch] = (p_out, p_out);
                    p_in += len;
                }
                Some(RnnOption::Last) => {
                    inputs[batch] = (p_in, p_in + len);
                    match len {
                        0 => outputs[batch] = (p_out, p_out),
                        _ => {
                            outputs[batch] = (p_out, p_out + 1);
                            headers.push(p_in + len - 1);
                            p_out += 1;
                        }
                    }
                    p_in += len;
                }
                Some(RnnOption::Full) => {
                    inputs[batch] = (p_in, p_in + len);
                    outputs[batch] = (p_out, p_out + len);
                    headers.append(&mut (p_in..p_in + len).collect());
                    p_out += len;
                    p_in += len;
                }
            }
        }
        RnnRedirect {
            headers,
            inputs,
            outputs,
        }
    }
}

impl JobInfo for RnnInfo {
    #[inline]
    fn check(&self, info: &Self) -> bool {
        self.num_token() == info.num_token() && self.redirect() == info.redirect()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RnnRedirect {
    /// Indices in the *input* tensor that are included in the output.
    pub headers: Vec<usize>,
    /// Maps batches to ranges in the *input* tensor.
    pub inputs: Vec<(usize, usize)>,
    /// Maps batches to ranges in the *output* tensor.
    pub outputs: Vec<(usize, usize)>,
}

impl RnnRedirect {
    pub fn op<F: Float>(
        &self,
        input: &TensorGpu<F, ReadWrite>,
        output: &TensorGpu<F, ReadWrite>,
    ) -> Result<(TensorOp, TensorGpu<F, ReadWrite>), TensorError> {
        let headers = &self.headers;
        let num_token = input.shape()[1];
        let num_header = headers.len();

        if num_token == 1 || num_token == num_header {
            Ok((TensorOp::empty(), input.clone()))
        } else {
            let mut start = 0;
            let mut end = 1;
            let mut ops = vec![];
            while end <= headers.len() {
                if end == headers.len() || headers[end - 1] + 1 != headers[end] {
                    let first = headers[start];
                    let last = headers[end - 1];
                    assert_eq!(last - first + 1, end - start);

                    let input = input.view(.., first..=last, .., ..)?;
                    let output = output.view(.., start..end, .., ..)?;
                    ops.push(TensorOp::blit(input, output)?);

                    start = end;
                }
                end += 1;
            }
            Ok((TensorOp::List(ops), output.clone()))
        }
    }
}

/// Inference option for outputs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum RnnOption {
    /// Only output the prediction for the last token.
    #[default]
    Last,
    /// Output predictions for all tokens.
    Full,
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct RnnChunk(pub Vec<RnnChunkBatch>);

impl RnnChunk {
    #[inline]
    pub fn num_token(&self) -> usize {
        self.0.iter().map(|x| x.0.len()).sum()
    }

    #[inline]
    pub fn num_batch(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Default, Clone, Deref, DerefMut)]
pub struct RnnChunkBatch(pub Vec<u16>);

/// One batch of the input task.
#[derive(Debug, Default, Clone)]
pub struct RnnInputBatch {
    /// Tokens to infer. If this is empty, inference won't occur for the batch.
    pub tokens: Vec<u16>,
    /// Inference option for outputs.
    pub option: RnnOption,
}

#[derive(Debug, Clone)]
pub struct RnnInput {
    pub batches: Vec<RnnInputBatch>,
    token_chunk_size: usize,
}

impl RnnInput {
    pub fn new(batches: Vec<RnnInputBatch>, token_chunk_size: usize) -> Self {
        let token_chunk_size = token_chunk_size
            .max(MIN_TOKEN_CHUNK_SIZE)
            .next_multiple_of(MIN_TOKEN_CHUNK_SIZE);
        Self {
            batches,
            token_chunk_size,
        }
    }

    #[inline]
    pub fn iter(&self) -> RnnIter {
        self.into_iter()
    }

    #[inline]
    pub fn token_chunk_size(&self) -> usize {
        self.token_chunk_size
    }

    #[inline]
    pub fn num_token(&self) -> usize {
        self.batches.iter().map(|batch| batch.tokens.len()).sum()
    }
}

impl JobInput for RnnInput {
    type Chunk = RnnChunk;

    fn step(&mut self) {
        let Some(info) = self.iter().next() else {
            return;
        };
        for (batch, info) in self.batches.iter_mut().zip_eq(info.0) {
            batch.tokens = batch.tokens.split_off(info.len);
        }
    }

    fn chunk(&self) -> Self::Chunk {
        let Some(info) = self.iter().next() else {
            return RnnChunk(vec![Default::default(); self.batches.len()]);
        };
        let chunk = self
            .batches
            .iter()
            .zip_eq(info.0)
            .map(|(batch, info)| RnnChunkBatch(batch.tokens[..info.len].to_vec()))
            .collect();
        RnnChunk(chunk)
    }
}

impl IntoIterator for &RnnInput {
    type Item = RnnInfo;
    type IntoIter = RnnIter;

    fn into_iter(self) -> Self::IntoIter {
        let batches = self
            .batches
            .iter()
            .map(|batch| (BatchState::Read(batch.tokens.len()), batch.option))
            .collect();
        let token_chunk_size = self.token_chunk_size;
        Self::IntoIter {
            batches,
            token_chunk_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RnnIter {
    batches: Vec<(BatchState, RnnOption)>,
    token_chunk_size: usize,
}

impl Iterator for RnnIter {
    type Item = RnnInfo;

    fn next(&mut self) -> Option<Self::Item> {
        let mut remains = self
            .batches
            .iter()
            .map(|&(x, ..)| match x {
                BatchState::Gen => 1,
                BatchState::Read(x) => x,
            })
            .collect_vec();

        let num_batch = remains.len();
        let num_token: usize = remains.iter().sum();
        let num_token = num_token.min(self.token_chunk_size);
        let mut num_token = match num_token > MIN_TOKEN_CHUNK_SIZE {
            true => num_token - num_token % MIN_TOKEN_CHUNK_SIZE,
            false => num_token,
        };

        let mut info = vec![RnnInfoBatch::default(); num_batch];
        while num_token > 0 {
            let mid = *remains.iter().filter(|&&x| x > 0).min().unwrap_or(&0);
            for (info, batch) in info.iter_mut().zip_eq(remains.iter_mut()) {
                if *batch == 0 {
                    continue;
                }

                let mid = mid.min(num_token);
                num_token -= mid;

                info.len += mid;
                *batch -= mid;
            }
        }

        for (info, batch, remain) in
            itertools::multizip((info.iter_mut(), self.batches.iter_mut(), remains.iter()))
        {
            if info.len > 0 {
                batch.0 = match remain {
                    0 => BatchState::Gen,
                    &x => BatchState::Read(x),
                };
            }
            info.option = match (batch.1, remain) {
                (RnnOption::Last, 0) => Some(RnnOption::Last),
                (RnnOption::Last, _) => None,
                (RnnOption::Full, _) => Some(RnnOption::Full),
            };
        }

        Some(RnnInfo(info))
    }
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct RnnOutputBatch(pub TensorCpu<f32>);

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct RnnOutput(pub Vec<RnnOutputBatch>);

#[derive(Debug, Clone, Copy)]
enum BatchState {
    Gen,
    Read(usize),
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::{RnnInfo, RnnInfoBatch, RnnInput, RnnInputBatch, RnnOption};
    use crate::runtime::JobInput;

    impl From<(usize, Option<RnnOption>)> for RnnInfoBatch {
        fn from((len, option): (usize, Option<RnnOption>)) -> Self {
            Self { len, option }
        }
    }

    #[test]
    fn test_run_iter() -> Result<()> {
        let run = RnnInput {
            batches: [
                (vec![0; 139], RnnOption::Last),
                (vec![1; 1], RnnOption::Last),
                (vec![2; 0], RnnOption::Full),
                (vec![3; 65], RnnOption::Full),
            ]
            .map(|(tokens, option)| RnnInputBatch { tokens, option })
            .to_vec(),
            token_chunk_size: 128,
        };
        let mut iter = run.iter();

        assert_eq!(
            iter.next(),
            Some(RnnInfo(
                [
                    (65, None),
                    (1, Some(RnnOption::Last)),
                    (0, Some(RnnOption::Full)),
                    (62, Some(RnnOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );
        assert_eq!(
            iter.next(),
            Some(RnnInfo(
                [
                    (60, None),
                    (1, Some(RnnOption::Last)),
                    (0, Some(RnnOption::Full)),
                    (3, Some(RnnOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );
        assert_eq!(
            iter.next(),
            Some(RnnInfo(
                [
                    (14, Some(RnnOption::Last)),
                    (1, Some(RnnOption::Last)),
                    (0, Some(RnnOption::Full)),
                    (1, Some(RnnOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );
        assert_eq!(
            iter.next(),
            Some(RnnInfo(
                [
                    (1, Some(RnnOption::Last)),
                    (1, Some(RnnOption::Last)),
                    (0, Some(RnnOption::Full)),
                    (1, Some(RnnOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );
        assert_eq!(
            iter.next(),
            Some(RnnInfo(
                [
                    (1, Some(RnnOption::Last)),
                    (1, Some(RnnOption::Last)),
                    (0, Some(RnnOption::Full)),
                    (1, Some(RnnOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );

        Ok(())
    }

    #[test]
    fn test_advance() -> Result<()> {
        let mut run = RnnInput {
            batches: [
                (vec![0; 139], RnnOption::Last),
                (vec![1; 1], RnnOption::Last),
                (vec![2; 0], RnnOption::Full),
                (vec![3; 65], RnnOption::Full),
            ]
            .map(|(tokens, option)| RnnInputBatch { tokens, option })
            .to_vec(),
            token_chunk_size: 128,
        };

        run.step();
        assert_eq!(
            run.iter().next(),
            Some(RnnInfo(
                [
                    (61, None),
                    (0, Some(RnnOption::Last)),
                    (0, Some(RnnOption::Full)),
                    (3, Some(RnnOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );

        // simulate adding one token to batch 1 after advancing.
        let run = RnnInput {
            batches: [
                (vec![0; 61], RnnOption::Last),
                (vec![1; 1], RnnOption::Last),
                (vec![2; 0], RnnOption::Full),
                (vec![3; 3], RnnOption::Full),
            ]
            .map(|(tokens, option)| RnnInputBatch { tokens, option })
            .to_vec(),
            token_chunk_size: 128,
        };
        assert_eq!(
            run.iter().next(),
            Some(RnnInfo(
                [
                    (60, None),
                    (1, Some(RnnOption::Last)),
                    (0, Some(RnnOption::Full)),
                    (3, Some(RnnOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );

        Ok(())
    }

    #[test]
    fn test_redirect() -> Result<()> {
        let run = RnnInput {
            batches: [
                (vec![0; 61], RnnOption::Last),
                (vec![1; 0], RnnOption::Last),
                (vec![2; 0], RnnOption::Full),
                (vec![3; 3], RnnOption::Full),
            ]
            .map(|(tokens, option)| RnnInputBatch { tokens, option })
            .to_vec(),
            token_chunk_size: 128,
        };
        let redirect = run.iter().next().unwrap().redirect();

        assert_eq!(redirect.headers, vec![60, 61, 62, 63]);
        assert_eq!(redirect.inputs, vec![(0, 61), (61, 61), (61, 61), (61, 64)]);
        assert_eq!(redirect.outputs, vec![(0, 1), (1, 1), (1, 1), (1, 4)]);

        let run = RnnInput {
            batches: [
                (vec![0; 11], RnnOption::Last),
                (vec![1; 8], RnnOption::Last),
                (vec![2; 9], RnnOption::Last),
                (vec![3; 4], RnnOption::Last),
                (vec![0; 11], RnnOption::Last),
                (vec![1; 8], RnnOption::Last),
                (vec![2; 9], RnnOption::Last),
                (vec![3; 4], RnnOption::Last),
            ]
            .map(|(tokens, option)| RnnInputBatch { tokens, option })
            .to_vec(),
            token_chunk_size: 32,
        };
        let redirect = run.iter().next().unwrap().redirect();

        assert_eq!(redirect.headers, vec![15, 31]);
        assert_eq!(
            redirect.inputs,
            vec![
                (0, 4),
                (4, 8),
                (8, 12),
                (12, 16),
                (16, 20),
                (20, 24),
                (24, 28),
                (28, 32)
            ]
        );
        assert_eq!(
            redirect.outputs,
            vec![
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 2)
            ]
        );

        Ok(())
    }
}
