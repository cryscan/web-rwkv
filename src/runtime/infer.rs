use itertools::Itertools;
use web_rwkv_derive::{Deref, DerefMut};

use super::JobInput;
use crate::tensor::TensorCpu;

pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;

#[derive(Debug, Clone, Deref, DerefMut, PartialEq, Eq)]
pub struct InferInfo(pub Vec<InferInfoBatch>);

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InferInfoBatch {
    pub len: usize,
    pub option: Option<InferOption>,
}

impl InferInfo {
    #[inline]
    pub fn num_token(&self) -> usize {
        self.0.iter().map(|x| x.len).sum()
    }

    #[inline]
    pub fn num_batch(&self) -> usize {
        self.0.len()
    }

    pub fn redirect(&self) -> InferRedirect {
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
                Some(InferOption::Last) => {
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
                Some(InferOption::Full) => {
                    inputs[batch] = (p_in, p_in + len);
                    outputs[batch] = (p_out, p_out + len);
                    headers.append(&mut (p_in..p_in + len).collect());
                    p_out += len;
                    p_in += len;
                }
            }
        }
        InferRedirect {
            headers,
            inputs,
            outputs,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InferRedirect {
    /// Indices in the *input* tensor that are included in the output.
    pub headers: Vec<usize>,
    /// Maps batches to ranges in the *input* tensor.
    pub inputs: Vec<(usize, usize)>,
    /// Maps batches to ranges in the *output* tensor.
    pub outputs: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
enum BatchState {
    Gen,
    Read(usize),
}

/// Inference option for outputs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum InferOption {
    /// Only output the prediction for the last token.
    #[default]
    Last,
    /// Output predictions for all tokens.
    Full,
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct InferChunk(pub Vec<InferChunkBatch>);

impl InferChunk {
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
pub struct InferChunkBatch(pub Vec<u16>);

/// One batch of the input task.
#[derive(Debug, Default, Clone)]
pub struct InferInputBatch {
    /// Tokens to infer. If this is empty, inference won't occur for the batch.
    pub tokens: Vec<u16>,
    /// Inference option for outputs.
    pub option: InferOption,
}

#[derive(Debug, Clone)]
pub struct InferInput {
    pub batches: Vec<InferInputBatch>,
    token_chunk_size: usize,
}

impl InferInput {
    pub fn new(batches: Vec<InferInputBatch>, token_chunk_size: usize) -> Self {
        let token_chunk_size = token_chunk_size
            .max(MIN_TOKEN_CHUNK_SIZE)
            .next_multiple_of(MIN_TOKEN_CHUNK_SIZE);
        Self {
            batches,
            token_chunk_size,
        }
    }

    #[inline]
    pub fn iter(&self) -> InferIter {
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

impl JobInput for InferInput {
    type Chunk = InferChunk;

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
            return InferChunk(vec![Default::default(); self.batches.len()]);
        };
        let chunk = self
            .batches
            .iter()
            .zip_eq(info.0)
            .map(|(batch, info)| InferChunkBatch(batch.tokens[..info.len].to_vec()))
            .collect();
        InferChunk(chunk)
    }
}

impl IntoIterator for &InferInput {
    type Item = InferInfo;
    type IntoIter = InferIter;

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

pub struct InferIter {
    batches: Vec<(BatchState, InferOption)>,
    token_chunk_size: usize,
}

impl Iterator for InferIter {
    type Item = InferInfo;

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

        let mut info = vec![InferInfoBatch::default(); num_batch];
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
                (InferOption::Last, 0) => Some(InferOption::Last),
                (InferOption::Last, _) => None,
                (InferOption::Full, _) => Some(InferOption::Full),
            };
        }

        Some(InferInfo(info))
    }
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct InferOutputBatch(pub TensorCpu<f32>);

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct InferOutput(pub Vec<InferOutputBatch>);

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::{InferInfo, InferInput, InferOption};
    use crate::runtime::{
        infer::{InferInfoBatch, InferInputBatch},
        JobInput,
    };

    impl From<(usize, Option<InferOption>)> for InferInfoBatch {
        fn from((len, option): (usize, Option<InferOption>)) -> Self {
            Self { len, option }
        }
    }

    #[test]
    fn test_run_iter() -> Result<()> {
        let run = InferInput {
            batches: [
                (vec![0; 139], InferOption::Last),
                (vec![1; 1], InferOption::Last),
                (vec![2; 0], InferOption::Full),
                (vec![3; 65], InferOption::Full),
            ]
            .map(|(tokens, option)| InferInputBatch { tokens, option })
            .to_vec(),
            token_chunk_size: 128,
        };
        let mut iter = run.iter();

        assert_eq!(
            iter.next(),
            Some(InferInfo(
                [
                    (65, None),
                    (1, Some(InferOption::Last)),
                    (0, Some(InferOption::Full)),
                    (62, Some(InferOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );
        assert_eq!(
            iter.next(),
            Some(InferInfo(
                [
                    (60, None),
                    (1, Some(InferOption::Last)),
                    (0, Some(InferOption::Full)),
                    (3, Some(InferOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );
        assert_eq!(
            iter.next(),
            Some(InferInfo(
                [
                    (14, Some(InferOption::Last)),
                    (1, Some(InferOption::Last)),
                    (0, Some(InferOption::Full)),
                    (1, Some(InferOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );
        assert_eq!(
            iter.next(),
            Some(InferInfo(
                [
                    (1, Some(InferOption::Last)),
                    (1, Some(InferOption::Last)),
                    (0, Some(InferOption::Full)),
                    (1, Some(InferOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );
        assert_eq!(
            iter.next(),
            Some(InferInfo(
                [
                    (1, Some(InferOption::Last)),
                    (1, Some(InferOption::Last)),
                    (0, Some(InferOption::Full)),
                    (1, Some(InferOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );

        Ok(())
    }

    #[test]
    fn test_advance() -> Result<()> {
        let mut run = InferInput {
            batches: [
                (vec![0; 139], InferOption::Last),
                (vec![1; 1], InferOption::Last),
                (vec![2; 0], InferOption::Full),
                (vec![3; 65], InferOption::Full),
            ]
            .map(|(tokens, option)| InferInputBatch { tokens, option })
            .to_vec(),
            token_chunk_size: 128,
        };

        run.step();
        assert_eq!(
            run.iter().next(),
            Some(InferInfo(
                [
                    (61, None),
                    (0, Some(InferOption::Last)),
                    (0, Some(InferOption::Full)),
                    (3, Some(InferOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );

        // simulate adding one token to batch 1 after advancing.
        let run = InferInput {
            batches: [
                (vec![0; 61], InferOption::Last),
                (vec![1; 1], InferOption::Last),
                (vec![2; 0], InferOption::Full),
                (vec![3; 3], InferOption::Full),
            ]
            .map(|(tokens, option)| InferInputBatch {
                tokens,
                option,
                ..Default::default()
            })
            .to_vec(),
            token_chunk_size: 128,
        };
        assert_eq!(
            run.iter().next(),
            Some(InferInfo(
                [
                    (60, None),
                    (1, Some(InferOption::Last)),
                    (0, Some(InferOption::Full)),
                    (3, Some(InferOption::Full))
                ]
                .map(Into::into)
                .to_vec()
            ))
        );

        Ok(())
    }

    #[test]
    fn test_redirect() -> Result<()> {
        let run = InferInput {
            batches: [
                (vec![0; 61], InferOption::Last),
                (vec![1; 0], InferOption::Last),
                (vec![2; 0], InferOption::Full),
                (vec![3; 3], InferOption::Full),
            ]
            .map(|(tokens, option)| InferInputBatch {
                tokens,
                option,
                ..Default::default()
            })
            .to_vec(),
            token_chunk_size: 128,
        };
        let redirect = run.iter().next().unwrap().redirect();

        assert_eq!(redirect.headers, vec![60, 61, 62, 63]);
        assert_eq!(redirect.inputs, vec![(0, 61), (61, 61), (61, 61), (61, 64)]);
        assert_eq!(redirect.outputs, vec![(0, 1), (1, 1), (1, 1), (1, 4)]);

        let run = InferInput {
            batches: [
                (vec![0; 11], InferOption::Last),
                (vec![1; 8], InferOption::Last),
                (vec![2; 9], InferOption::Last),
                (vec![3; 4], InferOption::Last),
                (vec![0; 11], InferOption::Last),
                (vec![1; 8], InferOption::Last),
                (vec![2; 9], InferOption::Last),
                (vec![3; 4], InferOption::Last),
            ]
            .map(|(tokens, option)| InferInputBatch {
                tokens,
                option,
                ..Default::default()
            })
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
