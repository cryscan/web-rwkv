use itertools::Itertools;
use web_rwkv_derive::{Deref, DerefMut};

use super::JobInput;
use crate::{num::Float, tensor::TensorCpu};

pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferInfo(pub Vec<(usize, Option<InferOption>, bool)>);

impl InferInfo {
    #[inline]
    pub fn num_token(&self) -> usize {
        self.0.iter().map(|(x, _, _)| x).sum()
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
        for (batch, (len, option, _)) in self.0.iter().enumerate() {
            match option {
                None => {
                    inputs[batch] = (p_in, p_in);
                    outputs[batch] = (p_out, p_out);
                }
                Some(InferOption::Last) => {
                    assert_ne!(*len, 0);
                    inputs[batch] = (p_in, p_in + len);
                    outputs[batch] = (p_out, p_out + 1);
                    headers.push(p_in + len - 1);
                    p_out += 1;
                    p_in += len;
                }
                Some(InferOption::Full) => {
                    assert_ne!(*len, 0);
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

    #[inline]
    pub fn back(&self) -> Vec<bool> {
        self.0.iter().map(|&(_, _, back)| back).collect()
    }
}

#[derive(Debug, Default, Clone)]
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

/// One batch of the input task.
#[derive(Debug, Default, Clone)]
pub struct InferInputBatch {
    /// Tokens to infer. If this is empty, inference won't occur for the batch.
    pub tokens: Vec<u16>,
    /// Inference option for outputs.
    pub option: InferOption,
    /// Load a state before inference.
    pub load: Option<TensorCpu<f32>>,
    /// Enable reading back the state after inference.
    pub back: bool,
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

#[derive(Debug, Clone)]
pub struct InferInput<const N: usize> {
    pub batches: [InferInputBatch; N],
    token_chunk_size: usize,
}

impl<const N: usize> InferInput<N> {
    pub fn new(batches: [InferInputBatch; N], token_chunk_size: usize) -> Self {
        let token_chunk_size = token_chunk_size
            .max(MIN_TOKEN_CHUNK_SIZE)
            .next_multiple_of(MIN_TOKEN_CHUNK_SIZE);
        Self {
            batches,
            token_chunk_size,
        }
    }

    pub fn iter(&self) -> InferIter {
        self.into_iter()
    }
}

impl<const N: usize> JobInput for InferInput<N> {
    type Chunk = Vec<Vec<u16>>;

    fn step(&mut self) {
        let Some(info) = self.iter().next() else {
            return;
        };
        for (batch, (len, ..)) in self.batches.iter_mut().zip_eq(info.0) {
            batch.tokens = batch.tokens.split_off(len);
        }
    }

    fn chunk(&self) -> Self::Chunk {
        let Some(info) = self.iter().next() else {
            return vec![vec![]; self.batches.len()];
        };
        self.batches
            .iter()
            .zip_eq(info.0)
            .map(|(batch, (len, ..))| batch.tokens[..len].to_vec())
            .collect()
    }
}

impl<const N: usize> IntoIterator for &InferInput<N> {
    type Item = InferInfo;
    type IntoIter = InferIter;

    fn into_iter(self) -> Self::IntoIter {
        let batches = self
            .batches
            .iter()
            .map(|batch| {
                (
                    BatchState::Read(batch.tokens.len()),
                    batch.option,
                    batch.back,
                )
            })
            .collect();
        let token_chunk_size = self.token_chunk_size;
        Self::IntoIter {
            batches,
            token_chunk_size,
        }
    }
}

pub struct InferIter {
    batches: Vec<(BatchState, InferOption, bool)>,
    token_chunk_size: usize,
}

impl Iterator for InferIter {
    type Item = InferInfo;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batches = self
            .batches
            .iter()
            .map(|&(x, ..)| match x {
                BatchState::Gen => 1,
                BatchState::Read(x) => x,
            })
            .collect_vec();

        let num_batch = batches.len();
        let num_token: usize = batches.iter().sum();
        let num_token = num_token.min(self.token_chunk_size);
        let mut num_token = match num_token > MIN_TOKEN_CHUNK_SIZE {
            true => num_token - num_token % MIN_TOKEN_CHUNK_SIZE,
            false => num_token,
        };

        if num_token == 0 {
            return None;
        }

        let mut info = vec![(0, Default::default(), false); num_batch];
        while num_token > 0 {
            let mid = batches
                .clone()
                .into_iter()
                .filter(|&x| x > 0)
                .min()
                .unwrap_or_default();
            for (info, batch) in info.iter_mut().zip_eq(batches.iter_mut()) {
                if *batch == 0 {
                    continue;
                }

                let mid = mid.min(num_token);
                num_token -= mid;

                info.0 += mid;
                *batch -= mid;
            }
        }

        itertools::multizip((info.iter_mut(), self.batches.iter_mut(), batches.iter())).for_each(
            |(info, batch, remain)| {
                if info.0 > 0 {
                    batch.0 = match remain {
                        0 => BatchState::Gen,
                        &x => BatchState::Read(x),
                    };
                    info.1 = match (batch.1, remain) {
                        (InferOption::Last, 0) => Some(InferOption::Last),
                        (InferOption::Last, _) => None,
                        (InferOption::Full, _) => Some(InferOption::Full),
                    };
                    info.2 = batch.2;
                }
            },
        );

        Some(InferInfo(info))
    }
}

#[derive(Debug, Clone)]
pub struct InferOutputBatch<F: Float> {
    pub output: TensorCpu<F>,
    pub state: Option<TensorCpu<f32>>,
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct RunOutput<F: Float>(pub Vec<InferOutputBatch<F>>);

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::{InferInfo, InferInput, InferOption};
    use crate::runtime::{infer::InferInputBatch, JobInput};

    #[test]
    fn test_run_iter() -> Result<()> {
        let run = InferInput {
            batches: [
                (vec![0; 139], InferOption::Last, false),
                (vec![1; 1], InferOption::Last, false),
                (vec![2; 0], InferOption::Full, false),
                (vec![3; 65], InferOption::Full, true),
            ]
            .map(|(tokens, option, back)| InferInputBatch {
                tokens,
                option,
                back,
                ..Default::default()
            }),
            token_chunk_size: 128,
        };
        let mut iter = run.iter();

        assert_eq!(
            iter.next(),
            Some(InferInfo(vec![
                (65, None, false),
                (1, Some(InferOption::Last), false),
                (0, None, false),
                (62, Some(InferOption::Full), true)
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(InferInfo(vec![
                (60, None, false),
                (1, Some(InferOption::Last), false),
                (0, None, false),
                (3, Some(InferOption::Full), true)
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(InferInfo(vec![
                (14, Some(InferOption::Last), false),
                (1, Some(InferOption::Last), false),
                (0, None, false),
                (1, Some(InferOption::Full), true)
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(InferInfo(vec![
                (1, Some(InferOption::Last), false),
                (1, Some(InferOption::Last), false),
                (0, None, false),
                (1, Some(InferOption::Full), true)
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(InferInfo(vec![
                (1, Some(InferOption::Last), false),
                (1, Some(InferOption::Last), false),
                (0, None, false),
                (1, Some(InferOption::Full), true)
            ]))
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
            .map(|(tokens, option)| InferInputBatch {
                tokens,
                option,
                ..Default::default()
            }),
            token_chunk_size: 128,
        };

        run.step();
        assert_eq!(
            run.iter().next(),
            Some(InferInfo(vec![
                (61, None, false),
                (0, None, false),
                (0, None, false),
                (3, Some(InferOption::Full), false)
            ]))
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
            }),
            token_chunk_size: 128,
        };
        assert_eq!(
            run.iter().next(),
            Some(InferInfo(vec![
                (60, None, false),
                (1, Some(InferOption::Last), false),
                (0, None, false),
                (3, Some(InferOption::Full), false)
            ]))
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
            }),
            token_chunk_size: 128,
        };
        let redirect = run.iter().next().unwrap().redirect();

        assert_eq!(redirect.headers, vec![60, 61, 62, 63]);
        assert_eq!(redirect.inputs, vec![(0, 61), (61, 61), (61, 61), (61, 64)]);
        assert_eq!(redirect.outputs, vec![(0, 1), (1, 1), (1, 1), (1, 4)]);

        Ok(())
    }
}
