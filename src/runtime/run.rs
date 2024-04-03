use itertools::Itertools;

use crate::{num::Float, tensor::TensorCpu};

pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;

#[derive(Debug, Default, Clone)]
pub struct RunInfo(pub Vec<(usize, Option<RunOption>)>);

#[derive(Debug, Clone, Copy)]
pub enum BatchInput {
    Gen,
    Read(usize),
}

impl BatchInput {
    pub fn num_token(self) -> usize {
        match self {
            BatchInput::Gen => 1,
            BatchInput::Read(x) => x,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum RunOption {
    #[default]
    Last,
    Full,
}

#[derive(Debug, Default, Clone)]
pub struct RunInput {
    pub batches: Vec<(Vec<u16>, RunOption)>,
    pub token_chunk_size: usize,
}

pub struct RunIter {
    batches: Vec<(BatchInput, RunOption)>,
    token_chunk_size: usize,
}

impl RunIter {
    #[inline]
    pub fn num_batch(&self) -> usize {
        self.batches.len()
    }

    #[inline]
    pub fn num_token(&self) -> usize {
        self.batches.iter().map(|(x, _)| x.num_token()).sum()
    }
}

impl Iterator for RunIter {
    type Item = RunInfo;

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_token() == 0 {
            return None;
        }

        let num_batch = self.num_batch();
        let num_token = self.num_token().min(self.token_chunk_size);
        let mut num_token = match num_token > MIN_TOKEN_CHUNK_SIZE {
            true => num_token - num_token % MIN_TOKEN_CHUNK_SIZE,
            false => num_token,
        };

        let mut info = vec![(0, None); num_batch];
        while num_token > 0 {
            let mid = self
                .batches
                .iter()
                .map(|(x, _)| x.num_token())
                .filter(|x| *x > 0)
                .min()
                .unwrap_or_default();
            for (info, batch) in info.iter_mut().zip_eq(self.batches.iter_mut()) {
                let mid = mid.min(batch.0.num_token()).min(num_token);
                num_token -= mid;

                if mid == 0 {
                    continue;
                }

                info.0 += mid;
                batch.0 = match batch.0 {
                    BatchInput::Gen => BatchInput::Gen,
                    BatchInput::Read(x) => match x - mid {
                        0 => BatchInput::Gen,
                        x => BatchInput::Read(x),
                    },
                };

                info.1 = match batch.1 {
                    RunOption::Last => match batch.0 {
                        BatchInput::Gen => Some(RunOption::Last),
                        _ => None,
                    },
                    RunOption::Full => Some(RunOption::Full),
                }
            }
        }

        Some(RunInfo(info))
    }
}

impl IntoIterator for &RunInput {
    type Item = RunInfo;
    type IntoIter = RunIter;

    fn into_iter(self) -> Self::IntoIter {
        let batches = self
            .batches
            .iter()
            .map(|(tokens, option)| (BatchInput::Read(tokens.len()), *option))
            .collect();
        let token_chunk_size = self.token_chunk_size;
        Self::IntoIter {
            batches,
            token_chunk_size,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RunOutput<F: Float>(pub Vec<TensorCpu<'static, F>>);
