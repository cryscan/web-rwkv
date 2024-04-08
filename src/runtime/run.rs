use itertools::Itertools;
use web_rwkv_derive::{Deref, DerefMut};

use super::JobInput;
use crate::{num::Float, tensor::TensorCpu};

pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunInfo<const N: usize>(pub [(usize, Option<RunOption>); N]);

impl<const N: usize> RunInfo<N> {
    #[inline]
    pub fn num_token(&self) -> usize {
        self.0.iter().map(|(x, _)| x).sum()
    }

    pub fn redirect(&self) -> RunRedirect {
        let mut headers = vec![];
        let mut inputs = vec![(0, 0); N];
        let mut outputs = vec![(0, 0); N];
        let mut p_in = 0;
        let mut p_out = 0;
        for (batch, (len, option)) in self.0.iter().enumerate() {
            match option {
                None => {
                    inputs[batch] = (p_in, p_in);
                    outputs[batch] = (p_out, p_out);
                }
                Some(RunOption::Last) => {
                    assert_ne!(*len, 0);
                    inputs[batch] = (p_in, p_in + len);
                    outputs[batch] = (p_out, p_out + 1);
                    headers.push(p_in + len - 1);
                    p_out += 1;
                    p_in += len;
                }
                Some(RunOption::Full) => {
                    assert_ne!(*len, 0);
                    inputs[batch] = (p_in, p_in + len);
                    outputs[batch] = (p_out, p_out + len);
                    headers.append(&mut (p_in..p_in + len).collect());
                    p_out += len;
                    p_in += len;
                }
            }
        }
        RunRedirect {
            headers,
            inputs,
            outputs,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RunRedirect {
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum RunOption {
    #[default]
    Last,
    Full,
}

#[derive(Debug, Clone)]
pub struct RunInput<const N: usize> {
    pub batches: [(Vec<u16>, RunOption); N],
    token_chunk_size: usize,
}

impl<const N: usize> RunInput<N> {
    pub fn new(batches: [(Vec<u16>, RunOption); N], token_chunk_size: usize) -> Self {
        let token_chunk_size = token_chunk_size
            .max(MIN_TOKEN_CHUNK_SIZE)
            .next_multiple_of(MIN_TOKEN_CHUNK_SIZE);
        Self {
            batches,
            token_chunk_size,
        }
    }

    pub fn iter(&self) -> RunIter<N> {
        self.into_iter()
    }
}

impl<const N: usize> JobInput for RunInput<N> {
    type Chunk = Vec<Vec<u16>>;

    fn step(&mut self) {
        let Some(info) = self.iter().next() else {
            return;
        };
        for ((tokens, _), (len, _)) in self.batches.iter_mut().zip_eq(info.0) {
            *tokens = tokens.split_off(len);
        }
    }

    fn chunk(&self) -> Self::Chunk {
        let Some(info) = self.iter().next() else {
            return vec![vec![]; self.batches.len()];
        };
        self.batches
            .iter()
            .zip_eq(info.0)
            .map(|((tokens, _), (len, _))| tokens[..len].to_vec())
            .collect()
    }
}

impl<const N: usize> IntoIterator for &RunInput<N> {
    type Item = RunInfo<N>;
    type IntoIter = RunIter<N>;

    fn into_iter(self) -> Self::IntoIter {
        let batches = self
            .batches
            .clone()
            .map(|(tokens, option)| (BatchState::Read(tokens.len()), option));
        let token_chunk_size = self.token_chunk_size;
        Self::IntoIter {
            batches,
            token_chunk_size,
        }
    }
}

pub struct RunIter<const N: usize> {
    batches: [(BatchState, RunOption); N],
    token_chunk_size: usize,
}

impl<const N: usize> Iterator for RunIter<N> {
    type Item = RunInfo<N>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batches = self.batches.map(|(x, _)| match x {
            BatchState::Gen => 1,
            BatchState::Read(x) => x,
        });

        let num_token: usize = batches.iter().sum();
        let num_token = num_token.min(self.token_chunk_size);
        let mut num_token = match num_token > MIN_TOKEN_CHUNK_SIZE {
            true => num_token - num_token % MIN_TOKEN_CHUNK_SIZE,
            false => num_token,
        };

        if num_token == 0 {
            return None;
        }

        let mut info = [(0, Default::default()); N];
        while num_token > 0 {
            let mid = batches
                .into_iter()
                .filter(|x| *x > 0)
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
                        (RunOption::Last, 0) => Some(RunOption::Last),
                        (RunOption::Last, _) => None,
                        (RunOption::Full, _) => Some(RunOption::Full),
                    };
                }
            },
        );

        Some(RunInfo(info))
    }
}

#[derive(Debug, Default, Clone, Deref, DerefMut)]
pub struct RunOutput<F: Float>(pub Vec<TensorCpu<'static, F>>);

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::{RunInfo, RunInput, RunOption};
    use crate::runtime::JobInput;

    #[test]
    fn test_run_iter() -> Result<()> {
        let run = RunInput {
            batches: [
                (vec![0; 139], RunOption::Last),
                (vec![1; 1], RunOption::Last),
                (vec![2; 0], RunOption::Full),
                (vec![3; 65], RunOption::Full),
            ],
            token_chunk_size: 128,
        };
        let mut iter = run.iter();

        assert_eq!(
            iter.next(),
            Some(RunInfo([
                (65, None),
                (1, Some(RunOption::Last)),
                (0, None),
                (62, Some(RunOption::Full))
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(RunInfo([
                (60, None),
                (1, Some(RunOption::Last)),
                (0, None),
                (3, Some(RunOption::Full))
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(RunInfo([
                (14, Some(RunOption::Last)),
                (1, Some(RunOption::Last)),
                (0, None),
                (1, Some(RunOption::Full))
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(RunInfo([
                (1, Some(RunOption::Last)),
                (1, Some(RunOption::Last)),
                (0, None),
                (1, Some(RunOption::Full))
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(RunInfo([
                (1, Some(RunOption::Last)),
                (1, Some(RunOption::Last)),
                (0, None),
                (1, Some(RunOption::Full))
            ]))
        );

        Ok(())
    }

    #[test]
    fn test_advance() -> Result<()> {
        let mut run = RunInput {
            batches: [
                (vec![0; 139], RunOption::Last),
                (vec![1; 1], RunOption::Last),
                (vec![2; 0], RunOption::Full),
                (vec![3; 65], RunOption::Full),
            ],
            token_chunk_size: 128,
        };

        run.step();
        assert_eq!(
            run.iter().next(),
            Some(RunInfo([
                (61, None),
                (0, None),
                (0, None),
                (3, Some(RunOption::Full))
            ]))
        );

        // simulate adding one token to batch 1 after advancing.
        let run = RunInput {
            batches: [
                (vec![0; 61], RunOption::Last),
                (vec![1; 1], RunOption::Last),
                (vec![2; 0], RunOption::Full),
                (vec![3; 3], RunOption::Full),
            ],
            token_chunk_size: 128,
        };
        assert_eq!(
            run.iter().next(),
            Some(RunInfo([
                (60, None),
                (1, Some(RunOption::Last)),
                (0, None),
                (3, Some(RunOption::Full))
            ]))
        );

        Ok(())
    }

    #[test]
    fn test_redirect() -> Result<()> {
        let run = RunInput {
            batches: [
                (vec![0; 61], RunOption::Last),
                (vec![1; 0], RunOption::Last),
                (vec![2; 0], RunOption::Full),
                (vec![3; 3], RunOption::Full),
            ],
            token_chunk_size: 128,
        };
        let redirect = run.iter().next().unwrap().redirect();

        assert_eq!(redirect.headers, vec![60, 61, 62, 63]);
        assert_eq!(redirect.inputs, vec![(0, 61), (61, 61), (61, 61), (61, 64)]);
        assert_eq!(redirect.outputs, vec![(0, 1), (1, 1), (1, 1), (1, 4)]);

        Ok(())
    }
}
