use itertools::Itertools;

use crate::{num::Float, tensor::TensorCpu};

pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunInfo(pub Vec<(usize, Option<RunOption>)>);

impl RunInfo {
    #[inline]
    pub fn num_batch(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn num_token(&self) -> usize {
        self.0.iter().map(|(x, _)| x).sum()
    }

    pub fn redirect(&self) -> RunRedirect {
        let mut batches = vec![(0, 0); self.num_batch()];
        let mut headers = vec![];
        let mut p = 0;
        for (batch, (len, option)) in self.0.iter().enumerate() {
            match option {
                None => batches[batch] = (p, p),
                Some(RunOption::Last) => {
                    assert_ne!(*len, 0);
                    batches[batch] = (p, p + len);
                    headers.push(p + len - 1);
                    p += len;
                }
                Some(RunOption::Full) => {
                    assert_ne!(*len, 0);
                    batches[batch] = (p, p + len);
                    headers.append(&mut (p..p + len).collect());
                    p += len;
                }
            }
        }
        RunRedirect { headers, batches }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RunRedirect {
    /// Indices in the *input* tensor that are included in the output.
    pub headers: Vec<usize>,
    /// Maps batches to ranges in the *output* tensor.
    pub batches: Vec<(usize, usize)>,
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
pub struct RunInput {
    pub batches: Vec<(Vec<u16>, RunOption)>,
    // pub stack: TensorStack<'static, f16>,
    pub token_chunk_size: usize,
}

impl RunInput {
    pub fn iter(&self) -> RunIter {
        self.into_iter()
    }

    pub fn chunk(&self) -> Vec<Vec<u16>> {
        let Some(info) = self.iter().next() else {
            return vec![];
        };
        self.batches
            .iter()
            .zip_eq(info.0)
            .map(|((tokens, _), (len, _))| tokens[..len].to_vec())
            .collect()
    }

    pub fn advance(mut self) -> Self {
        let Some(info) = self.iter().next() else {
            return self;
        };
        for ((tokens, _), (len, _)) in self.batches.iter_mut().zip_eq(info.0) {
            *tokens = tokens.split_off(len);
        }
        self
    }
}

impl IntoIterator for &RunInput {
    type Item = RunInfo;
    type IntoIter = RunIter;

    fn into_iter(self) -> Self::IntoIter {
        let batches = self
            .batches
            .iter()
            .map(|(tokens, option)| (BatchState::Read(tokens.len()), *option))
            .collect();
        let token_chunk_size = self.token_chunk_size;
        Self::IntoIter {
            batches,
            token_chunk_size,
        }
    }
}

pub struct RunIter {
    batches: Vec<(BatchState, RunOption)>,
    token_chunk_size: usize,
}

impl Iterator for RunIter {
    type Item = RunInfo;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batches = self
            .batches
            .iter()
            .map(|(x, _)| match x {
                BatchState::Gen => 1,
                BatchState::Read(x) => *x,
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

        let mut info = vec![(0, Default::default()); num_batch];
        while num_token > 0 {
            let mid = batches
                .clone()
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

#[derive(Debug, Default, Clone)]
pub struct RunOutput<F: Float>(pub Vec<TensorCpu<'static, F>>);

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::{RunInfo, RunInput, RunOption};

    #[test]
    fn test_run_iter() -> Result<()> {
        let run = RunInput {
            batches: vec![
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
            Some(RunInfo(vec![
                (65, None),
                (1, Some(RunOption::Last)),
                (0, None),
                (62, Some(RunOption::Full))
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(RunInfo(vec![
                (60, None),
                (1, Some(RunOption::Last)),
                (0, None),
                (3, Some(RunOption::Full))
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(RunInfo(vec![
                (14, Some(RunOption::Last)),
                (1, Some(RunOption::Last)),
                (0, None),
                (1, Some(RunOption::Full))
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(RunInfo(vec![
                (1, Some(RunOption::Last)),
                (1, Some(RunOption::Last)),
                (0, None),
                (1, Some(RunOption::Full))
            ]))
        );
        assert_eq!(
            iter.next(),
            Some(RunInfo(vec![
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
        let run = RunInput {
            batches: vec![
                (vec![0; 139], RunOption::Last),
                (vec![1; 1], RunOption::Last),
                (vec![2; 0], RunOption::Full),
                (vec![3; 65], RunOption::Full),
            ],
            token_chunk_size: 128,
        };

        let run = run.advance();
        assert_eq!(
            run.iter().next(),
            Some(RunInfo(vec![
                (61, None),
                (0, None),
                (0, None),
                (3, Some(RunOption::Full))
            ]))
        );

        // simulate adding one token to batch 1 after advancing.
        let run = RunInput {
            batches: vec![
                (vec![0; 61], RunOption::Last),
                (vec![1; 1], RunOption::Last),
                (vec![2; 0], RunOption::Full),
                (vec![3; 3], RunOption::Full),
            ],
            token_chunk_size: 128,
        };
        assert_eq!(
            run.iter().next(),
            Some(RunInfo(vec![
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
            batches: vec![
                (vec![0; 61], RunOption::Last),
                (vec![1; 0], RunOption::Last),
                (vec![2; 0], RunOption::Full),
                (vec![3; 3], RunOption::Full),
            ],
            token_chunk_size: 128,
        };
        let redirect = run.iter().next().unwrap().redirect();

        assert_eq!(redirect.headers, vec![60, 61, 62, 63]);
        assert_eq!(
            redirect.batches,
            vec![(0, 61), (61, 61), (61, 61), (61, 64)]
        );

        Ok(())
    }
}
