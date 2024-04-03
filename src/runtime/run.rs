use itertools::Itertools;

use crate::{num::Float, tensor::TensorCpu};

pub const MIN_TOKEN_CHUNK_SIZE: usize = 32;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunInfo(pub Vec<(usize, Option<RunOption>)>);

#[derive(Debug, Clone, Copy)]
pub enum BatchInput {
    Gen,
    Read(usize),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
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

impl RunInput {
    pub fn iter(&self) -> RunIter {
        self.into_iter()
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

pub struct RunIter {
    batches: Vec<(BatchInput, RunOption)>,
    token_chunk_size: usize,
}

impl Iterator for RunIter {
    type Item = RunInfo;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batches = self
            .batches
            .iter()
            .map(|(x, _)| match x {
                BatchInput::Gen => 1,
                BatchInput::Read(x) => *x,
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

        let mut info = vec![(0, None); num_batch];
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
                        0 => BatchInput::Gen,
                        &x => BatchInput::Read(x),
                    };
                    info.1 = match batch.1 {
                        RunOption::Last => match remain {
                            0 => Some(RunOption::Last),
                            _ => None,
                        },
                        RunOption::Full => Some(RunOption::Full),
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
    use super::{RunInput, RunOption};
    use crate::runtime::run::RunInfo;

    #[test]
    fn test_run_iter() {
        let run = RunInput {
            batches: vec![
                (vec![0; 139], RunOption::Last),
                (vec![0; 1], RunOption::Last),
                (vec![0; 0], RunOption::Full),
                (vec![0; 65], RunOption::Full),
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
        )
    }
}
