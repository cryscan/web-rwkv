use crate::{num::Float, tensor::TensorCpu};

#[derive(Debug, Default, Clone)]
pub struct RunInfo(pub Vec<(usize, Option<RunOption>)>);

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

pub struct RunPredictor {
    batches: Vec<(usize, RunOption)>,
    token_chunk_size: usize,
}

impl Iterator for RunPredictor {
    type Item = RunInfo;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl IntoIterator for &RunInput {
    type Item = RunInfo;
    type IntoIter = RunPredictor;

    fn into_iter(self) -> Self::IntoIter {
        let batches = self
            .batches
            .iter()
            .map(|(tokens, option)| (tokens.len(), *option))
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
