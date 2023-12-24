use std::{collections::HashMap, hash::Hash, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;

use super::{ModelBase, ModelError, ModelInfo, ModelState};
use crate::{
    context::Context,
    tensor::{
        ops::{TensorOp, TensorOpHook},
        shape::Shape,
        ReadBack, ReadWrite, TensorGpu,
    },
};

#[derive(Debug)]
pub struct Output {
    pub head_x: TensorGpu<f32, ReadWrite>,
    pub head_o: TensorGpu<f32, ReadWrite>,
    pub map: TensorGpu<f32, ReadBack>,
}

impl Output {
    pub fn new(context: &Context, info: &ModelInfo, num_batch: usize) -> Self {
        let head_shape = Shape::new(info.num_emb, num_batch, 1, 1);
        let output_shape = Shape::new(info.num_vocab, num_batch, 1, 1);

        Self {
            head_x: context.tensor_init(head_shape),
            head_o: context.tensor_init(output_shape),
            map: context.tensor_init(output_shape),
        }
    }
}

pub type HookMap<Hook, ModelState, Runtime> =
    HashMap<Hook, Box<dyn Fn(&ModelState, &Runtime) -> TensorOp + Send + Sync>>;

#[async_trait]
pub trait ModelRun: ModelBase {
    type Hook: TensorOpHook + Hash + Sync;
    type Runtime;

    fn request_output(&self, num_batch: usize) -> Arc<Output>;

    /// Actual implementation of the model's inference.
    #[allow(clippy::type_complexity)]
    fn run_internal(
        &self,
        tokens: Vec<Vec<u16>>,
        state: &Self::ModelState,
        should_output: Vec<bool>,
        hooks: &HookMap<Self::Hook, Self::ModelState, Self::Runtime>,
    ) -> Result<(TensorGpu<f32, ReadBack>, Vec<Option<usize>>)>;

    /// Run the model for a batch of tokens as input.
    /// The length of `tokens` must match the number of batches in `state`.
    /// `tokens` may have slots with no tokens, for which `run` won't compute that batch and will return an empty vector in that corresponding slot.
    async fn run(
        &self,
        tokens: &mut Vec<Vec<u16>>,
        state: &Self::ModelState,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let num_token: usize = tokens.iter().map(Vec::len).sum();
        let max_batch = state.max_batch();

        if tokens.len() != max_batch {
            return Err(ModelError::BatchSize(tokens.len(), max_batch).into());
        }
        if num_token == 0 {
            return Err(ModelError::EmptyInput.into());
        }

        // we only infer at most `token_chunk_size` tokens at a time
        let mut num_token = num_token.min(self.token_chunk_size());
        let mut inputs = vec![vec![]; max_batch];
        let mut should_output = vec![false; max_batch];

        // take `num_token` tokens out of all the inputs and put into `input`
        // first pass, make sure each slot computes at least one token
        for (output, input, remain) in itertools::multizip((
            should_output.iter_mut(),
            inputs.iter_mut(),
            tokens.iter_mut(),
        )) {
            let mid = 1.min(remain.len()).min(num_token);
            num_token -= mid;

            if mid > 0 {
                let (head, tail) = remain.split_at(mid);
                *output = tail.is_empty();
                *input = [&input, head].concat();
                *remain = tail.to_vec();
            }
        }

        // second pass, assign rest token budgets from left to right
        for (output, input, remain) in itertools::multizip((
            should_output.iter_mut(),
            inputs.iter_mut(),
            tokens.iter_mut(),
        )) {
            let mid = remain.len().min(num_token);
            num_token -= mid;

            if mid > 0 {
                let (head, tail) = remain.split_at(mid);
                *output = tail.is_empty();
                *input = [&input, head].concat();
                *remain = tail.to_vec();
            }
        }

        let (output, redirect) =
            self.run_internal(inputs, state, should_output, &Default::default())?;
        let output = output.back_async().await;

        Ok(redirect
            .into_iter()
            .map(|index| {
                index.map(|index| {
                    output
                        .slice(.., index, .., ..)
                        .expect("this never happens")
                        .to_vec()
                })
            })
            .collect())
    }

    /// Run the model for a batch of tokens as input, but with custom hooks.
    /// The length of `tokens` must match the number of batches in `state`.
    /// `tokens` may have slots with no tokens, for which `run` won't compute that batch and will return an empty vector in that corresponding slot.
    async fn run_with_hooks(
        &self,
        tokens: &mut Vec<Vec<u16>>,
        state: &Self::ModelState,
        hooks: &HookMap<Self::Hook, Self::ModelState, Self::Runtime>,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let num_token: usize = tokens.iter().map(Vec::len).sum();
        let max_batch = state.max_batch();

        if tokens.len() != max_batch {
            return Err(ModelError::BatchSize(tokens.len(), max_batch).into());
        }
        if num_token == 0 {
            return Err(ModelError::EmptyInput.into());
        }

        // we only infer at most `token_chunk_size` tokens at a time
        let mut num_token = num_token.min(self.token_chunk_size());
        let mut inputs = vec![vec![]; max_batch];
        let mut should_output = vec![false; max_batch];

        // take `num_token` tokens out of all the inputs and put into `input`
        // first pass, make sure each slot computes at least one token
        for (output, input, remain) in itertools::multizip((
            should_output.iter_mut(),
            inputs.iter_mut(),
            tokens.iter_mut(),
        )) {
            let mid = 1.min(remain.len()).min(num_token);
            num_token -= mid;

            if mid > 0 {
                let (head, tail) = remain.split_at(mid);
                *output = tail.is_empty();
                *input = [&input, head].concat();
                *remain = tail.to_vec();
            }
        }

        // second pass, assign rest token budgets from left to right
        for (output, input, remain) in itertools::multizip((
            should_output.iter_mut(),
            inputs.iter_mut(),
            tokens.iter_mut(),
        )) {
            let mid = remain.len().min(num_token);
            num_token -= mid;

            if mid > 0 {
                let (head, tail) = remain.split_at(mid);
                *output = tail.is_empty();
                *input = [&input, head].concat();
                *remain = tail.to_vec();
            }
        }

        let (output, redirect) = self.run_internal(inputs, state, should_output, hooks)?;
        let output = output.back_async().await;

        Ok(redirect
            .into_iter()
            .map(|index| {
                index.map(|index| {
                    output
                        .slice(.., index, .., ..)
                        .expect("this never happens")
                        .to_vec()
                })
            })
            .collect())
    }
}
