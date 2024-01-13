use std::{collections::HashMap, future::Future, sync::Arc};

use anyhow::Result;
use half::f16;
use itertools::Itertools;

use super::{ModelBase, ModelError, ModelInfo, ModelState};
use crate::{
    context::Context,
    tensor::{
        kind::{ReadBack, ReadWrite},
        ops::{TensorOp, TensorOpHook},
        shape::{Shape, TensorDimension},
        TensorCpu, TensorError, TensorGpu, TensorReshape, TensorStack,
    },
};

#[derive(Debug)]
pub struct Output {
    pub head_x: TensorGpu<f16, ReadWrite>,
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

pub type HookMap<Hook, Model, State, Runtime> = HashMap<
    Hook,
    Box<dyn Fn(&Model, &State, &Runtime) -> Result<TensorOp, TensorError> + Send + Sync>,
>;

pub(crate) trait ModelRunInternal: ModelBase + Sync {
    type Hook: TensorOpHook;
    type State: ModelState;
    type Runtime;

    fn request_runtime(&self, num_batch: usize) -> Arc<Self::Runtime>;
    fn request_output(&self, num_batch: usize) -> Arc<Output>;

    /// To prevent the GPU device from lost, this limits the maximum batch-token it processes one time.
    fn token_chunk_size(&self) -> usize;
    /// Whether to use fp16 GEMM for matmul computations, given a number of runtime tokens.
    fn turbo(&self, num_token: usize) -> bool;

    /// Actual implementation of the model's inference.
    #[allow(clippy::type_complexity)]
    fn run_internal(
        &self,
        tokens: Vec<Vec<u16>>,
        state: &Self::State,
        should_output: Vec<bool>,
        hooks: &HookMap<Self::Hook, Self, Self::State, Self::Runtime>,
    ) -> Result<(TensorGpu<f32, ReadBack>, Vec<Option<usize>>)>;

    fn create_input<'a>(
        &self,
        embed: &TensorCpu<'a, f16>,
        tokens: &[Vec<u16>],
    ) -> Result<TensorStack<'a, f16>, TensorError> {
        let info = self.info();
        let context = self.context();

        let input: Vec<_> = tokens
            .iter()
            .map(|tokens| -> Result<_, TensorError> {
                let stack = TensorCpu::stack(
                    tokens
                        .iter()
                        .map(|&token| embed.slice(.., token as usize, .., ..))
                        .try_collect()?,
                )
                .unwrap_or_else(|_| context.zeros(Shape::new(info.num_emb, 1, 0, 1)));
                stack.reshape(
                    TensorDimension::Full,
                    TensorDimension::Auto,
                    TensorDimension::Dimension(1),
                    TensorDimension::Full,
                )
            })
            .try_collect()?;
        TensorStack::try_from(input)
    }
}

pub trait ModelRun {
    type Hook: TensorOpHook;
    type State: ModelState;
    type Runtime;

    /// Run the model for a batch of tokens as input.
    /// The length of `tokens` must match the number of batches in `state`.
    /// `tokens` may have slots with no tokens, for which `run` won't compute that batch and will return an empty vector in that corresponding slot.
    fn run(
        &self,
        tokens: &mut Vec<Vec<u16>>,
        state: &Self::State,
    ) -> impl Future<Output = Result<Vec<Option<Vec<f32>>>>> + Send;

    /// Run the model for a batch of tokens as input, but with custom hooks.
    /// The length of `tokens` must match the number of batches in `state`.
    /// `tokens` may have slots with no tokens, for which `run` won't compute that batch and will return an empty vector in that corresponding slot.
    fn run_with_hooks(
        &self,
        tokens: &mut Vec<Vec<u16>>,
        state: &Self::State,
        hooks: &HookMap<Self::Hook, Self, Self::State, Self::Runtime>,
    ) -> impl Future<Output = Result<Vec<Option<Vec<f32>>>>> + Send;
}

impl<Hook, Runtime, Model, State> ModelRun for Model
where
    Hook: TensorOpHook,
    Model: ModelRunInternal<Hook = Hook, Runtime = Runtime, State = State>,
    State: super::ModelState,
{
    type Hook = Hook;
    type Runtime = Runtime;
    type State = State;

    async fn run(
        &self,
        tokens: &mut Vec<Vec<u16>>,
        state: &Self::State,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let hooks = Default::default();
        self.run_with_hooks(tokens, state, &hooks).await
    }

    async fn run_with_hooks(
        &self,
        tokens: &mut Vec<Vec<u16>>,
        state: &Self::State,
        hooks: &HookMap<Self::Hook, Self, Self::State, Self::Runtime>,
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
