use std::{collections::HashMap, future::Future, hash::Hash, sync::Arc};

use anyhow::Result;
use half::f16;
use itertools::Itertools;

use super::{
    ModelBase, ModelInfo, ModelInput, ModelOutput, ModelState, OutputType, MIN_TOKEN_CHUNK_SIZE,
};
use crate::{
    context::Context,
    tensor::{
        kind::{ReadBack, ReadWrite},
        ops::TensorOp,
        shape::{Shape, TensorDimension},
        TensorCpu, TensorError, TensorGpu, TensorReshape, TensorStack,
    },
};

#[derive(Debug)]
pub struct Header {
    pub head_x: TensorGpu<f16, ReadWrite>,
    pub head_o: TensorGpu<f32, ReadWrite>,
    pub map: TensorGpu<f32, ReadBack>,
}

impl Header {
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

pub type HookMap<Hook, Model, State, Runtime> =
    HashMap<Hook, Box<dyn Fn(&Model, &State, &Runtime) -> Result<TensorOp, TensorError>>>;

pub(crate) trait ModelRunInternal: ModelBase {
    type Hook: Hash;
    type State: ModelState;
    type Runtime;

    fn checkout_runtime(&self, num_batch: usize) -> Arc<Self::Runtime>;
    fn checkout_header(&self, num_batch: usize) -> Arc<Header>;

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
        outputs: Vec<Option<OutputType>>,
        hooks: &HookMap<Self::Hook, Self, Self::State, Self::Runtime>,
    ) -> Result<(TensorGpu<f32, ReadBack>, Vec<std::ops::Range<usize>>), TensorError>;

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
    type Hook: Hash;
    type State: ModelState;
    type Runtime;

    /// Run the model for a batch of tokens as input.
    /// The length of `tokens` must match the number of batches in `state`.
    /// `tokens` may have slots with no tokens, for which `run` won't compute that batch and will return an empty vector in that corresponding slot.
    fn run(
        &self,
        tokens: &mut Vec<ModelInput>,
        state: &Self::State,
    ) -> impl Future<Output = Result<Vec<ModelOutput>, TensorError>>;

    /// Run the model for a batch of tokens as input, but with custom hooks.
    /// The length of `tokens` must match the number of batches in `state`.
    /// `tokens` may have slots with no tokens, for which `run` won't compute that batch and will return an empty vector in that corresponding slot.
    fn run_with_hooks(
        &self,
        tokens: &mut Vec<ModelInput>,
        state: &Self::State,
        hooks: &HookMap<Self::Hook, Self, Self::State, Self::Runtime>,
    ) -> impl Future<Output = Result<Vec<ModelOutput>, TensorError>>;
}

impl<Hook, Runtime, Model, State> ModelRun for Model
where
    Hook: Hash,
    Model: ModelRunInternal<Hook = Hook, Runtime = Runtime, State = State>,
    State: super::ModelState,
{
    type Hook = Hook;
    type Runtime = Runtime;
    type State = State;

    async fn run(
        &self,
        tokens: &mut Vec<ModelInput>,
        state: &Self::State,
    ) -> Result<Vec<ModelOutput>, TensorError> {
        let hooks = Default::default();
        self.run_with_hooks(tokens, state, &hooks).await
    }

    async fn run_with_hooks(
        &self,
        tokens: &mut Vec<ModelInput>,
        state: &Self::State,
        hooks: &HookMap<Self::Hook, Self, Self::State, Self::Runtime>,
    ) -> Result<Vec<ModelOutput>, TensorError> {
        let num_token: usize = tokens.iter().map(|input| input.tokens.len()).sum();
        let max_batch = state.num_batch();

        if tokens.len() != max_batch {
            return Err(TensorError::Batch(tokens.len(), max_batch));
        }
        if num_token == 0 {
            return Ok(vec![ModelOutput::None; tokens.len()]);
        }

        // we only infer at most `token_chunk_size` tokens at a time
        let num_token = num_token.min(self.token_chunk_size());
        let mut num_token = match num_token > MIN_TOKEN_CHUNK_SIZE {
            true => num_token - num_token % MIN_TOKEN_CHUNK_SIZE,
            false => num_token,
        };

        let mut inputs = vec![vec![]; max_batch];
        let mut outputs: Vec<Option<OutputType>> = vec![None; max_batch];

        // consume all available token counts
        // assign them to as many slots as possible
        while num_token > 0 {
            let mid = tokens
                .iter()
                .map(|input| input.tokens.len())
                .filter(|x| x > &0)
                .min()
                .unwrap_or_default();
            for (output, input, slot) in
                itertools::multizip((outputs.iter_mut(), inputs.iter_mut(), tokens.iter_mut()))
            {
                let mid = mid.min(slot.tokens.len()).min(num_token);
                num_token -= mid;

                if mid > 0 {
                    let (head, tail) = slot.tokens.split_at(mid);
                    *output = match slot.ty {
                        OutputType::Last => tail.is_empty().then_some(OutputType::Last),
                        OutputType::Full => Some(OutputType::Full),
                    };
                    input.append(&mut head.to_vec());
                    slot.tokens = tail.to_vec();
                }
            }
        }

        let (output, redirect) = self.run_internal(inputs, state, outputs, hooks)?;
        let output = output.back_async().await;

        Ok(redirect
            .into_iter()
            .map(|r| match r.len() {
                0 => ModelOutput::None,
                1 => ModelOutput::Last(output.slice(.., r.start, .., ..).unwrap().to_vec()),
                _ => ModelOutput::Full(
                    r.map(|index| output.slice(.., index, .., ..).unwrap().to_vec())
                        .collect(),
                ),
            })
            .collect())
    }
}
