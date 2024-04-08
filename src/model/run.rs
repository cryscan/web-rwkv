use std::{collections::HashMap, future::Future, hash::Hash};

use anyhow::Result;
use half::f16;
use itertools::Itertools;

use super::{
    ModelBase, ModelInfo, ModelInput, ModelOutput, ModelState, OutputType, MIN_TOKEN_CHUNK_SIZE,
};
use crate::{
    context::Context,
    num::{CoHom, Float},
    tensor::{
        kind::ReadWrite,
        ops::TensorOp,
        shape::{Shape, TensorDimension},
        TensorCpu, TensorError, TensorGpu, TensorReshape, TensorStack,
    },
};

#[derive(Debug)]
pub struct Header<F: Float> {
    pub head_x: TensorGpu<F, ReadWrite>,
    pub head_o: TensorGpu<f32, ReadWrite>,
}

impl<F: Float> Header<F> {
    pub fn new(context: &Context, info: &ModelInfo, num_batch: usize) -> Self {
        let head_shape = Shape::new(info.num_emb, num_batch, 1, 1);
        let output_shape = Shape::new(info.num_vocab, num_batch, 1, 1);

        Self {
            head_x: context.tensor_init(head_shape),
            head_o: context.tensor_init(output_shape),
        }
    }
}

pub type HookMap<Hook, Tensor, State, Runtime, Header> =
    HashMap<Hook, Box<dyn Fn(&Tensor, &State, &Runtime, &Header) -> Result<TensorOp, TensorError>>>;

pub(crate) trait ModelRunInternal: ModelBase {
    type Hook: Hash;
    type State: ModelState;
    type Tensor;
    type Runtime;
    type Header;

    fn tensor(&self) -> &Self::Tensor;

    fn checkout_runtime(&self, num_batch: usize) -> Self::Runtime;
    fn checkout_header(&self, num_batch: usize) -> Self::Header;

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
        hooks: &HookMap<Self::Hook, Self::Tensor, Self::State, Self::Runtime, Self::Header>,
    ) -> Result<(TensorGpu<f32, ReadWrite>, Vec<std::ops::Range<usize>>), TensorError>;

    fn create_input<'a, F: Float>(
        &self,
        embed: &TensorCpu<'a, f16>,
        tokens: &[Vec<u16>],
    ) -> Result<TensorStack<'a, F>, TensorError> {
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
                .unwrap_or_else(|_| context.zeros(Shape::new(info.num_emb, 1, 0, 1)))
                .map(|x| CoHom::co_hom(*x));
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
    type Tensor;
    type Runtime;
    type Header;

    fn tensor(&self) -> &Self::Tensor;

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
    #[allow(clippy::type_complexity)]
    fn run_with_hooks(
        &self,
        tokens: &mut Vec<ModelInput>,
        state: &Self::State,
        hooks: &HookMap<Self::Hook, Self::Tensor, Self::State, Self::Runtime, Self::Header>,
    ) -> impl Future<Output = Result<Vec<ModelOutput>, TensorError>>;
}

impl<Hook, Model, Tensor, State, Runtime, Header> ModelRun for Model
where
    Hook: Hash,
    State: super::ModelState,
    Model: ModelRunInternal<
        Hook = Hook,
        Tensor = Tensor,
        State = State,
        Runtime = Runtime,
        Header = Header,
    >,
{
    type Hook = Hook;
    type State = State;
    type Tensor = Tensor;
    type Runtime = Runtime;
    type Header = Header;

    #[inline]
    fn tensor(&self) -> &Self::Tensor {
        <Self as ModelRunInternal>::tensor(self)
    }

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
        hooks: &HookMap<Self::Hook, Self::Tensor, Self::State, Self::Runtime, Self::Header>,
    ) -> Result<Vec<ModelOutput>, TensorError> {
        let num_token: usize = tokens.iter().map(|input| input.tokens.len()).sum();
        let num_batch = state.num_batch();

        if tokens.len() != num_batch {
            return Err(TensorError::Batch(tokens.len(), num_batch));
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

        let mut inputs = vec![vec![]; num_batch];
        let mut outputs: Vec<Option<OutputType>> = vec![None; num_batch];

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
        let output = output.back().await;

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
