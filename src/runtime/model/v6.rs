use std::marker::PhantomData;

use itertools::Itertools;
use serde::Serialize;
use web_rwkv_derive::DeserializeSeed;
use wgpu::CommandBuffer;

use super::ModelInfo;
use crate::{
    context::Context,
    num::Float,
    runtime::{
        run::{RunInfo, RunInput, RunOutput},
        Job, JobBuilder,
    },
    tensor::{kind::ReadWrite, shape::Shape, TensorError, TensorGpu},
};

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct Model {
    #[serde(serialize_with = "crate::tensor::serialization::serialize_context")]
    context: Context,
    info: ModelInfo,
    tensor: ModelTensor,
}

impl Model {
    const TIME_MIX_ADAPTER_SIZE: usize = 32;
    const TIME_DECAY_ADAPTER_SIZE: usize = 64;

    pub const LN_EPS: f32 = 1.0e-5;
    pub const GN_EPS: f32 = 64.0e-5;
}

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct ModelTensor {}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct State {
    info: ModelInfo,
    num_batch: usize,
    head_size: usize,
    data: Vec<TensorGpu<f32, ReadWrite>>,
}

#[derive(Debug)]
pub struct Runtime<F: Float> {
    pub tokens: TensorGpu<u32, ReadWrite>,
    pub cursors: TensorGpu<u32, ReadWrite>,
    pub input: TensorGpu<F, ReadWrite>,

    pub att_x: TensorGpu<F, ReadWrite>,
    pub att_xx: TensorGpu<F, ReadWrite>,
    /// Token shifted time decay input, `[C, T]`.
    pub att_wx: TensorGpu<F, ReadWrite>,
    pub att_kx: TensorGpu<F, ReadWrite>,
    pub att_vx: TensorGpu<F, ReadWrite>,
    pub att_rx: TensorGpu<F, ReadWrite>,
    pub att_gx: TensorGpu<F, ReadWrite>,
    /// Time decay LoRA intermediate, `[64, T]`.
    pub att_w: TensorGpu<F, ReadWrite>,
    pub att_k: TensorGpu<f32, ReadWrite>,
    pub att_v: TensorGpu<f32, ReadWrite>,
    pub att_r: TensorGpu<f32, ReadWrite>,
    pub att_g: TensorGpu<F, ReadWrite>,
    pub att_o: TensorGpu<F, ReadWrite>,

    /// Token shift LoRA intermediate, `[32, 5, T]`.
    pub time_mix_x: TensorGpu<F, ReadWrite>,
    /// Token shift LoRA intermediate transposed, `[32, T, 5]`.
    pub time_mix_t: TensorGpu<F, ReadWrite>,
    /// Token shift LoRA output, `[C, T, 5]`.
    pub time_mix: TensorGpu<F, ReadWrite>,
    pub time_decay: TensorGpu<f32, ReadWrite>,

    pub ffn_x: TensorGpu<F, ReadWrite>,
    pub ffn_kx: TensorGpu<F, ReadWrite>,
    pub ffn_rx: TensorGpu<F, ReadWrite>,
    pub ffn_k: TensorGpu<F, ReadWrite>,
    pub ffn_v: TensorGpu<F, ReadWrite>,
    pub ffn_r: TensorGpu<F, ReadWrite>,

    pub aux_x: TensorGpu<f32, ReadWrite>,
}

impl<F: Float> Runtime<F> {
    pub fn new(context: &Context, info: &ModelInfo, num_token: usize) -> Self {
        let shape = Shape::new(info.num_emb, num_token, 1, 1);
        let tokens_shape = Shape::new(num_token, 1, 1, 1);
        let cursors_shape = Shape::new(num_token, 1, 1, 1);
        let hidden_shape = Shape::new(info.num_hidden, num_token, 1, 1);
        let time_mix_shape = Shape::new(info.num_emb, num_token, 5, 1);
        let time_mix_x_shape = Shape::new(Model::TIME_MIX_ADAPTER_SIZE, 5, num_token, 1);
        let time_mix_t_shape = Shape::new(Model::TIME_MIX_ADAPTER_SIZE, num_token, 5, 1);
        let time_decay_shape = Shape::new(Model::TIME_DECAY_ADAPTER_SIZE, num_token, 1, 1);

        Self {
            tokens: context.tensor_init(tokens_shape),
            cursors: context.tensor_init(cursors_shape),
            input: context.tensor_init(shape),
            att_x: context.tensor_init(shape),
            att_xx: context.tensor_init(shape),
            att_wx: context.tensor_init(shape),
            att_kx: context.tensor_init(shape),
            att_vx: context.tensor_init(shape),
            att_rx: context.tensor_init(shape),
            att_gx: context.tensor_init(shape),
            att_w: context.tensor_init(time_decay_shape),
            att_k: context.tensor_init(shape),
            att_v: context.tensor_init(shape),
            att_r: context.tensor_init(shape),
            att_g: context.tensor_init(shape),
            att_o: context.tensor_init(shape),
            time_mix_x: context.tensor_init(time_mix_x_shape),
            time_mix_t: context.tensor_init(time_mix_t_shape),
            time_mix: context.tensor_init(time_mix_shape),
            time_decay: context.tensor_init(shape),
            ffn_x: context.tensor_init(shape),
            ffn_kx: context.tensor_init(shape),
            ffn_rx: context.tensor_init(shape),
            ffn_k: context.tensor_init(hidden_shape),
            ffn_v: context.tensor_init(shape),
            ffn_r: context.tensor_init(shape),
            aux_x: context.tensor_init(shape),
        }
    }
}

#[derive(Debug)]
pub struct Header<F: Float> {
    pub head_x: TensorGpu<F, ReadWrite>,
    pub head_o: TensorGpu<F, ReadWrite>,
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

pub struct RunJob<F: Float> {
    context: Context,
    commands: Vec<CommandBuffer>,
    redirect: Vec<(usize, usize)>,

    tokens: TensorGpu<u32, ReadWrite>,
    cursors: TensorGpu<u32, ReadWrite>,
    input: TensorGpu<F, ReadWrite>,
    output: TensorGpu<F, ReadWrite>,
}

impl<F: Float> Job for RunJob<F> {
    type Input = RunInput;
    type Output = RunOutput<F>;
    type Error = TensorError;

    fn load(&self, input: &Self::Input) -> Result<(), Self::Error> {
        todo!()
    }

    async fn submit(self) -> Result<Self::Output, Self::Error> {
        self.context.queue.submit(self.commands);
        let output = self.output.back().await;
        let batches = self
            .redirect
            .into_iter()
            .map(|(start, end)| output.slice(.., start..end, .., ..))
            .try_collect()?;
        Ok(RunOutput(batches))
    }
}

pub struct RunJobBuilder<'a, F: Float> {
    pub model: &'a Model,
    pub state: &'a State,

    pub turbo: bool,
    pub token_chunk_size: usize,
    pub _phantom: PhantomData<F>,
}

impl<F: Float> JobBuilder for RunJobBuilder<'_, F> {
    type Info = RunInfo;
    type Input = RunInput;
    type Output = RunOutput<F>;

    fn build(
        &self,
        info: Self::Info,
    ) -> impl Job<Input = Self::Input, Output = Self::Output> + 'static {
        RunJob {
            context: todo!(),
            commands: todo!(),
            redirect: todo!(),
            tokens: todo!(),
            cursors: todo!(),
            input: todo!(),
            output: todo!(),
        }
    }
}
