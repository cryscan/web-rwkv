use std::{marker::PhantomData, sync::Arc};

use half::f16;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::DeserializeSeed;
use wgpu::CommandBuffer;

use super::{
    model::{EmbedDevice, ModelInfo},
    run::{RunRedirect, MIN_TOKEN_CHUNK_SIZE},
};
use crate::{
    context::Context,
    num::{CoHom, Float},
    runtime::{
        run::{RunInfo, RunInput, RunOutput},
        Job, JobBuilder,
    },
    tensor::{
        kind::ReadWrite,
        matrix::Matrix,
        ops::{Activation, TensorCommand, TensorOp, TensorPass},
        shape::{Shape, TensorDimension},
        DeepClone, IntoPackedCursors, TensorCpu, TensorError, TensorGpu, TensorGpuView, TensorInit,
        TensorReshape, TensorShape, TensorStack,
    },
};

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct Model {
    #[serde(serialize_with = "crate::tensor::serialization::serialize_context")]
    pub context: Context,
    pub info: ModelInfo,
    pub tensor: ModelTensor,
}

impl Model {
    const RESCALE_LAYER: usize = 6;
    const TIME_MIX_ADAPTER_SIZE: usize = 32;
    const TIME_DECAY_ADAPTER_SIZE: usize = 64;

    pub const LN_EPS: f32 = 1.0e-5;
    pub const GN_EPS: f32 = 64.0e-5;
}

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct ModelTensor {
    pub embed: Embed,
    pub head: Head,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct LayerNorm {
    pub w: TensorGpu<f16, ReadWrite>,
    pub b: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct Att {
    pub time_decay: TensorGpu<f16, ReadWrite>,
    pub time_first: TensorGpu<f32, ReadWrite>,

    pub time_mix_x: TensorGpu<f16, ReadWrite>,
    pub time_mix_w: TensorGpu<f16, ReadWrite>,
    pub time_mix_k: TensorGpu<f16, ReadWrite>,
    pub time_mix_v: TensorGpu<f16, ReadWrite>,
    pub time_mix_r: TensorGpu<f16, ReadWrite>,
    pub time_mix_g: TensorGpu<f16, ReadWrite>,

    pub time_decay_w1: Matrix,
    pub time_decay_w2: Matrix,
    pub time_mix_w1: Matrix,
    pub time_mix_w2: Matrix,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
    pub w_g: Matrix,
    pub w_o: Matrix,

    pub group_norm: LayerNorm,
}

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct Ffn {
    pub time_mix_k: TensorGpu<f16, ReadWrite>,
    pub time_mix_r: TensorGpu<f16, ReadWrite>,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
}

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct Layer {
    pub att_layer_norm: LayerNorm,
    pub ffn_layer_norm: LayerNorm,
    pub att: Att,
    pub ffn: Ffn,
}

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct Embed {
    pub layer_norm: LayerNorm,
    pub w: TensorCpu<'static, f16>,
    pub u: Option<TensorGpu<f16, ReadWrite>>,
}

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct Head {
    pub layer_norm: LayerNorm,
    pub w: Matrix,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct State {
    pub info: ModelInfo,
    pub num_batch: usize,
    pub data: Vec<TensorGpu<f32, ReadWrite>>,
}

impl State {
    fn att(&self, layer: usize) -> Result<TensorGpuView<f32>, TensorError> {
        let head_size = self.info.num_emb / self.info.num_head;
        self.data[layer].view(.., 0..head_size + 1, .., ..)
    }

    fn ffn(&self, layer: usize) -> Result<TensorGpuView<f32>, TensorError> {
        let head_size = self.info.num_emb / self.info.num_head;
        self.data[layer].view(.., head_size + 1, .., ..)
    }
}

impl DeepClone for State {
    fn deep_clone(&self) -> Self {
        let data = self.data.iter().map(|tensor| tensor.deep_clone()).collect();
        Self {
            data,
            ..self.clone()
        }
    }
}

#[derive(Debug)]
pub struct Runtime<F: Float> {
    pub cursors: TensorGpu<u32, ReadWrite>,
    pub tokens: TensorGpu<u32, ReadWrite>,
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
        let cursors_shape = Shape::new(num_token, 1, 1, 1);
        let tokens_shape = Shape::new(num_token, 1, 1, 1);
        let hidden_shape = Shape::new(info.num_hidden, num_token, 1, 1);
        let time_mix_shape = Shape::new(info.num_emb, num_token, 5, 1);
        let time_mix_x_shape = Shape::new(Model::TIME_MIX_ADAPTER_SIZE, 5, num_token, 1);
        let time_mix_t_shape = Shape::new(Model::TIME_MIX_ADAPTER_SIZE, num_token, 5, 1);
        let time_decay_shape = Shape::new(Model::TIME_DECAY_ADAPTER_SIZE, num_token, 1, 1);

        Self {
            cursors: context.tensor_init(cursors_shape),
            tokens: context.tensor_init(tokens_shape),
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
    pub fn new(context: &Context, info: &ModelInfo, num_header: usize) -> Self {
        let head_shape = Shape::new(info.num_emb, num_header, 1, 1);
        let output_shape = Shape::new(info.num_vocab, num_header, 1, 1);

        Self {
            head_x: context.tensor_init(head_shape),
            head_o: context.tensor_init(output_shape),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Hook {
    PostEmbedLoaded,
    PostEmbedLayerNorm,
    PreAtt(usize),
    PostAttLayerNorm(usize),
    PreAttTokenShift(usize),
    PostAttTokenShift(usize),
    PreAttTokenShiftAdapt(usize),
    PostAttTokenShiftAdapt(usize),
    PostAttTokenShiftAdaptActivate(usize),
    PreAttGatedTokenShift(usize),
    PostAttGatedTokenShift(usize),
    PreAttTimeDecayAdapt(usize),
    PostAttTimeDecayAdapt(usize),
    PostAttTimeDecayAdaptActivate(usize),
    PreAttTimeDecayActivate(usize),
    PostAttTimeDecayActivate(usize),
    PreAttLinear(usize),
    PostAttLinear(usize),
    PreAttTimeMix(usize),
    PostAttTimeMix(usize),
    PreAttGate(usize),
    PostAttGate(usize),
    PreAttOut(usize),
    PostAttOut(usize),
    PostAtt(usize),
    PreFfn(usize),
    PostFfnLayerNorm(usize),
    PreFfnTokenShift(usize),
    PostFfnTokenShift(usize),
    PreFfnLinear(usize),
    PostFfnLinear(usize),
    PostFfnActivate(usize),
    PreFfnChannelMix(usize),
    PostFfnChannelMix(usize),
    PostFfn(usize),
    PreHead,
    PostHeadLayerNorm,
    PostHead,
}

struct RunJob<F: Float> {
    model: Arc<Model>,

    commands: Vec<CommandBuffer>,
    redirect: RunRedirect,
    embed_device: EmbedDevice,

    cursors: TensorGpu<u32, ReadWrite>,
    tokens: TensorGpu<u32, ReadWrite>,
    input: TensorGpu<F, ReadWrite>,
    output: TensorGpu<F, ReadWrite>,
}

impl<F: Float> Job for RunJob<F> {
    type Input = RunInput;
    type Output = RunOutput<F>;
    type Error = TensorError;

    fn load(self, input: &Self::Input) -> Result<Self, Self::Error> {
        if input.iter().next().is_none() {
            return Ok(self);
        }

        let chunk = input.chunk();
        let stack: Vec<TensorCpu<F>> = chunk
            .iter()
            .map(|tokens| -> Result<_, Self::Error> {
                let info = &self.model.info;
                let embed = &self.model.tensor.embed.w;
                TensorCpu::stack(
                    tokens
                        .iter()
                        .map(|&token| embed.slice(.., token as usize, .., ..))
                        .try_collect()?,
                )
                .unwrap_or_else(|_| TensorCpu::init(Shape::new(info.num_emb, 1, 0, 1)))
                .map(|x| CoHom::co_hom(*x))
                .reshape(
                    TensorDimension::Full,
                    TensorDimension::Auto,
                    TensorDimension::Dimension(1),
                    TensorDimension::Full,
                )
            })
            .try_collect()?;
        let stack = TensorStack::try_from(stack)?;

        let cursors = stack.cursors.clone().into_cursors();
        let cursors = TensorCpu::from_data(self.cursors.shape(), cursors)?;
        self.cursors.load(&cursors)?;

        match self.embed_device {
            EmbedDevice::Cpu => self.input.load(&stack.tensor)?,
            EmbedDevice::Gpu => {
                let tokens = chunk
                    .into_iter()
                    .concat()
                    .into_iter()
                    .map(|token| token as u32)
                    .collect_vec();
                let tokens = TensorCpu::from_data(self.tokens.shape(), tokens)?;
                self.tokens.load(&tokens)?;
            }
        }

        Ok(self)
    }

    async fn submit(self) -> Result<Self::Output, Self::Error> {
        self.output.context.queue.submit(self.commands);
        let output = self.output.back().await;
        let batches = self
            .redirect
            .batches
            .into_iter()
            .map(|(start, end)| output.slice(.., start..end, .., ..))
            .try_collect()?;
        Ok(RunOutput(batches))
    }
}

struct RunJobBuilder<F: Float> {
    model: Arc<Model>,
    state: Arc<State>,
    phantom: PhantomData<F>,
}

impl<F: Float> JobBuilder for RunJobBuilder<F> {
    type Info = RunInfo;
    type Input = RunInput;
    type Output = RunOutput<F>;
    type Error = TensorError;

    fn build(
        &self,
        input: Self::Info,
    ) -> Result<
        impl Job<Input = Self::Input, Output = Self::Output, Error = Self::Error>,
        Self::Error,
    > {
        let model = self.model.clone();
        let state = self.state.clone();
        let context = &model.context;
        let info = &model.info;
        let tensor = &model.tensor;

        let num_token = input.num_token();
        let head_size = info.num_emb / info.num_head;

        let redirect = input.redirect();
        let num_header = redirect.headers.len();

        let buffer = Runtime::<F>::new(context, info, num_token);
        let header = Header::<F>::new(context, info, num_header);

        let turbo = |num_token: usize| num_token % MIN_TOKEN_CHUNK_SIZE == 0;
        let hook_op = |_hook: Hook| -> Result<TensorOp, TensorError> { Ok(TensorOp::List(vec![])) };

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let (head_ops, head_x) = if num_token == 1 || num_token == num_header {
            (TensorOp::List(vec![]), buffer.ffn_x.clone())
        } else {
            let headers = &redirect.headers;
            let mut start = 0;
            let mut end = 1;
            let mut ops = vec![];
            while end <= headers.len() {
                if end == headers.len() || headers[end - 1] + 1 != headers[end] {
                    let first = headers[start];
                    let last = headers[end - 1];
                    assert_eq!(last - first + 1, end - start);

                    let input = buffer.ffn_x.view(.., first..=last, .., ..)?;
                    let output = header.head_x.view(.., start..end, .., ..)?;
                    ops.push(TensorOp::blit(input, output)?);

                    start = end;
                }
                end += 1;
            }
            (TensorOp::List(ops), header.head_x.clone())
        };

        let mut ops = vec![];
        let embed_device = match &tensor.embed.u {
            Some(u) => {
                ops.push(TensorOp::embed(&buffer.tokens, u, &buffer.input)?);
                EmbedDevice::Gpu
            }
            None => EmbedDevice::Cpu,
        };
        ops.append(&mut vec![
            hook_op(Hook::PostEmbedLoaded)?,
            TensorOp::layer_norm(
                &tensor.embed.layer_norm.w,
                &tensor.embed.layer_norm.b,
                &buffer.input,
                None,
                Model::LN_EPS,
            )?,
            hook_op(Hook::PostEmbedLayerNorm)?,
        ]);

        {
            let ops = TensorOp::List(ops);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&ops);
        }

        for (index, layer) in tensor.layers.iter().enumerate() {
            use TensorDimension::{Auto, Dimension};
            let time_first = layer.att.time_first.reshape(
                Dimension(head_size),
                Auto,
                Dimension(1),
                Dimension(1),
            )?;
            let time_decay = buffer.time_decay.reshape(
                Dimension(head_size),
                Auto,
                Dimension(num_token),
                Dimension(1),
            )?;
            let time_mix_x = buffer.time_mix_x.reshape(
                Auto,
                Dimension(num_token),
                Dimension(1),
                Dimension(1),
            )?;
            let aux_x = buffer.aux_x.reshape(
                Dimension(head_size),
                Auto,
                Dimension(num_token),
                Dimension(1),
            )?;
            let att_k = buffer.att_k.reshape(
                Dimension(head_size),
                Auto,
                Dimension(num_token),
                Dimension(1),
            )?;
            let att_v = buffer.att_v.reshape(
                Dimension(head_size),
                Auto,
                Dimension(num_token),
                Dimension(1),
            )?;
            let att_r = buffer.att_r.reshape(
                Dimension(head_size),
                Auto,
                Dimension(num_token),
                Dimension(1),
            )?;

            encoder.copy_tensor(&buffer.input, &buffer.att_x)?;

            let ops = TensorOp::List(vec![
                hook_op(Hook::PreAtt(index))?,
                TensorOp::layer_norm(
                    &layer.att_layer_norm.w,
                    &layer.att_layer_norm.b,
                    &buffer.att_x,
                    None,
                    Model::LN_EPS,
                )?,
                hook_op(Hook::PostAttLayerNorm(index))?,
                hook_op(Hook::PreAttTokenShift(index))?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    layer.att.time_mix_x.view(.., .., .., ..)?,
                    state.att(index)?,
                    &buffer.att_x,
                    &buffer.att_xx,
                    true,
                )?,
                hook_op(Hook::PostAttTokenShift(index))?,
                hook_op(Hook::PreAttTokenShiftAdapt(index))?,
                layer.att.time_mix_w1.matmul_op(
                    buffer.att_xx.view(.., .., .., ..)?,
                    time_mix_x.view(.., .., .., ..)?,
                    Activation::Tanh,
                    turbo(num_token),
                )?,
                TensorOp::transpose(
                    buffer.time_mix_x.view(.., .., .., ..)?,
                    buffer.time_mix_t.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostAttTokenShiftAdaptActivate(index))?,
                layer.att.time_mix_w2.matmul_op(
                    buffer.time_mix_t.view(.., .., .., ..)?,
                    buffer.time_mix.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_token),
                )?,
                hook_op(Hook::PostAttTokenShiftAdapt(index))?,
                TensorOp::add(
                    layer.att.time_mix_w.view(.., .., .., ..)?,
                    buffer.time_mix.view(.., .., 0, ..)?,
                )?,
                TensorOp::add(
                    layer.att.time_mix_k.view(.., .., .., ..)?,
                    buffer.time_mix.view(.., .., 1, ..)?,
                )?,
                TensorOp::add(
                    layer.att.time_mix_v.view(.., .., .., ..)?,
                    buffer.time_mix.view(.., .., 2, ..)?,
                )?,
                TensorOp::add(
                    layer.att.time_mix_r.view(.., .., .., ..)?,
                    buffer.time_mix.view(.., .., 3, ..)?,
                )?,
                TensorOp::add(
                    layer.att.time_mix_g.view(.., .., .., ..)?,
                    buffer.time_mix.view(.., .., 4, ..)?,
                )?,
                hook_op(Hook::PreAttGatedTokenShift(index))?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    buffer.time_mix.view(.., .., 0, ..)?,
                    state.att(index)?,
                    &buffer.att_x,
                    &buffer.att_wx,
                    true,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    buffer.time_mix.view(.., .., 1, ..)?,
                    state.att(index)?,
                    &buffer.att_x,
                    &buffer.att_kx,
                    true,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    buffer.time_mix.view(.., .., 2, ..)?,
                    state.att(index)?,
                    &buffer.att_x,
                    &buffer.att_vx,
                    true,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    buffer.time_mix.view(.., .., 3, ..)?,
                    state.att(index)?,
                    &buffer.att_x,
                    &buffer.att_rx,
                    true,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    buffer.time_mix.view(.., .., 4, ..)?,
                    state.att(index)?,
                    &buffer.att_x,
                    &buffer.att_gx,
                    true,
                )?,
                hook_op(Hook::PostAttGatedTokenShift(index))?,
                hook_op(Hook::PreAttLinear(index))?,
                layer.att.w_k.matmul_op(
                    buffer.att_kx.view(.., .., .., ..)?,
                    buffer.att_k.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_token),
                )?,
                layer.att.w_v.matmul_op(
                    buffer.att_vx.view(.., .., .., ..)?,
                    buffer.att_v.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_token),
                )?,
                layer.att.w_r.matmul_op(
                    buffer.att_rx.view(.., .., .., ..)?,
                    buffer.att_r.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_token),
                )?,
                layer.att.w_g.matmul_op(
                    buffer.att_gx.view(.., .., .., ..)?,
                    buffer.att_g.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_token),
                )?,
                hook_op(Hook::PostAttLinear(index))?,
                hook_op(Hook::PreAttTimeDecayAdapt(index))?,
                layer.att.time_decay_w1.matmul_op(
                    buffer.att_wx.view(.., .., .., ..)?,
                    buffer.att_w.view(.., .., .., ..)?,
                    Activation::Tanh,
                    turbo(num_token),
                )?,
                hook_op(Hook::PostAttTimeDecayAdaptActivate(index))?,
                layer.att.time_decay_w2.matmul_op(
                    buffer.att_w.view(.., .., .., ..)?,
                    buffer.time_decay.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_token),
                )?,
                hook_op(Hook::PostAttTimeDecayAdapt(index))?,
                TensorOp::add(
                    layer.att.time_decay.view(.., .., .., ..)?,
                    buffer.time_decay.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PreAttTimeDecayActivate(index))?,
                TensorOp::stable_exp(&buffer.time_decay)?,
                hook_op(Hook::PostAttTimeDecayActivate(index))?,
                hook_op(Hook::PreAttTimeMix(index))?,
                TensorOp::blit(
                    buffer.att_x.view(.., .., .., ..)?,
                    buffer.aux_x.view(.., .., .., ..)?,
                )?,
                TensorOp::time_mix_v6(
                    &buffer.cursors,
                    &time_decay,
                    &time_first,
                    state.att(index)?,
                    &att_k,
                    &att_v,
                    &att_r,
                    &aux_x,
                )?,
                TensorOp::group_norm(
                    &layer.att.group_norm.w,
                    &layer.att.group_norm.b,
                    &aux_x,
                    Model::GN_EPS,
                )?,
                TensorOp::blit(
                    buffer.aux_x.view(.., .., .., ..)?,
                    buffer.att_x.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostAttTimeMix(index))?,
                hook_op(Hook::PreAttGate(index))?,
                TensorOp::silu(&buffer.att_g, &buffer.att_x)?,
                hook_op(Hook::PostAttGate(index))?,
                hook_op(Hook::PreAttOut(index))?,
                layer.att.w_o.matmul_op(
                    buffer.att_x.view(.., .., .., ..)?,
                    buffer.att_o.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_token),
                )?,
                hook_op(Hook::PostAttOut(index))?,
                TensorOp::add(
                    buffer.input.view(.., .., .., ..)?,
                    buffer.att_o.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostAtt(index))?,
            ]);

            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.execute_tensor_op(&ops);
            }

            encoder.copy_tensor(&buffer.att_o, &buffer.ffn_x)?;

            let ops = TensorOp::List(vec![
                hook_op(Hook::PreFfn(index))?,
                TensorOp::layer_norm(
                    &layer.ffn_layer_norm.w,
                    &layer.ffn_layer_norm.b,
                    &buffer.ffn_x,
                    None,
                    Model::LN_EPS,
                )?,
                hook_op(Hook::PostFfnLayerNorm(index))?,
                hook_op(Hook::PreFfnTokenShift(index))?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    layer.ffn.time_mix_k.view(.., .., .., ..)?,
                    state.ffn(index)?,
                    &buffer.ffn_x,
                    &buffer.ffn_kx,
                    true,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    layer.ffn.time_mix_r.view(.., .., .., ..)?,
                    state.ffn(index)?,
                    &buffer.ffn_x,
                    &buffer.ffn_rx,
                    true,
                )?,
                hook_op(Hook::PostFfnTokenShift(index))?,
                hook_op(Hook::PreFfnLinear(index))?,
                layer.ffn.w_k.matmul_op(
                    buffer.ffn_kx.view(.., .., .., ..)?,
                    buffer.ffn_k.view(.., .., .., ..)?,
                    Activation::SquaredRelu,
                    turbo(num_token),
                )?,
                hook_op(Hook::PostFfnActivate(index))?,
                layer.ffn.w_v.matmul_op(
                    buffer.ffn_k.view(.., .., .., ..)?,
                    buffer.ffn_v.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_token),
                )?,
                layer.ffn.w_r.matmul_op(
                    buffer.ffn_rx.view(.., .., .., ..)?,
                    buffer.ffn_r.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_token),
                )?,
                hook_op(Hook::PostFfnLinear(index))?,
                hook_op(Hook::PreFfnChannelMix(index))?,
                TensorOp::channel_mix(
                    &buffer.cursors,
                    state.ffn(index)?,
                    &buffer.ffn_r,
                    &buffer.ffn_v,
                    &buffer.ffn_x,
                )?,
                hook_op(Hook::PostFfnChannelMix(index))?,
                TensorOp::add(
                    buffer.att_o.view(.., .., .., ..)?,
                    buffer.ffn_x.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostFfn(index))?,
            ]);

            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.execute_tensor_op(&ops);
            }

            if (index + 1) % Model::RESCALE_LAYER == 0 {
                let op = TensorOp::discount(&buffer.ffn_x, 0.5)?;
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.execute_tensor_op(&op);
            }

            if index != model.info.num_layer - 1 {
                encoder.copy_tensor(&buffer.ffn_x, &buffer.input)?;
            }
        }

        if num_header > 0 {
            let ops = TensorOp::List(vec![
                hook_op(Hook::PreHead)?,
                TensorOp::layer_norm(
                    &tensor.head.layer_norm.w,
                    &tensor.head.layer_norm.b,
                    &head_x,
                    None,
                    Model::LN_EPS,
                )?,
                hook_op(Hook::PostHeadLayerNorm)?,
                tensor.head.w.matmul_op(
                    head_x.view(.., .., .., ..)?,
                    header.head_o.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_header),
                )?,
                hook_op(Hook::PostHead)?,
            ]);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&head_ops);
            pass.execute_tensor_op(&ops);
        }

        let commands = vec![encoder.finish()];

        Ok(RunJob {
            model,
            commands,
            redirect,
            embed_device,
            tokens: buffer.tokens,
            cursors: buffer.cursors,
            input: buffer.input,
            output: header.head_o,
        })
    }
}
