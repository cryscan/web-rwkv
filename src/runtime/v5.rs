use std::{marker::PhantomData, sync::Arc};

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::DeserializeSeed;
use wgpu::CommandBuffer;

use super::{
    infer::{InferInfo, InferOutputBatch, InferRedirect, RunOutput, MIN_TOKEN_CHUNK_SIZE},
    loader::{Loader, Reader},
    model::{Build, EmbedDevice, ModelBuilder, ModelInfo, Quant},
    Job, JobBuilder,
};
use crate::{
    context::Context,
    num::Float,
    tensor::{
        kind::ReadWrite,
        matrix::Matrix,
        ops::{Activation, TensorCommand, TensorOp, TensorPass},
        shape::{Shape, TensorDimension},
        DeepClone, IntoPackedCursors, TensorCpu, TensorError, TensorGpu, TensorGpuView, TensorInit,
        TensorReshape, TensorShape, TensorStack,
    },
};

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct Model {
    pub context: Context,
    pub info: ModelInfo,
    pub tensor: ModelTensor,
}

impl Model {
    pub const RESCALE_LAYER: usize = 6;

    pub const LN_EPS: f32 = 1.0e-5;
    pub const GN_EPS: f32 = 64.0e-5;
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct ModelTensor {
    pub embed: Embed,
    pub head: Head,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct LayerNorm {
    pub w: TensorGpu<f16, ReadWrite>,
    pub b: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct Att {
    pub time_decay: TensorGpu<f32, ReadWrite>,
    pub time_first: TensorGpu<f32, ReadWrite>,

    pub time_mix_k: TensorGpu<f16, ReadWrite>,
    pub time_mix_v: TensorGpu<f16, ReadWrite>,
    pub time_mix_r: TensorGpu<f16, ReadWrite>,
    pub time_mix_g: TensorGpu<f16, ReadWrite>,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
    pub w_g: Matrix,
    pub w_o: Matrix,

    pub group_norm: LayerNorm,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct Ffn {
    pub time_mix_k: TensorGpu<f16, ReadWrite>,
    pub time_mix_r: TensorGpu<f16, ReadWrite>,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct Layer {
    pub att_layer_norm: LayerNorm,
    pub ffn_layer_norm: LayerNorm,
    pub att: Att,
    pub ffn: Ffn,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct Embed {
    pub layer_norm: LayerNorm,
    pub w: Arc<TensorCpu<'static, f16>>,
    pub u: Option<TensorGpu<f16, ReadWrite>>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct Head {
    pub layer_norm: LayerNorm,
    pub w: Matrix,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
pub struct State<const N: usize> {
    pub info: ModelInfo,
    pub data: Vec<TensorGpu<f32, ReadWrite>>,
}

impl<const N: usize> State<N> {
    fn att(&self, layer: usize) -> Result<TensorGpuView<f32>, TensorError> {
        let head_size = self.info.num_emb / self.info.num_head;
        let end = head_size + 1;
        self.data[layer].view(.., 0..end, .., ..)
    }

    fn ffn(&self, layer: usize) -> Result<TensorGpuView<f32>, TensorError> {
        let head_size = self.info.num_emb / self.info.num_head;
        let start = head_size + 1;
        self.data[layer].view(.., start, .., ..)
    }
}

impl<const N: usize> DeepClone for State<N> {
    fn deep_clone(&self) -> Self {
        let data = self.data.iter().map(|tensor| tensor.deep_clone()).collect();
        Self {
            data,
            ..self.clone()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Runtime<F: Float> {
    pub tokens: TensorGpu<u32, ReadWrite>,
    pub cursors: TensorGpu<u32, ReadWrite>,
    pub input: TensorGpu<F, ReadWrite>,

    pub att_x: TensorGpu<F, ReadWrite>,
    pub att_kx: TensorGpu<F, ReadWrite>,
    pub att_vx: TensorGpu<F, ReadWrite>,
    pub att_rx: TensorGpu<F, ReadWrite>,
    pub att_gx: TensorGpu<F, ReadWrite>,
    pub att_k: TensorGpu<f32, ReadWrite>,
    pub att_v: TensorGpu<f32, ReadWrite>,
    pub att_r: TensorGpu<f32, ReadWrite>,
    pub att_g: TensorGpu<F, ReadWrite>,
    pub att_o: TensorGpu<F, ReadWrite>,

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

        Self {
            cursors: context.tensor_init(cursors_shape),
            tokens: context.tensor_init(tokens_shape),
            input: context.tensor_init(shape),
            att_x: context.tensor_init(shape),
            att_kx: context.tensor_init(shape),
            att_vx: context.tensor_init(shape),
            att_rx: context.tensor_init(shape),
            att_gx: context.tensor_init(shape),
            att_k: context.tensor_init(shape),
            att_v: context.tensor_init(shape),
            att_r: context.tensor_init(shape),
            att_g: context.tensor_init(shape),
            att_o: context.tensor_init(shape),
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

#[derive(Debug, Clone)]
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

pub struct RunJob<F: Float> {
    commands: Vec<CommandBuffer>,
    redirect: InferRedirect,

    embed_device: EmbedDevice,
    embed: Arc<TensorCpu<'static, f16>>,

    cursors: TensorGpu<u32, ReadWrite>,
    tokens: TensorGpu<u32, ReadWrite>,
    input: TensorGpu<F, ReadWrite>,
    output: TensorGpu<F, ReadWrite>,
}

impl<F: Float> Job for RunJob<F> {
    type Input = Vec<Vec<u16>>;
    type Output = RunOutput<F>;

    fn check(&self, input: &Self::Input) -> bool {
        let num_tokens: usize = input.iter().map(|tokens| tokens.len()).sum();
        num_tokens == self.cursors.shape()[0]
    }

    fn load(self, input: &Self::Input) -> Result<Self> {
        let stack: Vec<TensorCpu<F>> = input
            .iter()
            .map(|tokens| -> Result<TensorCpu<'_, F>, _> {
                let num_emb = self.embed.shape()[0];
                let num_token = tokens.len();
                let data = self.embed.data();
                let data = tokens
                    .iter()
                    .map(|&token| {
                        let start = num_emb * token as usize;
                        let end = start + num_emb;
                        data[start..end].to_vec()
                    })
                    .concat();
                let data = data.into_iter().map(|x| F::co_hom(x)).collect_vec();
                let shape = Shape::new(num_emb, num_token, 1, 1);
                TensorCpu::from_data(shape, data)
            })
            .try_collect()?;
        let stack = TensorStack::try_from(stack)?;

        let cursors = stack.cursors.clone().into_cursors();
        let cursors = TensorCpu::from_data(self.cursors.shape(), cursors)?;
        self.cursors.load(&cursors)?;

        match self.embed_device {
            EmbedDevice::Cpu => self.input.load(&stack.tensor)?,
            EmbedDevice::Gpu => {
                let tokens = input
                    .clone()
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

    fn submit(&mut self) {
        let commands = std::mem::take(&mut self.commands);
        self.output.context.queue.submit(commands);
    }

    async fn back(self) -> Result<Self::Output> {
        let output = self.output.back().await;
        let batches: Vec<_> = self
            .redirect
            .outputs
            .into_iter()
            .map(|(start, end)| output.slice(.., start..end, .., ..))
            .try_collect()?;
        let batches = batches
            .into_iter()
            .map(|output| InferOutputBatch {
                output,
                state: None,
            })
            .collect();
        Ok(RunOutput(batches))
    }
}

pub struct ModelRuntime<F: Float, const N: usize> {
    model: Model,
    state: State<N>,
    phantom: PhantomData<F>,
}

fn turbo(num_token: usize) -> bool {
    num_token % MIN_TOKEN_CHUNK_SIZE == 0
}

fn hook_op(_: Hook) -> Result<TensorOp, TensorError> {
    Ok(TensorOp::List(vec![]))
}

impl<F: Float, const N: usize> JobBuilder<RunJob<F>> for ModelRuntime<F, N> {
    type Seed = InferInfo;

    async fn build(&self, seed: Self::Seed) -> Result<RunJob<F>> {
        let model = &self.model;
        let state = &self.state;
        let context = &model.context;
        let info = &model.info;
        let tensor = &model.tensor;

        let num_token = seed.num_token();
        let head_size = info.num_emb / info.num_head;

        let redirect = seed.redirect();
        let num_header = redirect.headers.len();

        let buffer = Runtime::<F>::new(context, info, num_token);
        let header = Header::<F>::new(context, info, num_header);

        #[cfg(feature = "async-build")]
        let mut tasks = tokio::task::JoinSet::new();
        #[cfg(not(feature = "async-build"))]
        let mut commands = Vec::new();

        let (head_ops, head_x) = if num_token == 1 || num_token == num_header {
            (vec![], buffer.ffn_x.clone())
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
            (ops, header.head_x.clone())
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
            let context = context.clone();
            let f = move || -> Result<_> {
                let ops = TensorOp::List(ops);
                let mut encoder = context.device.create_command_encoder(&Default::default());
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.execute_tensor_op(&ops);
                drop(pass);
                Ok((0, encoder.finish()))
            };
            #[cfg(feature = "async-build")]
            tasks.spawn_blocking(f);
            #[cfg(not(feature = "async-build"))]
            commands.push(f()?)
        }

        for (index, layer) in tensor.layers.iter().enumerate() {
            let context = context.clone();
            let layer = layer.clone();
            let state = state.clone();
            let buffer = buffer.clone();
            let f = move || -> Result<_> {
                Ok((
                    index + 32,
                    Self::build_layer(context, layer, state, buffer, index, num_token, head_size)?,
                ))
            };
            #[cfg(feature = "async-build")]
            tasks.spawn_blocking(f);
            #[cfg(not(feature = "async-build"))]
            commands.push(f()?)
        }

        {
            let context = context.clone();
            let head = model.tensor.head.clone();
            let header = header.clone();
            let f = move || -> Result<_> {
                Ok((
                    usize::MAX,
                    Self::build_header(context, head, header, head_x, num_header, head_ops)?,
                ))
            };
            #[cfg(feature = "async-build")]
            tasks.spawn_blocking(f);
            #[cfg(not(feature = "async-build"))]
            commands.push(f()?)
        }

        #[cfg(feature = "async-build")]
        let mut commands = vec![];
        #[cfg(feature = "async-build")]
        while let Some(result) = tasks.join_next().await {
            commands.push(result??);
        }
        let commands = commands
            .into_iter()
            .sorted_by_key(|x| x.0)
            .map(|x| x.1)
            .collect_vec();

        Ok(RunJob {
            commands,
            redirect,
            embed_device,
            embed: model.tensor.embed.w.clone(),
            tokens: buffer.tokens,
            cursors: buffer.cursors,
            input: buffer.input,
            output: header.head_o,
        })
    }
}

impl<F: Float, const N: usize> ModelRuntime<F, N> {
    #[allow(clippy::too_many_arguments)]
    fn build_layer(
        context: Context,
        layer: Layer,
        state: State<N>,
        buffer: Runtime<F>,
        index: usize,
        num_token: usize,
        head_size: usize,
    ) -> Result<CommandBuffer> {
        let info = &state.info;
        let mut encoder = context.device.create_command_encoder(&Default::default());

        use TensorDimension::{Auto, Dimension};
        let time_first =
            layer
                .att
                .time_first
                .reshape(Dimension(head_size), Auto, Dimension(1), Dimension(1))?;
        let time_decay =
            layer
                .att
                .time_decay
                .reshape(Dimension(head_size), Auto, Dimension(1), Dimension(1))?;
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
                layer.att.time_mix_k.view(.., .., .., ..)?,
                state.att(index)?,
                &buffer.att_x,
                &buffer.att_kx,
                false,
            )?,
            TensorOp::token_shift(
                &buffer.cursors,
                layer.att.time_mix_v.view(.., .., .., ..)?,
                state.att(index)?,
                &buffer.att_x,
                &buffer.att_vx,
                false,
            )?,
            TensorOp::token_shift(
                &buffer.cursors,
                layer.att.time_mix_r.view(.., .., .., ..)?,
                state.att(index)?,
                &buffer.att_x,
                &buffer.att_rx,
                false,
            )?,
            TensorOp::token_shift(
                &buffer.cursors,
                layer.att.time_mix_g.view(.., .., .., ..)?,
                state.att(index)?,
                &buffer.att_x,
                &buffer.att_gx,
                false,
            )?,
            hook_op(Hook::PostAttTokenShift(index))?,
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
            hook_op(Hook::PreAttTimeMix(index))?,
            TensorOp::blit(
                buffer.att_x.view(.., .., .., ..)?,
                buffer.aux_x.view(.., .., .., ..)?,
            )?,
            TensorOp::time_mix_v5(
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
                false,
            )?,
            TensorOp::token_shift(
                &buffer.cursors,
                layer.ffn.time_mix_r.view(.., .., .., ..)?,
                state.ffn(index)?,
                &buffer.ffn_x,
                &buffer.ffn_rx,
                false,
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

        if index != info.num_layer - 1 {
            encoder.copy_tensor(&buffer.ffn_x, &buffer.input)?;
        }

        Ok(encoder.finish())
    }

    fn build_header(
        context: Context,
        head: Head,
        header: Header<F>,
        head_x: TensorGpu<F, ReadWrite>,
        num_header: usize,
        mut ops: Vec<TensorOp>,
    ) -> Result<CommandBuffer> {
        let mut encoder = context.device.create_command_encoder(&Default::default());
        if num_header > 0 {
            ops.append(&mut vec![
                hook_op(Hook::PreHead)?,
                TensorOp::layer_norm(
                    &head.layer_norm.w,
                    &head.layer_norm.b,
                    &head_x,
                    None,
                    Model::LN_EPS,
                )?,
                hook_op(Hook::PostHeadLayerNorm)?,
                head.w.matmul_op(
                    head_x.view(.., .., .., ..)?,
                    header.head_o.view(.., .., .., ..)?,
                    Activation::None,
                    turbo(num_header),
                )?,
                hook_op(Hook::PostHead)?,
            ]);
            let ops = TensorOp::List(ops);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&ops);
        }
        Ok(encoder.finish())
    }
}

impl<F: Float, R: Reader, const N: usize> Build<ModelRuntime<F, N>> for ModelBuilder<R> {
    async fn build(self) -> Result<ModelRuntime<F, N>> {
        let ModelBuilder {
            context,
            model,
            lora,
            quant,
            embed_device,
        } = self;

        let info = Loader::info(&model)?;
        let loader = Loader {
            context: context.clone(),
            model,
            lora,
        };

        let embed = Embed {
            layer_norm: LayerNorm {
                w: loader.load_vector_f16("blocks.0.ln0.weight").await?,
                b: loader.load_vector_f16("blocks.0.ln0.bias").await?,
            },
            w: loader.load_embed().await?.into(),
            u: match embed_device {
                EmbedDevice::Cpu => None,
                EmbedDevice::Gpu => Some(loader.load_matrix_f16("emb.weight").await?),
            },
        };

        let head = Head {
            layer_norm: LayerNorm {
                w: loader.load_vector_f16("ln_out.weight").await?,
                b: loader.load_vector_f16("ln_out.bias").await?,
            },
            w: Matrix::Fp16(loader.load_matrix_f16("head.weight").await?),
        };

        context.queue.submit(None);
        context.device.poll(wgpu::MaintainBase::Wait);

        let load_matrix = |name: String, quant: Quant| loader.load_matrix(name, quant);
        let load_matrix_discount = |name: String, quant: Quant, discount: f32| {
            loader.load_matrix_discount(name, quant, discount)
        };

        let mut layers = vec![];
        for layer in 0..info.num_layer {
            let quant = quant.get(&layer).copied().unwrap_or_default();
            let discount = 2.0_f32.powi(-((layer / Model::RESCALE_LAYER) as i32));

            let att_layer_norm = LayerNorm {
                w: loader
                    .load_vector_f16(format!("blocks.{layer}.ln1.weight"))
                    .await?,
                b: loader
                    .load_vector_f16(format!("blocks.{layer}.ln1.bias"))
                    .await?,
            };

            let att = format!("blocks.{layer}.att");
            let time_decay = loader
                .load_vector_exp_exp_f32(format!("{att}.time_decay"))
                .await?;
            let time_first = loader.load_vector_f32(format!("{att}.time_first")).await?;
            let time_mix_k = loader.load_vector_f16(format!("{att}.time_mix_k")).await?;
            let time_mix_v = loader.load_vector_f16(format!("{att}.time_mix_v")).await?;
            let time_mix_r = loader.load_vector_f16(format!("{att}.time_mix_r")).await?;
            let time_mix_g = loader.load_vector_f16(format!("{att}.time_mix_g")).await?;

            let group_norm = LayerNorm {
                w: loader
                    .load_vector_f16(format!("{att}.ln_x.weight"))
                    .await?
                    .reshape(
                        TensorDimension::Auto,
                        TensorDimension::Dimension(info.num_head),
                        TensorDimension::Dimension(1),
                        TensorDimension::Dimension(1),
                    )?,
                b: loader
                    .load_vector_f16(format!("{att}.ln_x.bias"))
                    .await?
                    .reshape(
                        TensorDimension::Auto,
                        TensorDimension::Dimension(info.num_head),
                        TensorDimension::Dimension(1),
                        TensorDimension::Dimension(1),
                    )?,
            };

            let att = Att {
                time_decay,
                time_first,
                time_mix_k,
                time_mix_v,
                time_mix_r,
                time_mix_g,
                w_k: load_matrix(format!("{att}.key.weight"), quant).await?,
                w_v: load_matrix(format!("{att}.value.weight"), quant).await?,
                w_r: load_matrix(format!("{att}.receptance.weight"), quant).await?,
                w_g: load_matrix(format!("{att}.gate.weight"), quant).await?,
                w_o: load_matrix_discount(format!("{att}.output.weight"), quant, discount).await?,
                group_norm,
            };

            let ffn_layer_norm = LayerNorm {
                w: loader
                    .load_vector_f16(format!("blocks.{layer}.ln2.weight"))
                    .await?,
                b: loader
                    .load_vector_f16(format!("blocks.{layer}.ln2.bias"))
                    .await?,
            };

            let ffn = format!("blocks.{layer}.ffn");
            let time_mix_k = loader.load_vector_f16(format!("{ffn}.time_mix_k")).await?;
            let time_mix_r = loader.load_vector_f16(format!("{ffn}.time_mix_r")).await?;

            let ffn = Ffn {
                time_mix_k,
                time_mix_r,
                w_r: load_matrix(format!("{ffn}.receptance.weight"), quant).await?,
                w_k: load_matrix(format!("{ffn}.key.weight"), quant).await?,
                w_v: load_matrix_discount(format!("{ffn}.value.weight"), quant, discount).await?,
            };

            context.queue.submit(None);
            context.device.poll(wgpu::MaintainBase::Wait);

            layers.push(Layer {
                att_layer_norm,
                ffn_layer_norm,
                att,
                ffn,
            })
        }

        context.queue.submit(None);
        context.device.poll(wgpu::MaintainBase::Wait);

        let tensor = ModelTensor {
            embed,
            head,
            layers,
        };

        let model = {
            let context = context.clone();
            let info = info.clone();
            Model {
                context,
                info,
                tensor,
            }
        };

        let state = {
            let head_size = info.num_emb / info.num_head;
            let shape = Shape::new(info.num_emb, head_size + 2, N, 1);
            let data = (0..info.num_layer).map(|_| context.zeros(shape)).collect();
            State { info, data }
        };

        Ok(ModelRuntime {
            model,
            state,
            phantom: PhantomData,
        })
    }
}
