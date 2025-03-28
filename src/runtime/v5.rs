use std::{collections::HashMap, marker::PhantomData, sync::Arc};

#[cfg(not(target_arch = "wasm32"))]
use futures::future::BoxFuture;
#[cfg(target_arch = "wasm32")]
use futures::future::LocalBoxFuture;
use half::f16;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::DeserializeSeed;
use wgpu::CommandBuffer;

use super::{
    infer::{RnnChunk, RnnInfo, RnnInput, RnnOutput, RnnOutputBatch, RnnRedirect, Token},
    loader::{Loader, LoaderError, Reader},
    model::{AsAny, ModelBuilder, ModelInfo, Quant, State as _},
    Dispatcher, Job, RuntimeError,
};
use crate::{
    context::Context,
    num::Float,
    tensor::{
        cache::ResourceCache,
        kind::ReadWrite,
        matrix::Matrix,
        ops::{Activation, TensorCommand, TensorOp},
        serialization::Seed,
        shape::{Shape, TensorDimension},
        DeepClone, IntoPackedCursors, TensorCpu, TensorError, TensorGpu, TensorGpuView, TensorInit,
        TensorReshape, TensorShape, TensorStack,
    },
};

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Model {
    pub context: Context,
    pub info: ModelInfo,
    pub rescale: usize,
    pub sep: usize,
    pub tensor: ModelTensor,
}

impl Model {
    pub const LN_EPS: f32 = 1.0e-5;
    pub const GN_EPS: f32 = 64.0e-5;

    pub const DEFAULT_RESCALE: usize = 6;
    pub const DEFAULT_SEP: usize = 1024;
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct ModelTensor {
    pub embed: Embed,
    pub head: Head,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct LayerNorm {
    pub w: TensorGpu<f16, ReadWrite>,
    pub b: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
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
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Ffn {
    pub time_mix_k: TensorGpu<f16, ReadWrite>,
    pub time_mix_r: TensorGpu<f16, ReadWrite>,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Layer {
    pub att_layer_norm: LayerNorm,
    pub ffn_layer_norm: LayerNorm,
    pub att: Att,
    pub ffn: Ffn,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Embed {
    pub layer_norm: LayerNorm,
    pub w: TensorCpu<f16>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Head {
    pub layer_norm: LayerNorm,
    pub w: Matrix,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct State {
    pub context: Context,
    pub info: ModelInfo,
    pub data: Vec<TensorGpu<f32, ReadWrite>>,
}

impl State {
    async fn back(&self, batch: usize) -> Result<TensorCpu<f32>, TensorError> {
        let context = &self.context;
        let mut tensors = Vec::with_capacity(self.info.num_layer);
        let mut encoder = context.device.create_command_encoder(&Default::default());
        for data in self.data.iter() {
            let shape = data.shape();
            let destination = context.tensor_init([shape[0], shape[1], 1, 1]);
            encoder.copy_tensor_batch(data, &destination, batch, 0)?;
            tensors.push(destination);
        }
        context.queue.submit(Some(encoder.finish()));

        let mut backed = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            backed.push(tensor.back().await);
        }
        TensorCpu::stack(backed, 2)
    }
}

impl AsAny for State {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl super::model::State for State {
    #[inline]
    fn num_batch(&self) -> usize {
        self.data[0].shape()[2]
    }

    #[inline]
    fn init_shape(&self) -> Shape {
        let info = &self.info;
        let head_size = info.num_emb / info.num_head;
        [info.num_emb, head_size + 2, info.num_layer, 1].into()
    }

    fn init(&self) -> TensorCpu<f32> {
        let shape = self.init_shape();
        let data = vec![0.0; shape.len()];
        TensorCpu::from_data(shape, data).unwrap()
    }

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

    fn load(&self, tensor: TensorCpu<f32>, batch: usize) -> Result<(), TensorError> {
        let head_size = self.info.num_emb / self.info.num_head;
        tensor.check_shape([self.info.num_emb, head_size + 2, self.info.num_layer, 1])?;
        for (data, source) in self.data.iter().zip(tensor.split(2)?.into_iter()) {
            data.load_batch(&source, batch)?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn back(&self, batch: usize) -> BoxFuture<Result<TensorCpu<f32>, TensorError>> {
        Box::pin(self.back(batch))
    }

    #[cfg(target_arch = "wasm32")]
    fn back(&self, batch: usize) -> LocalBoxFuture<Result<TensorCpu<f32>, TensorError>> {
        Box::pin(self.back(batch))
    }

    fn write(&self, tensor: TensorGpu<f32, ReadWrite>, batch: usize) -> Result<(), TensorError> {
        let head_size = self.info.num_emb / self.info.num_head;
        tensor.check_shape([self.info.num_emb, head_size + 2, self.info.num_layer, 1])?;

        let context = &self.context;
        let mut ops = Vec::with_capacity(self.data.len());
        for (layer, data) in self.data.iter().enumerate() {
            ops.push(TensorOp::blit(
                tensor.view(.., .., layer, ..)?,
                data.view(.., .., batch, ..)?,
            )?);
        }
        context.queue.submit(context.encode(&TensorOp::List(ops)));

        Ok(())
    }

    fn read(&self, batch: usize) -> Result<TensorGpu<f32, ReadWrite>, TensorError> {
        let context = &self.context;
        let head_size = self.info.num_emb / self.info.num_head;
        let shape = [self.info.num_emb, head_size + 2, self.info.num_layer, 1];
        let tensor: TensorGpu<_, _> = context.tensor_init(shape);

        let mut ops = Vec::with_capacity(self.data.len());
        for (layer, data) in self.data.iter().enumerate() {
            ops.push(TensorOp::blit(
                data.view(.., .., batch, ..)?,
                tensor.view(.., .., layer, ..)?,
            )?);
        }
        context.queue.submit(context.encode(&TensorOp::List(ops)));

        Ok(tensor)
    }

    fn embed(&self, layer: usize, backed: TensorCpu<f32>) -> Result<TensorCpu<f32>, TensorError> {
        backed.slice(.., 0, layer, ..)
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

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Runtime<F: Float> {
    pub cursors: TensorGpu<u32, ReadWrite>,
    pub input: TensorGpu<f16, ReadWrite>,

    pub x: TensorGpu<F, ReadWrite>,
    pub aux_x: TensorGpu<f32, ReadWrite>,

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
}

impl<F: Float> Runtime<F> {
    pub fn new(context: &Context, info: &ModelInfo, num_token: usize) -> Self {
        let shape = Shape::new(info.num_emb, num_token, 1, 1);
        let cursors_shape = Shape::new(num_token, 1, 1, 1);
        let hidden_shape = Shape::new(info.num_hidden, num_token, 1, 1);

        Self {
            cursors: context.tensor_init(cursors_shape),
            input: context.tensor_init(shape),
            x: context.tensor_init(shape),
            aux_x: context.tensor_init(shape),
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
        }
    }
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Header<F: Float> {
    pub head_x: TensorGpu<F, ReadWrite>,
    pub head_o: TensorGpu<f32, ReadWrite>,
}

impl<F: Float> Header<F> {
    pub fn new(context: &Context, info: &ModelInfo, num_header: usize) -> Self {
        let head_shape = Shape::new(info.num_emb, num_header, 1, 1);
        let output_shape = Shape::new(info.num_vocab_padded(), num_header, 1, 1);

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

pub struct RnnJob {
    commands: Vec<CommandBuffer>,
    redirect: RnnRedirect,

    embed: TensorCpu<f16>,

    cursors: TensorGpu<u32, ReadWrite>,
    input: TensorGpu<f16, ReadWrite>,
    output: TensorGpu<f32, ReadWrite>,
}

impl Job for RnnJob {
    type Input = RnnInput;
    type Output = RnnOutput;

    fn load(&self, input: &RnnChunk) -> Result<(), RuntimeError> {
        if input.num_token() == 0 {
            return Ok(());
        }

        let stack: Vec<TensorCpu<f16>> = input
            .iter()
            .map(|chunk| {
                let num_emb = self.embed.shape()[0];
                let data = self.embed.data();
                let data = chunk
                    .iter()
                    .map(|token| match token {
                        &Token::Token(token) => {
                            let start = num_emb * token as usize;
                            let end = start + num_emb;
                            let data = data[start..end].to_vec();
                            TensorCpu::from_data_1d(data)
                        }
                        Token::Embed(tensor) => tensor.clone(),
                    })
                    .collect_vec();
                match TensorCpu::stack(data, 1) {
                    Ok(tensor) => tensor,
                    Err(_) => TensorCpu::init([num_emb, 0, 1, 1]),
                }
            })
            .collect();
        let stack = TensorStack::try_from(stack)?;

        let cursors = stack.cursors.clone().into_cursors();
        let cursors = TensorCpu::from_data(self.cursors.shape(), cursors)?;
        self.cursors.load(&cursors)?;
        self.input.load(&stack.tensor)?;

        Ok(())
    }

    fn submit(&mut self) {
        let commands = std::mem::take(&mut self.commands);
        self.output.context.queue.submit(commands);
    }

    async fn back(self) -> Result<Self::Output, RuntimeError> {
        let output = self.output.back().await;
        let batches: Vec<_> = self
            .redirect
            .outputs
            .into_iter()
            .map(|(start, end)| output.slice(.., start..end, .., ..))
            .try_collect()?;
        let batches = batches.into_iter().map(RnnOutputBatch).collect();
        Ok(RnnOutput(batches))
    }
}

#[derive(Debug, Clone)]
pub struct Frame<F: Float> {
    pub state: State,
    pub buffer: Arc<Runtime<F>>,
    pub header: Arc<Header<F>>,
}

pub type HookFn<F> = Box<dyn Fn(Frame<F>) -> Result<TensorOp, TensorError> + Send + Sync>;
pub type HookMap<F> = HashMap<Hook, HookFn<F>>;

#[derive(Clone)]
pub struct Bundle<F: Float> {
    model: Model,
    state: State,
    hooks: Arc<HookMap<F>>,
    buffers: ResourceCache<usize, Runtime<F>>,
    headers: ResourceCache<usize, Header<F>>,
    phantom: PhantomData<F>,
}

impl<F: Float> Bundle<F> {
    pub fn new(model: Model, num_batch: usize) -> Self {
        let context = model.context.clone();
        let info = model.info.clone();
        let state = {
            let head_size = info.num_emb / info.num_head;
            let shape = Shape::new(info.num_emb, head_size + 2, num_batch, 1);
            let data = (0..info.num_layer).map(|_| context.zeros(shape)).collect();
            State {
                context,
                info,
                data,
            }
        };
        Self {
            model,
            state,
            hooks: Default::default(),
            buffers: ResourceCache::new(4),
            headers: ResourceCache::new(4),
            phantom: PhantomData,
        }
    }

    pub fn new_with_hooks(model: Model, num_batch: usize, hooks: HookMap<F>) -> Self {
        Self {
            hooks: Arc::new(hooks),
            ..Self::new(model, num_batch)
        }
    }

    fn checkout_buffer(
        &self,
        context: &Context,
        info: &ModelInfo,
        num_token: usize,
    ) -> Arc<Runtime<F>> {
        self.buffers
            .checkout(num_token, || Runtime::new(context, info, num_token))
    }

    fn checkout_header(
        &self,
        context: &Context,
        info: &ModelInfo,
        num_header: usize,
    ) -> Arc<Header<F>> {
        self.headers
            .checkout(num_header, || Header::new(context, info, num_header))
    }
}

impl<F: Float> super::model::Bundle for Bundle<F> {
    #[inline]
    fn info(&self) -> ModelInfo {
        self.model.info.clone()
    }

    #[inline]
    fn state(&self) -> impl super::model::State + AsAny + 'static {
        self.state.clone()
    }

    fn model(&self) -> impl Serialize + 'static {
        self.model.clone()
    }
}

fn turbo(num_token: usize) -> bool {
    num_token % super::infer::rnn::MIN_TOKEN_CHUNK_SIZE == 0
}

fn hook_op<F: Float>(
    hooks: &HookMap<F>,
    hook: &Hook,
    frame: &Frame<F>,
) -> Result<TensorOp, TensorError> {
    match hooks.get(hook) {
        Some(f) => f(frame.clone()),
        None => Ok(TensorOp::empty()),
    }
}

impl<F: Float> Dispatcher<RnnJob> for Bundle<F> {
    type Info = RnnInfo;

    fn dispatch(&self, seed: Self::Info) -> Result<RnnJob, RuntimeError> {
        let model = &self.model;
        let state = &self.state;
        let context = &model.context;
        let info = &model.info;
        let tensor = &model.tensor;

        let num_token = seed.num_token();
        let head_size = info.num_emb / info.num_head;

        let redirect = seed.redirect();
        let num_header = redirect.headers.len();

        let buffer = self.checkout_buffer(context, info, num_token);
        let header = self.checkout_header(context, info, num_header);
        let frame = Frame {
            state: state.clone(),
            buffer: buffer.clone(),
            header: header.clone(),
        };

        context.maintain();

        if num_token == 0 {
            return Ok(RnnJob {
                commands: vec![],
                redirect,
                embed: model.tensor.embed.w.clone(),
                cursors: buffer.cursors.clone(),
                input: buffer.input.clone(),
                output: header.head_o.clone(),
            });
        }

        #[cfg(feature = "trace")]
        let _span = tracing::trace_span!("build").entered();

        let (head_op, head_x) = redirect.op(&buffer.x, &header.head_x)?;

        let hook_op = |hook: Hook| hook_op(&self.hooks, &hook, &frame);
        let mut ops = vec![];

        {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("embed").entered();

            ops.extend([
                hook_op(Hook::PostEmbedLoaded)?,
                TensorOp::layer_norm(
                    &tensor.embed.layer_norm.w,
                    &tensor.embed.layer_norm.b,
                    &buffer.input,
                    Model::LN_EPS,
                )?,
                TensorOp::blit(&buffer.input, &buffer.x)?,
                hook_op(Hook::PostEmbedLayerNorm)?,
            ]);
        };

        for (index, layer) in tensor.layers.iter().enumerate() {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("layer", index).entered();

            let hooks = self.hooks.clone();
            let frame = frame.clone();
            let layer = layer.clone();

            let op = dispatch_layer(
                hooks,
                frame,
                layer,
                index,
                num_token,
                head_size,
                model.rescale,
            )?;
            ops.push(op);

            if (index + 1) % model.sep == 0 {
                ops.push(TensorOp::Sep);
            }
        }

        {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("header").entered();

            let hooks = self.hooks.clone();
            let frame = frame.clone();
            let head = model.tensor.head.clone();

            let op = dispatch_header(hooks, frame, head, head_x, num_header, head_op)?;
            ops.push(op);
        }

        let commands = {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("encode").entered();
            context.encode(&TensorOp::List(ops))
        };

        Ok(RnnJob {
            commands,
            redirect,
            embed: model.tensor.embed.w.clone(),
            cursors: buffer.cursors.clone(),
            input: buffer.input.clone(),
            output: header.head_o.clone(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_layer<F: Float>(
    hooks: Arc<HookMap<F>>,
    frame: Frame<F>,
    layer: Layer,
    index: usize,
    num_token: usize,
    head_size: usize,
    rescale: usize,
) -> Result<TensorOp, TensorError> {
    let hook_op = |hook: Hook| hook_op(&hooks, &hook, &frame);
    let Frame { state, buffer, .. } = &frame;

    let time_first = layer.att.time_first.reshape(
        TensorDimension::Size(head_size),
        TensorDimension::Auto,
        TensorDimension::Size(1),
        TensorDimension::Size(1),
    )?;
    let time_decay = layer.att.time_decay.reshape(
        TensorDimension::Size(head_size),
        TensorDimension::Auto,
        TensorDimension::Size(1),
        TensorDimension::Size(1),
    )?;
    let aux_x = buffer.aux_x.reshape(
        TensorDimension::Size(head_size),
        TensorDimension::Auto,
        TensorDimension::Size(num_token),
        TensorDimension::Size(1),
    )?;
    let att_k = buffer.att_k.reshape(
        TensorDimension::Size(head_size),
        TensorDimension::Auto,
        TensorDimension::Size(num_token),
        TensorDimension::Size(1),
    )?;
    let att_v = buffer.att_v.reshape(
        TensorDimension::Size(head_size),
        TensorDimension::Auto,
        TensorDimension::Size(num_token),
        TensorDimension::Size(1),
    )?;
    let att_r = buffer.att_r.reshape(
        TensorDimension::Size(head_size),
        TensorDimension::Auto,
        TensorDimension::Size(num_token),
        TensorDimension::Size(1),
    )?;

    let mut ops = vec![];

    ops.extend([
        TensorOp::blit(&buffer.x, &buffer.att_x)?,
        hook_op(Hook::PreAtt(index))?,
        TensorOp::layer_norm(
            &layer.att_layer_norm.w,
            &layer.att_layer_norm.b,
            &buffer.att_x,
            Model::LN_EPS,
        )?,
        hook_op(Hook::PostAttLayerNorm(index))?,
        hook_op(Hook::PreAttTokenShift(index))?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.att.time_mix_k,
            state.att(index)?,
            &buffer.att_x,
            &buffer.att_kx,
            false,
        )?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.att.time_mix_v,
            state.att(index)?,
            &buffer.att_x,
            &buffer.att_vx,
            false,
        )?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.att.time_mix_r,
            state.att(index)?,
            &buffer.att_x,
            &buffer.att_rx,
            false,
        )?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.att.time_mix_g,
            state.att(index)?,
            &buffer.att_x,
            &buffer.att_gx,
            false,
        )?,
        hook_op(Hook::PostAttTokenShift(index))?,
        hook_op(Hook::PreAttLinear(index))?,
        layer.att.w_k.matmul_op(
            &buffer.att_kx,
            &buffer.att_k,
            Activation::None,
            turbo(num_token),
        )?,
        layer.att.w_v.matmul_op(
            &buffer.att_vx,
            &buffer.att_v,
            Activation::None,
            turbo(num_token),
        )?,
        layer.att.w_r.matmul_op(
            &buffer.att_rx,
            &buffer.att_r,
            Activation::None,
            turbo(num_token),
        )?,
        layer.att.w_g.matmul_op(
            &buffer.att_gx,
            &buffer.att_g,
            Activation::None,
            turbo(num_token),
        )?,
        hook_op(Hook::PostAttLinear(index))?,
        hook_op(Hook::PreAttTimeMix(index))?,
        TensorOp::blit(&buffer.att_x, &buffer.aux_x)?,
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
        TensorOp::blit(&buffer.aux_x, &buffer.att_x)?,
        hook_op(Hook::PostAttTimeMix(index))?,
        hook_op(Hook::PreAttGate(index))?,
        TensorOp::mul_activate(
            &buffer.att_g,
            &buffer.att_x,
            Activation::Silu,
            Activation::None,
            Activation::None,
        )?,
        hook_op(Hook::PostAttGate(index))?,
        hook_op(Hook::PreAttOut(index))?,
        layer.att.w_o.matmul_op(
            &buffer.att_x,
            &buffer.att_o,
            Activation::None,
            turbo(num_token),
        )?,
        hook_op(Hook::PostAttOut(index))?,
        TensorOp::add(&buffer.att_o, &buffer.x)?,
        hook_op(Hook::PostAtt(index))?,
    ]);

    ops.extend([
        TensorOp::blit(&buffer.x, &buffer.ffn_x)?,
        hook_op(Hook::PreFfn(index))?,
        TensorOp::layer_norm(
            &layer.ffn_layer_norm.w,
            &layer.ffn_layer_norm.b,
            &buffer.ffn_x,
            Model::LN_EPS,
        )?,
        hook_op(Hook::PostFfnLayerNorm(index))?,
        hook_op(Hook::PreFfnTokenShift(index))?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.ffn.time_mix_k,
            state.ffn(index)?,
            &buffer.ffn_x,
            &buffer.ffn_kx,
            false,
        )?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.ffn.time_mix_r,
            state.ffn(index)?,
            &buffer.ffn_x,
            &buffer.ffn_rx,
            false,
        )?,
        hook_op(Hook::PostFfnTokenShift(index))?,
        hook_op(Hook::PreFfnLinear(index))?,
        layer.ffn.w_k.matmul_op(
            &buffer.ffn_kx,
            &buffer.ffn_k,
            Activation::SquaredRelu,
            turbo(num_token),
        )?,
        hook_op(Hook::PostFfnActivate(index))?,
        layer.ffn.w_v.matmul_op_sparse(
            &buffer.ffn_k,
            &buffer.ffn_v,
            Activation::None,
            turbo(num_token),
        )?,
        layer.ffn.w_r.matmul_op(
            &buffer.ffn_rx,
            &buffer.ffn_r,
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
        TensorOp::add(&buffer.ffn_x, &buffer.x)?,
        hook_op(Hook::PostFfn(index))?,
    ]);

    if (index + 1) % rescale == 0 {
        ops.push(TensorOp::affine(&buffer.x, 0.5, 0.0)?);
    }

    Ok(TensorOp::List(ops))
}

fn dispatch_header<F: Float>(
    hooks: Arc<HookMap<F>>,
    frame: Frame<F>,
    head: Head,
    head_x: TensorGpu<F, ReadWrite>,
    num_header: usize,
    head_op: TensorOp,
) -> Result<TensorOp, TensorError> {
    let hook_op = |hook: Hook| hook_op(&hooks, &hook, &frame);
    let header = &frame.header;
    let mut ops = vec![head_op];

    if num_header > 0 {
        ops.extend([
            hook_op(Hook::PreHead)?,
            TensorOp::layer_norm(
                &head.layer_norm.w,
                &head.layer_norm.b,
                &head_x,
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
    }
    Ok(TensorOp::List(ops))
}

impl<R: Reader> ModelBuilder<R> {
    pub async fn build_v5(self) -> Result<Model, LoaderError> {
        let ModelBuilder {
            context,
            model,
            rescale,
            sep,
            lora,
            quant,
            ..
        } = self;

        let rescale = rescale.unwrap_or(Model::DEFAULT_RESCALE);
        let sep = sep.unwrap_or(Model::DEFAULT_SEP);

        let info = Loader::info(&model)?;
        let loader = Loader {
            context: context.clone(),
            model,
            lora,
        };

        let embed = Embed {
            layer_norm: LayerNorm {
                w: loader.load_vector_f16("blocks.0.ln0.weight")?,
                b: loader.load_vector_f16("blocks.0.ln0.bias")?,
            },
            w: loader.load_matrix_f16_padded_cpu("emb.weight")?,
        };

        let head = Head {
            layer_norm: LayerNorm {
                w: loader.load_vector_f16("ln_out.weight")?,
                b: loader.load_vector_f16("ln_out.bias")?,
            },
            w: Matrix::Fp16(loader.load_matrix_f16("head.weight")?),
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
            let discount = 2.0_f32.powi(-((layer / rescale) as i32));

            let att_layer_norm = LayerNorm {
                w: loader.load_vector_f16(format!("blocks.{layer}.ln1.weight"))?,
                b: loader.load_vector_f16(format!("blocks.{layer}.ln1.bias"))?,
            };

            let att = format!("blocks.{layer}.att");
            let time_decay = loader.load_vector_exp_exp_f32(format!("{att}.time_decay"))?;
            let time_first = loader.load_vector_f32(format!("{att}.time_first"))?;
            let time_mix_k = loader.load_vector_f16(format!("{att}.time_mix_k"))?;
            let time_mix_v = loader.load_vector_f16(format!("{att}.time_mix_v"))?;
            let time_mix_r = loader.load_vector_f16(format!("{att}.time_mix_r"))?;
            let time_mix_g = loader.load_vector_f16(format!("{att}.time_mix_g"))?;

            let group_norm = LayerNorm {
                w: loader
                    .load_vector_f16(format!("{att}.ln_x.weight"))?
                    .reshape(
                        TensorDimension::Auto,
                        TensorDimension::Size(info.num_head),
                        TensorDimension::Size(1),
                        TensorDimension::Size(1),
                    )?,
                b: loader
                    .load_vector_f16(format!("{att}.ln_x.bias"))?
                    .reshape(
                        TensorDimension::Auto,
                        TensorDimension::Size(info.num_head),
                        TensorDimension::Size(1),
                        TensorDimension::Size(1),
                    )?,
            };

            let att = Att {
                time_decay,
                time_first,
                time_mix_k,
                time_mix_v,
                time_mix_r,
                time_mix_g,
                w_k: load_matrix(format!("{att}.key.weight"), quant)?,
                w_v: load_matrix(format!("{att}.value.weight"), quant)?,
                w_r: load_matrix(format!("{att}.receptance.weight"), quant)?,
                w_g: load_matrix(format!("{att}.gate.weight"), quant)?,
                w_o: load_matrix_discount(format!("{att}.output.weight"), quant, discount)?,
                group_norm,
            };

            let ffn_layer_norm = LayerNorm {
                w: loader.load_vector_f16(format!("blocks.{layer}.ln2.weight"))?,
                b: loader.load_vector_f16(format!("blocks.{layer}.ln2.bias"))?,
            };

            let ffn = format!("blocks.{layer}.ffn");
            let time_mix_k = loader.load_vector_f16(format!("{ffn}.time_mix_k"))?;
            let time_mix_r = loader.load_vector_f16(format!("{ffn}.time_mix_r"))?;

            let ffn = Ffn {
                time_mix_k,
                time_mix_r,
                w_r: load_matrix(format!("{ffn}.receptance.weight"), quant)?,
                w_k: load_matrix(format!("{ffn}.key.weight"), quant)?,
                w_v: load_matrix_discount(format!("{ffn}.value.weight"), quant, discount)?,
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
                rescale,
                sep,
                tensor,
            }
        };
        Ok(model)
    }
}

/// Read the pre-trained state from the file.
pub async fn read_state<R: Reader>(
    context: &Context,
    info: &ModelInfo,
    model: R,
) -> Result<TensorCpu<f32>, LoaderError> {
    let loader = Loader {
        context: context.clone(),
        model,
        lora: vec![],
    };

    let head_size = info.num_emb / info.num_head;
    let data: TensorGpu<f32, _> = context.zeros([info.num_emb, head_size + 2, info.num_layer, 1]);

    let mut ops = vec![];
    for layer in 0..info.num_layer {
        let matrix = loader.load_matrix_f16(format!("blocks.{layer}.att.time_state"))?;
        let state: TensorGpu<_, _> = context.tensor_init([head_size, info.num_head, head_size, 1]);
        let reshaped: TensorGpu<f16, _> = state.reshape(
            TensorDimension::Size(info.num_emb),
            TensorDimension::Size(head_size),
            TensorDimension::Size(1),
            TensorDimension::Auto,
        )?;
        ops.extend([
            TensorOp::transpose(&matrix, &state)?,
            TensorOp::blit(&reshaped, data.view(.., 1..head_size + 1, layer, ..)?)?,
        ]);
    }
    context.queue.submit(context.encode(&TensorOp::List(ops)));

    Ok(data.back().await)
}
