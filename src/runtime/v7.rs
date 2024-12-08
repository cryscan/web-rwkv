use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use anyhow::Result;
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
    infer::{InferChunk, InferInfo, InferInput, InferOutput, InferOutputBatch, InferRedirect},
    model::{AsAny, EmbedDevice, ModelCustomInfo, ModelInfo, State as _},
    Dispatcher, Job,
};
use crate::{
    context::Context,
    num::Float,
    tensor::{
        kind::ReadWrite,
        matrix::Matrix,
        ops::{Activation, BinaryActivation, TensorCommand, TensorOp},
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
    pub tensor: ModelTensor,
}

impl Model {
    pub const L2_EPS: f32 = 1.0e-12;
    pub const LN_EPS: f32 = 1.0e-5;
    pub const GN_EPS: f32 = 64.0e-5;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CustomInfo {
    pub w: usize,
    pub a: usize,
    pub g: usize,
    pub v: usize,
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
    pub x_r: TensorGpu<f16, ReadWrite>,
    pub x_w: TensorGpu<f16, ReadWrite>,
    pub x_k: TensorGpu<f16, ReadWrite>,
    pub x_v: TensorGpu<f16, ReadWrite>,
    pub x_a: TensorGpu<f16, ReadWrite>,
    pub x_g: TensorGpu<f16, ReadWrite>,

    pub w0: TensorGpu<f16, ReadWrite>,
    pub a0: TensorGpu<f16, ReadWrite>,
    pub v0: TensorGpu<f16, ReadWrite>,

    pub w1: Matrix,
    pub w2: Matrix,
    pub a1: Matrix,
    pub a2: Matrix,
    pub g1: Matrix,
    pub g2: Matrix,
    pub v1: Matrix,
    pub v2: Matrix,

    pub r_k: TensorGpu<f16, ReadWrite>,
    pub k_k: TensorGpu<f16, ReadWrite>,
    pub k_a: TensorGpu<f16, ReadWrite>,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
    pub w_o: Matrix,

    pub ln: LayerNorm,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Ffn {
    pub x_k: TensorGpu<f16, ReadWrite>,

    pub w_k: TensorGpu<f16, ReadWrite>,
    pub w_v: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Layer {
    pub att_ln: LayerNorm,
    pub ffn_ln: LayerNorm,
    pub att: Att,
    pub ffn: Ffn,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Embed {
    pub ln: LayerNorm,
    pub w: TensorCpu<f16>,
    pub u: Option<TensorGpu<f16, ReadWrite>>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Head {
    pub ln: LayerNorm,
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
        TensorCpu::stack(backed)
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

    fn init(&self) -> TensorCpu<f32> {
        let info = &self.info;
        let head_size = info.num_emb / info.num_head;
        let shape = Shape::new(info.num_emb, head_size + 2, info.num_layer, 1);
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

#[derive(Debug, Clone)]
pub struct Runtime<F: Float> {
    pub cursors: TensorGpu<u32, ReadWrite>,
    pub tokens: TensorGpu<u32, ReadWrite>,
    pub input: TensorGpu<f16, ReadWrite>,

    pub x: TensorGpu<F, ReadWrite>,
    pub aux_x: TensorGpu<f32, ReadWrite>,

    pub att_x: TensorGpu<F, ReadWrite>,
    pub att_v0: TensorGpu<F, ReadWrite>,

    pub att_rx: TensorGpu<F, ReadWrite>,
    pub att_wx: TensorGpu<F, ReadWrite>,
    pub att_kx: TensorGpu<F, ReadWrite>,
    pub att_vx: TensorGpu<F, ReadWrite>,
    pub att_ax: TensorGpu<F, ReadWrite>,
    pub att_gx: TensorGpu<F, ReadWrite>,

    pub att_r: TensorGpu<F, ReadWrite>,
    pub att_w: TensorGpu<F, ReadWrite>,
    pub att_k: TensorGpu<F, ReadWrite>,
    pub att_v: TensorGpu<F, ReadWrite>,
    pub att_a: TensorGpu<F, ReadWrite>,
    pub att_g: TensorGpu<F, ReadWrite>,

    pub att_kk: TensorGpu<F, ReadWrite>,
    pub att_vv: TensorGpu<F, ReadWrite>,

    pub att_kv: TensorGpu<F, ReadWrite>,
    pub att_ab: TensorGpu<F, ReadWrite>,

    /// Time decay LoRA intermediate.
    pub temp_w: TensorGpu<F, ReadWrite>,
    pub temp_a: TensorGpu<F, ReadWrite>,
    pub temp_g: TensorGpu<F, ReadWrite>,
    pub temp_v: TensorGpu<F, ReadWrite>,

    pub ffn_x: TensorGpu<F, ReadWrite>,
    pub ffn_kx: TensorGpu<F, ReadWrite>,
    pub ffn_k: TensorGpu<F, ReadWrite>,
    pub ffn_v: TensorGpu<F, ReadWrite>,
}

impl<F: Float> Runtime<F> {
    pub fn new(context: &Context, info: &ModelInfo, num_token: usize) -> Self {
        let ModelCustomInfo::V7(custom) = info.custom else {
            unreachable!()
        };

        let shape = Shape::new(info.num_emb, num_token, 1, 1);
        let cursors_shape = Shape::new(num_token, 1, 1, 1);
        let tokens_shape = Shape::new(num_token, 1, 1, 1);
        let hidden_shape = Shape::new(info.num_hidden, num_token, 1, 1);
        let att_kv_shape = Shape::new(info.num_emb, num_token, 2, 1);
        let att_ab_shape = Shape::new(info.num_emb, num_token, 2, 1);
        let temp_w_shape = Shape::new(custom.w, num_token, 1, 1);
        let temp_a_shape = Shape::new(custom.a, num_token, 1, 1);
        let temp_g_shape = Shape::new(custom.g, num_token, 1, 1);
        let temp_v_shape = Shape::new(custom.v, num_token, 1, 1);

        Self {
            cursors: context.tensor_init(cursors_shape),
            tokens: context.tensor_init(tokens_shape),
            input: context.tensor_init(shape),
            x: context.tensor_init(shape),
            aux_x: context.tensor_init(shape),
            att_x: context.tensor_init(shape),
            att_v0: context.tensor_init(shape),
            att_rx: context.tensor_init(shape),
            att_wx: context.tensor_init(shape),
            att_kx: context.tensor_init(shape),
            att_vx: context.tensor_init(shape),
            att_ax: context.tensor_init(shape),
            att_gx: context.tensor_init(shape),
            att_r: context.tensor_init(shape),
            att_w: context.tensor_init(shape),
            att_k: context.tensor_init(shape),
            att_v: context.tensor_init(shape),
            att_a: context.tensor_init(shape),
            att_g: context.tensor_init(shape),
            att_kk: context.tensor_init(shape),
            att_vv: context.tensor_init(shape),
            att_kv: context.tensor_init(att_kv_shape),
            att_ab: context.tensor_init(att_ab_shape),
            temp_w: context.tensor_init(temp_w_shape),
            temp_a: context.tensor_init(temp_a_shape),
            temp_g: context.tensor_init(temp_g_shape),
            temp_v: context.tensor_init(temp_v_shape),
            ffn_x: context.tensor_init(shape),
            ffn_kx: context.tensor_init(shape),
            ffn_k: context.tensor_init(hidden_shape),
            ffn_v: context.tensor_init(shape),
        }
    }
}

#[derive(Debug, Clone)]
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
    PreAttLora(usize),
    PostAttLora(usize),
    PreAttControl(usize),
    PostAttControl(usize),
    PreAttDeltaRule(usize),
    PostAttDeltaRule(usize),
    PreAttTimeDecayActivate(usize),
    PostAttTimeDecayActivate(usize),
    PreAttTimeMix(usize),
    PostAttTimeMix(usize),
    PreHead,
    PostHeadLayerNorm,
    PostHead,
}

pub struct InferJob {
    commands: Vec<CommandBuffer>,
    redirect: InferRedirect,

    embed_device: EmbedDevice,
    embed: TensorCpu<f16>,

    cursors: TensorGpu<u32, ReadWrite>,
    tokens: TensorGpu<u32, ReadWrite>,
    input: TensorGpu<f16, ReadWrite>,
    output: TensorGpu<f32, ReadWrite>,
}

impl Job for InferJob {
    type Input = InferInput;
    type Output = InferOutput;

    fn load(self, input: &InferChunk) -> Result<Self> {
        if input.num_token() == 0 {
            return Ok(self);
        }

        let stack: Vec<TensorCpu<f16>> = input
            .iter()
            .map(|chunk| {
                let num_emb = self.embed.shape()[0];
                let num_token = chunk.len();
                let data = self.embed.data();
                let data = chunk
                    .iter()
                    .map(|&token| {
                        let start = num_emb * token as usize;
                        let end = start + num_emb;
                        data[start..end].to_vec()
                    })
                    .concat();
                let data = data.into_iter().collect_vec();
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
                    .iter()
                    .map(|chunk| chunk.0.clone())
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
        let batches = batches.into_iter().map(InferOutputBatch).collect();
        Ok(InferOutput(batches))
    }
}

#[derive(Debug, Clone)]
pub struct Frame<F: Float> {
    pub state: State,
    pub buffer: Runtime<F>,
    pub header: Header<F>,
}

pub type HookFn<F> = Box<dyn Fn(Frame<F>) -> Result<TensorOp, TensorError> + Send + Sync>;
pub type HookMap<F> = HashMap<Hook, HookFn<F>>;

#[derive(Clone)]
pub struct Bundle<F: Float> {
    model: Model,
    state: State,
    hooks: Arc<HookMap<F>>,
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
            phantom: PhantomData,
        }
    }

    pub fn new_with_hooks(model: Model, num_batch: usize, hooks: HookMap<F>) -> Self {
        Self {
            hooks: Arc::new(hooks),
            ..Self::new(model, num_batch)
        }
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

    #[inline]
    fn model(&self) -> impl Serialize + 'static {
        self.model.clone()
    }
}

fn turbo(num_token: usize) -> bool {
    num_token % super::infer::MIN_TOKEN_CHUNK_SIZE == 0
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

impl<F: Float> Dispatcher<InferJob> for Bundle<F> {
    type Info = InferInfo;

    fn dispatch(&self, seed: Self::Info) -> Result<InferJob> {
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
        let frame = Frame {
            state: state.clone(),
            buffer: buffer.clone(),
            header: header.clone(),
        };

        context.maintain();

        if num_token == 0 {
            let embed_device = match &tensor.embed.u {
                Some(_) => EmbedDevice::Gpu,
                None => EmbedDevice::Cpu,
            };
            return Ok(InferJob {
                commands: vec![],
                redirect,
                embed_device,
                embed: model.tensor.embed.w.clone(),
                tokens: buffer.tokens,
                cursors: buffer.cursors,
                input: buffer.input,
                output: header.head_o,
            });
        }

        #[cfg(feature = "trace")]
        let _span = tracing::trace_span!("build").entered();

        let (head_ops, head_x) = if num_token == 1 || num_token == num_header {
            (vec![], buffer.x.clone())
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

                    let input = buffer.x.view(.., first..=last, .., ..)?;
                    let output = header.head_x.view(.., start..end, .., ..)?;
                    ops.push(TensorOp::blit(input, output)?);

                    start = end;
                }
                end += 1;
            }
            (ops, header.head_x.clone())
        };

        let hook_op = |hook: Hook| hook_op(&self.hooks, &hook, &frame);
        let mut ops = vec![];

        let embed_device = {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("embed").entered();

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
                    &tensor.embed.ln.w,
                    &tensor.embed.ln.b,
                    &buffer.input,
                    Model::LN_EPS,
                )?,
                TensorOp::blit(&buffer.input, &buffer.x)?,
                hook_op(Hook::PostEmbedLayerNorm)?,
            ]);
            embed_device
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

            if (index + 1) % (info.num_layer / super::infer::NUM_LAYER_CHUNK) == 0 {
                ops.push(TensorOp::Sep);
            }
        }

        {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("header").entered();

            let hooks = self.hooks.clone();
            let frame = frame.clone();
            let head = model.tensor.head.clone();

            let op = dispatch_header(hooks, frame, head, head_x, num_header, head_ops)?;
            ops.push(op);
        }

        let commands = {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("encode").entered();
            context.encode(&TensorOp::List(ops))
        };

        Ok(InferJob {
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

#[allow(clippy::too_many_arguments)]
fn dispatch_layer<F: Float>(
    hooks: Arc<HookMap<F>>,
    frame: Frame<F>,
    layer: Layer,
    index: usize,
    num_token: usize,
    head_size: usize,
    rescale: usize,
) -> Result<TensorOp> {
    let hook_op = |hook: Hook| hook_op(&hooks, &hook, &frame);
    let Frame { state, buffer, .. } = &frame;

    let att_kk = buffer.att_kk.reshape(
        TensorDimension::Dimension(head_size),
        TensorDimension::Auto,
        TensorDimension::Dimension(num_token),
        TensorDimension::Dimension(1),
    )?;
    let att_k = buffer.att_k.reshape(
        TensorDimension::Dimension(head_size),
        TensorDimension::Auto,
        TensorDimension::Dimension(num_token),
        TensorDimension::Dimension(1),
    )?;
    let att_v = buffer.att_v.reshape(
        TensorDimension::Dimension(head_size),
        TensorDimension::Auto,
        TensorDimension::Dimension(num_token),
        TensorDimension::Dimension(1),
    )?;

    let mut ops = vec![];

    ops.extend([
        TensorOp::blit(&buffer.x, &buffer.att_x)?,
        hook_op(Hook::PreAtt(index))?,
        TensorOp::layer_norm(
            &layer.att_ln.w,
            &layer.att_ln.b,
            &buffer.att_x,
            Model::LN_EPS,
        )?,
        hook_op(Hook::PostAttLayerNorm(index))?,
        hook_op(Hook::PreAttTokenShift(index))?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.att.x_r,
            state.att(index)?,
            &buffer.att_x,
            &buffer.att_rx,
            true,
        )?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.att.x_w,
            state.att(index)?,
            &buffer.att_x,
            &buffer.att_wx,
            true,
        )?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.att.x_k,
            state.att(index)?,
            &buffer.att_x,
            &buffer.att_kx,
            true,
        )?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.att.x_v,
            state.att(index)?,
            &buffer.att_x,
            &buffer.att_vx,
            true,
        )?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.att.x_a,
            state.att(index)?,
            &buffer.att_x,
            &buffer.att_ax,
            true,
        )?,
        TensorOp::token_shift(
            &buffer.cursors,
            &layer.att.x_g,
            state.att(index)?,
            &buffer.att_x,
            &buffer.att_gx,
            true,
        )?,
        hook_op(Hook::PostAttTokenShift(index))?,
        hook_op(Hook::PreAttLinear(index))?,
        layer.att.w_r.matmul_op(
            &buffer.att_rx,
            &buffer.att_r,
            Activation::None,
            turbo(num_token),
        )?,
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
        hook_op(Hook::PostAttLinear(index))?,
        hook_op(Hook::PreAttLora(index))?,
        layer.att.w1.matmul_op(
            &buffer.att_wx,
            &buffer.temp_w,
            Activation::Tanh,
            turbo(num_token),
        )?,
        layer.att.w2.matmul_op(
            &buffer.temp_w,
            &buffer.att_w,
            Activation::None,
            turbo(num_token),
        )?,
        layer.att.a1.matmul_op(
            &buffer.att_ax,
            &buffer.temp_a,
            Activation::None,
            turbo(num_token),
        )?,
        layer.att.a2.matmul_op(
            &buffer.temp_a,
            &buffer.att_a,
            Activation::None,
            turbo(num_token),
        )?,
        TensorOp::add_activate(
            &layer.att.a0,
            &buffer.att_a,
            BinaryActivation::out(Activation::Sigmoid),
        )?,
        layer.att.g1.matmul_op(
            &buffer.att_gx,
            &buffer.temp_g,
            Activation::Sigmoid,
            turbo(num_token),
        )?,
        layer.att.g2.matmul_op(
            &buffer.temp_g,
            &buffer.att_g,
            Activation::None,
            turbo(num_token),
        )?,
        hook_op(Hook::PostAttLora(index))?,
        hook_op(Hook::PreAttControl(index))?,
        TensorOp::blit(&buffer.att_k, &buffer.att_kk)?,
        TensorOp::mul(&layer.att.k_k, &buffer.att_kk)?,
        TensorOp::l2_norm(&att_kk, Model::L2_EPS)?,
        TensorOp::control_k_v7(&layer.att.k_a, &buffer.att_a, &buffer.att_k)?,
        hook_op(Hook::PostAttControl(index))?,
    ]);

    ops.push(hook_op(Hook::PreAttDeltaRule(index))?);
    match index {
        0 => ops.push(TensorOp::blit(&buffer.att_v, &buffer.att_v0)?),
        _ => ops.extend([
            layer.att.v1.matmul_op(
                &buffer.att_vx,
                &buffer.temp_v,
                Activation::None,
                turbo(num_token),
            )?,
            layer.att.v2.matmul_op(
                &buffer.temp_v,
                &buffer.att_vv,
                Activation::None,
                turbo(num_token),
            )?,
            TensorOp::add_activate(
                &layer.att.v0,
                &buffer.att_vv,
                BinaryActivation::out(Activation::Sigmoid),
            )?,
            TensorOp::lerp(&buffer.att_v0, &buffer.att_v, &buffer.att_vv, true)?,
        ]),
    };
    ops.push(hook_op(Hook::PostAttDeltaRule(index))?);

    ops.extend([
        hook_op(Hook::PreAttTimeDecayActivate(index))?,
        TensorOp::add(&layer.att.w0, &buffer.att_w)?,
        hook_op(Hook::PostAttTimeDecayActivate(index))?,
        hook_op(Hook::PreAttTimeMix(index))?,
        TensorOp::blit(&buffer.att_k, buffer.att_kv.view(.., .., 0, ..)?)?,
        TensorOp::blit(&buffer.att_v, buffer.att_kv.view(.., .., 1, ..)?)?,
        TensorOp::blit(&buffer.att_a, buffer.att_ab.view(.., .., 0, ..)?)?,
        TensorOp::blit(&buffer.att_kk, buffer.att_ab.view(.., .., 1, ..)?)?,
        hook_op(Hook::PostAttTimeMix(index))?,
    ]);

    if (index + 1) % rescale == 0 {
        ops.push(TensorOp::discount(&buffer.x, 0.5, 0.0)?);
    }

    Ok(TensorOp::List(ops))
}

fn dispatch_header<F: Float>(
    hooks: Arc<HookMap<F>>,
    frame: Frame<F>,
    head: Head,
    head_x: TensorGpu<F, ReadWrite>,
    num_header: usize,
    mut ops: Vec<TensorOp>,
) -> Result<TensorOp> {
    let hook_op = |hook: Hook| hook_op(&hooks, &hook, &frame);
    let header = &frame.header;

    if num_header > 0 {
        ops.extend([
            hook_op(Hook::PreHead)?,
            TensorOp::layer_norm(&head.ln.w, &head.ln.b, &head_x, Model::LN_EPS)?,
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
