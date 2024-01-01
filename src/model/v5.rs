use std::{convert::Infallible, sync::Arc};

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor};

use super::{
    matrix::Matrix,
    run::{HookMap, ModelRunInner, Output},
    softmax::{ModelSoftmaxInner, Softmax},
    FromBuilder, ModelBase, ModelBuilder, ModelError, ModelInfo, PreparedModelBuilder, Quant,
    StateBuilder,
};
use crate::{
    context::Context,
    model::RESCALE_LAYER,
    tensor::{
        cache::ResourceCache,
        ops::{TensorCommand, TensorOp, TensorOpHook, TensorPass},
        shape::{Shape, TensorDimension},
        DeepClone, IntoPackedCursors, ReadBack, ReadWrite, TensorCpu, TensorError, TensorGpu,
        TensorReshape, TensorShape, TensorView,
    },
};

#[derive(Debug)]
pub struct Model<'a> {
    context: Context,
    info: ModelInfo,

    /// Whether to half the activations every [`RESCALE_LAYER`] layers.
    rescale: bool,
    /// Whether to use fp16 GEMM for matmul computations.
    turbo: bool,
    /// To prevent the GPU device from lost, this limits the maximum batch-token it processes one time.
    token_chunk_size: usize,

    tensor: ModelTensor<'a>,
    runtime_cache: ResourceCache<usize, Runtime>,
    output_cache: ResourceCache<usize, Output>,
    softmax_cache: ResourceCache<usize, Softmax>,
}

#[derive(Debug)]
struct ModelTensor<'a> {
    embed: Embed<'a>,
    head: Head,
    layers: Vec<Layer>,
}

#[derive(Debug)]
struct LayerNorm {
    w: TensorGpu<f16, ReadWrite>,
    b: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug)]
struct Att {
    time_decay: TensorGpu<f32, ReadWrite>,
    time_first: TensorGpu<f32, ReadWrite>,

    time_mix_k: TensorGpu<f16, ReadWrite>,
    time_mix_v: TensorGpu<f16, ReadWrite>,
    time_mix_r: TensorGpu<f16, ReadWrite>,
    time_mix_g: TensorGpu<f16, ReadWrite>,

    w_k: Matrix,
    w_v: Matrix,
    w_r: Matrix,
    w_g: Matrix,
    w_o: Matrix,

    group_norm: LayerNorm,
}

#[derive(Debug)]
struct Ffn {
    time_mix_k: TensorGpu<f16, ReadWrite>,
    time_mix_r: TensorGpu<f16, ReadWrite>,

    w_k: Matrix,
    w_v: Matrix,
    w_r: Matrix,
}

#[derive(Debug)]
struct Layer {
    att_layer_norm: LayerNorm,
    ffn_layer_norm: LayerNorm,
    att: Att,
    ffn: Ffn,
}

#[derive(Debug)]
struct Embed<'a> {
    layer_norm: LayerNorm,
    w: TensorCpu<'a, f16>,
    u: Option<TensorGpu<f16, ReadWrite>>,
}

#[derive(Debug)]
struct Head {
    layer_norm: LayerNorm,
    w: Matrix,
}

/// Runtime buffers.
#[derive(Debug)]
pub struct Runtime {
    pub tokens: TensorGpu<u32, ReadWrite>,
    pub cursors: TensorGpu<u32, ReadWrite>,
    pub input: TensorGpu<f32, ReadWrite>,

    pub att_x: TensorGpu<f32, ReadWrite>,
    pub att_kx: TensorGpu<f32, ReadWrite>,
    pub att_vx: TensorGpu<f32, ReadWrite>,
    pub att_rx: TensorGpu<f32, ReadWrite>,
    pub att_gx: TensorGpu<f32, ReadWrite>,
    pub att_k: TensorGpu<f32, ReadWrite>,
    pub att_v: TensorGpu<f32, ReadWrite>,
    pub att_r: TensorGpu<f32, ReadWrite>,
    pub att_g: TensorGpu<f32, ReadWrite>,
    pub att_o: TensorGpu<f32, ReadWrite>,

    pub ffn_x: TensorGpu<f32, ReadWrite>,
    pub ffn_kx: TensorGpu<f32, ReadWrite>,
    pub ffn_rx: TensorGpu<f32, ReadWrite>,
    pub ffn_k: TensorGpu<f32, ReadWrite>,
    pub ffn_v: TensorGpu<f32, ReadWrite>,
    pub ffn_r: TensorGpu<f32, ReadWrite>,

    pub half_x: TensorGpu<f16, ReadWrite>,
    pub half_k: TensorGpu<f16, ReadWrite>,
}

impl Runtime {
    pub fn new(context: &Context, info: &ModelInfo, num_token: usize, max_token: usize) -> Self {
        let shape = Shape::new(info.num_emb, num_token, 1, 1);
        let tokens_shape = Shape::new(num_token, 1, 1, 1);
        let cursors_shape = Shape::new(max_token, 1, 1, 1);
        let hidden_shape = Shape::new(info.num_hidden, num_token, 1, 1);

        Self {
            tokens: context.tensor_init(tokens_shape),
            cursors: context.tensor_init(cursors_shape),
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
            half_x: context.tensor_init(shape),
            half_k: context.tensor_init(hidden_shape),
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
    PreFfnActivate(usize),
    PostFfnActivate(usize),
    PreFfnChannelMix(usize),
    PostFfnChannelMix(usize),
    PostFfn(usize),
    PreHead,
    PostHeadLayerNorm,
    PostHead,
}

impl TensorOpHook for Hook {}

#[derive(Debug, Clone)]
pub struct ModelState {
    context: Context,
    info: ModelInfo,
    max_batch: usize,
    chunk_size: usize,
    head_size: usize,
    state: Vec<TensorGpu<f32, ReadWrite>>,
}

impl ModelState {
    fn att(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let chunk = layer / self.chunk_size;
        let offset = layer % self.chunk_size;
        let head_size = self.info.num_emb / self.info.num_head;

        let start = offset * (head_size + 2);
        let end = start + head_size + 1;
        self.state[chunk].view(.., start..end, .., ..)
    }

    fn ffn(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let chunk = layer / self.chunk_size;
        let offset = layer % self.chunk_size;
        let head_size = self.info.num_emb / self.info.num_head;

        let start = offset * (head_size + 2) + head_size + 1;
        self.state[chunk].view(.., start..=start, .., ..)
    }
}

impl DeepClone for ModelState {
    fn deep_clone(&self) -> Self {
        let state = self
            .state
            .iter()
            .map(|tensor| tensor.deep_clone())
            .collect();
        Self {
            state,
            ..self.clone()
        }
    }
}

impl FromBuilder for ModelState {
    type Builder<'a> = StateBuilder;
    type Error = Infallible;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error> {
        let StateBuilder {
            context,
            info,
            max_batch,
            chunk_size,
        } = builder;
        let num_chunk = (info.num_layer + chunk_size - 1) / chunk_size;
        let head_size = info.num_emb / info.num_head;
        let state = (0..num_chunk)
            .map(|_| {
                let data = (0..max_batch)
                    .map(|_| vec![0.0; chunk_size * info.num_emb * (head_size + 2)])
                    .collect_vec()
                    .concat();
                context
                    .tensor_from_data(
                        Shape::new(info.num_emb, chunk_size * (head_size + 2), max_batch, 1),
                        data,
                    )
                    .expect("state creation")
            })
            .collect();
        Ok(Self {
            context,
            info,
            max_batch,
            chunk_size,
            head_size,
            state,
        })
    }
}

impl super::ModelState for ModelState {
    type BackedState = BackedState;

    #[inline]
    fn context(&self) -> &Context {
        &self.context
    }

    #[inline]
    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn load(&self, backed: &BackedState) -> Result<()> {
        use super::BackedState;
        if backed.max_batch() != self.max_batch() {
            return Err(ModelError::BatchSize(backed.max_batch(), self.max_batch()).into());
        }
        for (state, (shape, backed)) in self.state.iter().zip(backed.data.iter()) {
            let host = state.context.tensor_from_data(*shape, backed)?;
            state.load(&host)?;
        }
        Ok(())
    }

    fn load_batch(&self, backed: &BackedState, batch: usize) -> Result<()> {
        use super::BackedState;
        if backed.max_batch() != 1 {
            return Err(ModelError::BatchSize(backed.max_batch(), 1).into());
        }
        for (state, (_, backed)) in self.state.iter().zip(backed.data.iter()) {
            let shape = state.shape();
            let shape = Shape::new(shape[0], shape[1], 1, 1);
            let host = state.context.tensor_from_data(shape, backed)?;
            state.load_batch(&host, batch)?;
        }
        Ok(())
    }

    async fn back(&self) -> BackedState {
        let max_batch = self.max_batch;
        let chunk_size = self.chunk_size;
        let head_size = self.head_size;

        let mut data = Vec::with_capacity(self.state.len());
        for state in self.state.iter() {
            let shape = state.shape();
            let map = state.context.tensor_init(shape);

            let mut encoder = state
                .context
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            encoder.copy_tensor(state, &map).expect("back entire state");
            state.context.queue.submit(Some(encoder.finish()));

            let host = map.back_async().await;
            data.push((shape, host.to_vec()))
        }
        let data = data.into();

        BackedState {
            max_batch,
            chunk_size,
            head_size,
            data,
        }
    }

    async fn back_batch(&self, batch: usize) -> Result<BackedState> {
        let max_batch = self.max_batch;
        let chunk_size = self.chunk_size;
        let head_size = self.head_size;

        if batch >= max_batch {
            return Err(ModelError::BatchOutOfRange {
                batch,
                max: max_batch,
            }
            .into());
        }

        let mut data = Vec::with_capacity(self.state.len());
        for state in self.state.iter() {
            let shape = state.shape();
            let shape = Shape::new(shape[0], shape[1], 1, 1);
            let map = state.context.tensor_init(shape);

            let mut encoder = state
                .context
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            encoder.copy_tensor_batch(state, &map, batch)?;
            state.context.queue.submit(Some(encoder.finish()));

            let host = map.back_async().await;
            data.push((shape, host.to_vec()));
        }
        let data = data.into();

        Ok(BackedState {
            max_batch: 1,
            chunk_size,
            head_size,
            data,
        })
    }

    fn blit(&self, other: &ModelState) -> Result<(), TensorError> {
        for (state, other) in self.state.iter().zip(other.state.iter()) {
            state.check_shape(other.shape())?;
            let mut encoder = state
                .context
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            encoder.copy_tensor(state, other)?;
            state.context.queue.submit(Some(encoder.finish()));
        }
        Ok(())
    }

    fn blit_batch(
        &self,
        other: &ModelState,
        from_batch: usize,
        to_batch: usize,
    ) -> Result<(), TensorError> {
        for (state, other) in self.state.iter().zip(other.state.iter()) {
            let op = TensorOp::blit(
                state.view(.., .., from_batch, ..)?,
                other.view(.., .., to_batch, ..)?,
            )?;
            let mut encoder = state
                .context
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.execute_tensor_op(&op);
            drop(pass);

            state.context.queue.submit(Some(encoder.finish()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BackedState {
    pub max_batch: usize,
    pub chunk_size: usize,
    pub head_size: usize,
    pub data: Arc<Vec<(Shape, Vec<f32>)>>,
}

impl FromBuilder for BackedState {
    type Builder<'a> = StateBuilder;
    type Error = Infallible;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error> {
        let StateBuilder {
            info,
            max_batch,
            chunk_size,
            ..
        } = builder;
        let head_size = info.num_emb / info.num_head;
        let shape = Shape::new(info.num_emb, chunk_size * (head_size + 2), max_batch, 1);
        let data = (0..info.num_layer)
            .map(|_| {
                (0..max_batch)
                    .map(|_| vec![0.0; chunk_size * info.num_emb * (head_size + 2)])
                    .collect_vec()
                    .concat()
            })
            .map(|x| (shape, x))
            .collect_vec()
            .into();
        Ok(Self {
            max_batch,
            chunk_size,
            head_size,
            data,
        })
    }
}

impl super::BackedState for BackedState {
    #[inline]
    fn max_batch(&self) -> usize {
        self.max_batch
    }

    #[inline]
    fn num_layer(&self) -> usize {
        self.chunk_size * self.data.len()
    }

    fn embed(&self, batch: usize, layer: usize) -> Vec<f32> {
        let index = layer / self.chunk_size;
        let offset = layer % self.chunk_size;

        let chunk = &self.data[index];
        let num_emb = chunk.0[0];

        let start = ((batch * self.chunk_size + offset) * (self.head_size + 2) + 1) * num_emb;
        let end = start + num_emb;

        chunk.1[start..end].to_vec()
    }
}

impl<'a> Model<'a> {
    #[inline]
    fn request_runtime(&self, num_token: usize) -> Arc<Runtime> {
        self.runtime_cache.request(num_token, || {
            Runtime::new(&self.context, &self.info, num_token, self.token_chunk_size)
        })
    }
}

impl<'a> FromBuilder for Model<'a> {
    type Builder<'b> = ModelBuilder<'b>;
    type Error = anyhow::Error;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error> {
        let PreparedModelBuilder {
            context,
            info,
            loader,
            quant,
            embed_device,
            turbo,
            rescale,
            token_chunk_size,
        } = builder.prepare()?;

        let embed = Embed {
            layer_norm: LayerNorm {
                w: loader.load_vector_f16("blocks.0.ln0.weight")?,
                b: loader.load_vector_f16("blocks.0.ln0.bias")?,
            },
            w: loader.load_embed()?,
            u: match embed_device {
                super::EmbedDevice::Cpu => None,
                super::EmbedDevice::Gpu => Some(loader.load_matrix_f16("emb.weight")?),
            },
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

        let matrix_f16_cache = ResourceCache::<Shape, TensorGpu<f16, ReadWrite>>::new(0);
        let load_matrix = |name: String, quant: Quant| -> Result<Matrix> {
            match quant {
                Quant::None => Ok(Matrix::Fp16(loader.load_matrix_f16(name)?)),
                Quant::Int8 => {
                    let shape = loader.tensor_shape(&name)?;
                    let buffer = matrix_f16_cache.request(shape, || context.tensor_init(shape));
                    loader.load_in_place_matrix_f16(&buffer, &name)?;
                    Ok(Matrix::quant_u8(&buffer)?)
                }
                Quant::NF4 => {
                    let shape = loader.tensor_shape(&name)?;
                    let buffer = matrix_f16_cache.request(shape, || context.tensor_init(shape));
                    loader.load_in_place_matrix_f16(&buffer, &name)?;
                    Ok(Matrix::quant_nf4(&buffer)?)
                }
            }
        };
        let load_matrix_discount = |name: String, quant: Quant, discount: f32| -> Result<Matrix> {
            match quant {
                Quant::None => Ok(Matrix::Fp16(
                    loader.load_matrix_f16_discount(name, discount)?,
                )),
                Quant::Int8 => {
                    let shape = loader.tensor_shape(&name)?;
                    let buffer = matrix_f16_cache.request(shape, || context.tensor_init(shape));
                    loader.load_in_place_matrix_f16_discount(&buffer, &name, discount)?;
                    Ok(Matrix::quant_u8(&buffer)?)
                }
                Quant::NF4 => {
                    let shape = loader.tensor_shape(&name)?;
                    let buffer = matrix_f16_cache.request(shape, || context.tensor_init(shape));
                    loader.load_in_place_matrix_f16_discount(&buffer, &name, discount)?;
                    Ok(Matrix::quant_nf4(&buffer)?)
                }
            }
        };

        let layers = (0..info.num_layer)
            .map(|layer| {
                let quant = quant.get(&layer).copied().unwrap_or_default();
                let discount = match rescale {
                    true => 2.0_f32.powi(-((layer / RESCALE_LAYER) as i32)),
                    false => 1.0,
                };
                if matches!(quant, Quant::None) {
                    matrix_f16_cache.clear();
                }

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
                            TensorDimension::Dimension(info.num_head),
                            TensorDimension::Dimension(1),
                            TensorDimension::Dimension(1),
                        )?,
                    b: loader
                        .load_vector_f16(format!("{att}.ln_x.bias"))?
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

                Ok(Layer {
                    att_layer_norm,
                    ffn_layer_norm,
                    att,
                    ffn,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        context.queue.submit(None);
        context.device.poll(wgpu::MaintainBase::Wait);

        let tensor = ModelTensor {
            embed,
            head,
            layers,
        };
        Ok(Self {
            context,
            info,
            rescale,
            turbo,
            token_chunk_size,
            tensor,
            runtime_cache: ResourceCache::new(1),
            output_cache: ResourceCache::new(1),
            softmax_cache: ResourceCache::new(1),
        })
    }
}

impl ModelBase for Model<'_> {
    type ModelState = ModelState;

    #[inline]
    fn context(&self) -> &Context {
        &self.context
    }

    #[inline]
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    #[inline]
    fn token_chunk_size(&self) -> usize {
        self.token_chunk_size
    }

    #[inline]
    fn head_shape(&self, num_batch: usize) -> Shape {
        Shape::new(self.info.num_vocab, 1, num_batch, 1)
    }
}

impl ModelSoftmaxInner for Model<'_> {
    #[inline]
    fn request_softmax(&self, num_batch: usize) -> Arc<Softmax> {
        self.softmax_cache.request(num_batch, || {
            Softmax::new(&self.context, &self.info, num_batch)
        })
    }
}

impl ModelRunInner for Model<'_> {
    type Hook = Hook;
    type Runtime = Runtime;

    #[inline]
    fn request_output(&self, num_batch: usize) -> Arc<Output> {
        self.output_cache.request(num_batch, || {
            Output::new(&self.context, &self.info, num_batch)
        })
    }

    fn run_internal(
        &self,
        tokens: Vec<Vec<u16>>,
        state: &ModelState,
        should_output: Vec<bool>,
        hooks: &HookMap<Hook, ModelState, Runtime>,
    ) -> Result<(TensorGpu<f32, ReadBack>, Vec<Option<usize>>)> {
        let context = &self.context;
        let tensor = &self.tensor;

        let input = self.create_input(&tensor.embed.w, &tokens)?;
        let num_batch = input.num_batch();
        let num_token = input.num_token();
        let head_size = self.info.num_emb / self.info.num_head;
        assert_ne!(num_token, 0);

        let turbo = self.turbo && num_token == self.token_chunk_size;

        // collect batch output copy commands for later
        let mut redirect = vec![None; num_batch];
        let headers = input
            .cursors
            .iter()
            .filter(|cursor| cursor.len > 0)
            .filter(|cursor| should_output[cursor.batch])
            .enumerate()
            .map(|(index, cursor)| {
                redirect[cursor.batch] = Some(index);
                cursor.token + cursor.len - 1
            })
            .collect_vec();
        let num_header = headers.len();

        let buffer = self.request_runtime(num_token);
        let output = self.request_output(num_header.max(1));

        let hook_op = |hook: Hook| -> TensorOp {
            hooks
                .get(&hook)
                .map(|f| f(state, &buffer))
                .unwrap_or(TensorOp::List(vec![]))
        };

        // gather and group copy operations
        let (head_ops, head_x) = if num_token == 1 || num_token == num_header {
            (TensorOp::List(vec![]), &buffer.ffn_x)
        } else {
            let mut start = 0;
            let mut end = 1;
            let mut ops = vec![];
            while end <= headers.len() {
                if end == headers.len() || headers[end - 1] + 1 != headers[end] {
                    let first = headers[start];
                    let last = headers[end - 1];
                    assert_eq!(last - first + 1, end - start);

                    let input = buffer.ffn_x.view(.., first..=last, .., ..)?;
                    let output = output.head_x.view(.., start..end, .., ..)?;
                    ops.push(TensorOp::blit(input, output)?);

                    start = end;
                }
                end += 1;
            }
            (TensorOp::List(ops), &output.head_x)
        };

        // let head_ops: Vec<_> = input
        //     .cursors
        //     .iter()
        //     .filter(|cursor| cursor.len > 0)
        //     .filter(|cursor| !last.is_some_and(|index| cursor.batch == index))
        //     .enumerate()
        //     .map(|(index, cursor)| -> Result<TensorOp<'_>, TensorError> {
        //         redirect[cursor.batch] = Some(index);
        //         let token = cursor.token + cursor.len - 1;
        //         let input = buffer.ffn_x.as_view((.., token, ..))?;
        //         let output = buffer.head_x.as_view((.., .., index))?;
        //         TensorOp::blit(input, output)
        //     })
        //     .try_collect()?;

        let mut ops = vec![];

        let mut cursors = input.cursors.into_cursors();
        cursors.resize(self.token_chunk_size, 0);

        let cursors = context.tensor_from_data(buffer.cursors.shape(), cursors)?;
        buffer.cursors.load(&cursors)?;

        match &tensor.embed.u {
            Some(u) => {
                let tokens = tokens
                    .concat()
                    .into_iter()
                    .map(|token| token as u32)
                    .collect_vec();

                let tokens = context.tensor_from_data(buffer.tokens.shape(), tokens)?;
                buffer.tokens.load(&tokens)?;

                ops.push(TensorOp::embed(&buffer.tokens, u, &buffer.input)?);
            }
            None => buffer.input.load(&input.tensor)?,
        }
        ops.append(&mut vec![
            hook_op(Hook::PostEmbedLoaded),
            TensorOp::layer_norm(
                &tensor.embed.layer_norm.w,
                &tensor.embed.layer_norm.b,
                &buffer.input,
            )?,
            hook_op(Hook::PostEmbedLayerNorm),
        ]);

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let ops = TensorOp::List(ops);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&ops);
        drop(pass);

        for (index, layer) in tensor.layers.iter().enumerate() {
            use TensorDimension::{Auto, Dimension};
            let time_first = layer.att.time_first.reshape(
                Dimension(head_size),
                Auto,
                Dimension(1),
                Dimension(1),
            )?;
            let time_decay = layer.att.time_decay.reshape(
                Dimension(head_size),
                Auto,
                Dimension(1),
                Dimension(1),
            )?;
            let att_x = buffer.att_x.reshape(
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
                hook_op(Hook::PreAtt(index)),
                TensorOp::layer_norm(
                    &layer.att_layer_norm.w,
                    &layer.att_layer_norm.b,
                    &buffer.att_x,
                )?,
                hook_op(Hook::PostAttLayerNorm(index)),
                hook_op(Hook::PreAttTokenShift(index)),
                TensorOp::token_shift_fp16(
                    &buffer.cursors,
                    layer.att.time_mix_k.view(.., .., .., ..)?,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_kx,
                    false,
                )?,
                TensorOp::token_shift_fp16(
                    &buffer.cursors,
                    layer.att.time_mix_v.view(.., .., .., ..)?,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_vx,
                    false,
                )?,
                TensorOp::token_shift_fp16(
                    &buffer.cursors,
                    layer.att.time_mix_r.view(.., .., .., ..)?,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_rx,
                    false,
                )?,
                TensorOp::token_shift_fp16(
                    &buffer.cursors,
                    layer.att.time_mix_g.view(.., .., .., ..)?,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_gx,
                    false,
                )?,
                hook_op(Hook::PostAttTokenShift(index)),
                hook_op(Hook::PreAttLinear(index)),
                layer.att.w_k.matmul_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    buffer.att_kx.view(.., .., .., ..)?,
                    buffer.att_k.view(.., .., .., ..)?,
                    turbo,
                )?,
                layer.att.w_v.matmul_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    buffer.att_vx.view(.., .., .., ..)?,
                    buffer.att_v.view(.., .., .., ..)?,
                    turbo,
                )?,
                layer.att.w_r.matmul_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    buffer.att_rx.view(.., .., .., ..)?,
                    buffer.att_r.view(.., .., .., ..)?,
                    turbo,
                )?,
                layer.att.w_g.matmul_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    buffer.att_gx.view(.., .., .., ..)?,
                    buffer.att_g.view(.., .., .., ..)?,
                    turbo,
                )?,
                hook_op(Hook::PostAttLinear(index)),
                hook_op(Hook::PreAttTimeMix(index)),
                TensorOp::time_mix_v5(
                    &buffer.cursors,
                    &time_decay,
                    &time_first,
                    &att_k,
                    &att_v,
                    &att_r,
                    &att_x,
                    state.att(index)?,
                )?,
                TensorOp::group_norm(&layer.att.group_norm.w, &layer.att.group_norm.b, &att_x)?,
                hook_op(Hook::PostAttTimeMix(index)),
                hook_op(Hook::PreAttGate(index)),
                TensorOp::silu(&buffer.att_g, &buffer.att_x)?,
                hook_op(Hook::PostAttGate(index)),
                hook_op(Hook::PreAttOut(index)),
                layer.att.w_o.matmul_vec_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    buffer.att_x.view(.., .., .., ..)?,
                    buffer.att_o.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostAttOut(index)),
                TensorOp::add_fp32(
                    buffer.input.view(.., .., .., ..)?,
                    buffer.att_o.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostAtt(index)),
            ]);

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.execute_tensor_op(&ops);
            drop(pass);

            encoder.copy_tensor(&buffer.att_o, &buffer.ffn_x)?;

            let ops = TensorOp::List(vec![
                hook_op(Hook::PreFfn(index)),
                TensorOp::layer_norm(
                    &layer.ffn_layer_norm.w,
                    &layer.ffn_layer_norm.b,
                    &buffer.ffn_x,
                )?,
                hook_op(Hook::PostFfnLayerNorm(index)),
                hook_op(Hook::PreFfnTokenShift(index)),
                TensorOp::token_shift_fp16(
                    &buffer.cursors,
                    layer.ffn.time_mix_k.view(.., .., .., ..)?,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                    &buffer.ffn_kx,
                    false,
                )?,
                TensorOp::token_shift_fp16(
                    &buffer.cursors,
                    layer.ffn.time_mix_r.view(.., .., .., ..)?,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                    &buffer.ffn_rx,
                    false,
                )?,
                hook_op(Hook::PostFfnTokenShift(index)),
                hook_op(Hook::PreFfnLinear(index)),
                layer.ffn.w_k.matmul_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    buffer.ffn_kx.view(.., .., .., ..)?,
                    buffer.ffn_k.view(.., .., .., ..)?,
                    turbo,
                )?,
                hook_op(Hook::PreFfnActivate(index)),
                TensorOp::squared_relu(&buffer.ffn_k)?,
                hook_op(Hook::PostFfnActivate(index)),
                layer.ffn.w_v.matmul_op(
                    buffer.half_k.view(.., .., .., ..)?,
                    buffer.ffn_k.view(.., .., .., ..)?,
                    buffer.ffn_v.view(.., .., .., ..)?,
                    turbo,
                )?,
                layer.ffn.w_r.matmul_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    buffer.ffn_rx.view(.., .., .., ..)?,
                    buffer.ffn_r.view(.., .., .., ..)?,
                    turbo,
                )?,
                hook_op(Hook::PostFfnLinear(index)),
                hook_op(Hook::PreFfnChannelMix(index)),
                TensorOp::channel_mix(
                    &buffer.cursors,
                    &buffer.ffn_r,
                    &buffer.ffn_v,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                )?,
                hook_op(Hook::PostFfnChannelMix(index)),
                TensorOp::add_fp32(
                    buffer.att_o.view(.., .., .., ..)?,
                    buffer.ffn_x.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostFfn(index)),
            ]);

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.execute_tensor_op(&ops);
            drop(pass);

            if self.rescale && (index + 1) % RESCALE_LAYER == 0 {
                let op = TensorOp::half(&buffer.ffn_x)?;
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.execute_tensor_op(&op);
                drop(pass);
            }

            if index != self.info.num_layer - 1 {
                encoder.copy_tensor(&buffer.ffn_x, &buffer.input)?;
            }
        }

        if num_header > 0 {
            let ops = TensorOp::List(vec![
                hook_op(Hook::PreHead),
                TensorOp::layer_norm(&tensor.head.layer_norm.w, &tensor.head.layer_norm.b, head_x)?,
                hook_op(Hook::PostHeadLayerNorm),
                tensor.head.w.matmul_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    head_x.view(.., .., .., ..)?,
                    output.head_o.view(.., .., .., ..)?,
                    turbo,
                )?,
                hook_op(Hook::PostHead),
            ]);

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.execute_tensor_op(&head_ops);
            pass.execute_tensor_op(&ops);
            drop(pass);

            encoder.copy_tensor(&output.head_o, &output.map)?;
        }

        context.queue.submit(Some(encoder.finish()));
        Ok((output.map.clone(), redirect))
    }
}
