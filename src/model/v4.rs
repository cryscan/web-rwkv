use std::{convert::Infallible, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use half::f16;
use itertools::Itertools;
use web_rwkv_derive::{Deref, DerefMut};
use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor};

use super::{
    loader::Loader,
    matrix::Matrix,
    run::{ModelRun, Output},
    softmax::{ModelSoftmax, Softmax},
    FromBuilder, ModelBase, ModelBuilder, ModelError, ModelInfo, Quant, StateBuilder,
};
use crate::{
    context::Context,
    model::RESCALE_LAYER,
    tensor::{
        cache::ResourceCache,
        ops::{TensorCommand, TensorOp, TensorPass},
        shape::{Shape, TensorDimension},
        DeepClone, IntoPackedCursors, ReadWrite, TensorCpu, TensorError, TensorGpu, TensorReshape,
        TensorShape, TensorStack, TensorView,
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
    /// The head matrix is too big for a storage buffer so it's divided into chunks.
    head_chunk_size: usize,
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

    w_k: Matrix,
    w_v: Matrix,
    w_r: Matrix,
    w_o: Matrix,
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
}

#[derive(Debug)]
struct Head {
    layer_norm: LayerNorm,
    w: Vec<TensorGpu<f16, ReadWrite>>,
}

/// Runtime buffers.
#[derive(Debug)]
struct Runtime {
    cursors: TensorGpu<u32, ReadWrite>,
    input: TensorGpu<f32, ReadWrite>,

    att_x: TensorGpu<f32, ReadWrite>,
    att_kx: TensorGpu<f32, ReadWrite>,
    att_vx: TensorGpu<f32, ReadWrite>,
    att_rx: TensorGpu<f32, ReadWrite>,
    att_k: TensorGpu<f32, ReadWrite>,
    att_v: TensorGpu<f32, ReadWrite>,
    att_r: TensorGpu<f32, ReadWrite>,
    att_o: TensorGpu<f32, ReadWrite>,

    ffn_x: TensorGpu<f32, ReadWrite>,
    ffn_kx: TensorGpu<f32, ReadWrite>,
    ffn_rx: TensorGpu<f32, ReadWrite>,
    ffn_k: TensorGpu<f32, ReadWrite>,
    ffn_v: TensorGpu<f32, ReadWrite>,
    ffn_r: TensorGpu<f32, ReadWrite>,

    half_x: TensorGpu<f16, ReadWrite>,
    half_k: TensorGpu<f16, ReadWrite>,
}

impl Runtime {
    pub fn new(context: &Context, info: &ModelInfo, num_token: usize, max_token: usize) -> Self {
        let shape = Shape::new(info.num_emb, num_token, 1, 1);
        let cursors_shape = Shape::new(max_token, 1, 1, 1);
        let hidden_shape = Shape::new(info.num_hidden, num_token, 1, 1);

        Self {
            cursors: context.tensor_init(cursors_shape),
            input: context.tensor_init(shape),
            att_x: context.tensor_init(shape),
            att_kx: context.tensor_init(shape),
            att_vx: context.tensor_init(shape),
            att_rx: context.tensor_init(shape),
            att_k: context.tensor_init(shape),
            att_v: context.tensor_init(shape),
            att_r: context.tensor_init(shape),
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

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct ModelState(TensorGpu<f32, ReadWrite>);

impl ModelState {
    fn att(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let start = 5 * layer;
        let end = start + 4;
        self.view(.., start..end, .., ..)
    }

    fn ffn(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let start = 5 * layer + 4;
        self.view(.., start..=start, .., ..)
    }
}

impl DeepClone for ModelState {
    fn deep_clone(&self) -> Self {
        Self(self.0.deep_clone())
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
            ..
        } = builder;
        let data = (0..max_batch)
            .map(|_| {
                (0..info.num_layer)
                    .map(|_| {
                        [
                            vec![0.0; info.num_emb],
                            vec![0.0; info.num_emb],
                            vec![0.0; info.num_emb],
                            vec![f32::MIN; info.num_emb],
                            vec![0.0; info.num_emb],
                        ]
                        .concat()
                    })
                    .collect_vec()
                    .concat()
            })
            .collect_vec()
            .concat();
        let state = context
            .tensor_from_data(
                Shape::new(info.num_emb, 5 * info.num_layer, max_batch, 1),
                data,
            )
            .unwrap();
        Ok(Self(state))
    }
}

#[async_trait]
impl super::ModelState for ModelState {
    type BackedState = BackedState;

    #[inline]
    fn context(&self) -> &Context {
        &self.context
    }

    #[inline]
    fn max_batch(&self) -> usize {
        self.0.shape()[2]
    }

    fn load(&self, backed: &Self::BackedState) -> Result<()> {
        use super::BackedState;
        if backed.max_batch() != self.max_batch() {
            return Err(ModelError::BatchSize(backed.max_batch(), self.max_batch()).into());
        }
        let host = self.context.tensor_from_data(self.shape(), &*backed.data)?;
        self.0.load(&host).map_err(|err| err.into())
    }

    fn load_batch(&self, backed: &Self::BackedState, batch: usize) -> Result<()> {
        use super::BackedState;
        if backed.max_batch() != 1 {
            return Err(ModelError::BatchSize(backed.max_batch(), 1).into());
        }
        let shape = self.shape();
        let shape = Shape::new(shape[0], shape[1], 1, 1);
        let host = self.context.tensor_from_data(shape, &*backed.data)?;
        self.0.load_batch(&host, batch).map_err(|err| err.into())
    }

    async fn back(&self) -> Self::BackedState {
        let shape = self.shape();
        let map = self.context.tensor_init(shape);

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_tensor(self, &map).expect("back entire state");
        self.context.queue.submit(Some(encoder.finish()));

        let data = map.back_async().await.to_vec().into();
        BackedState { shape, data }
    }

    async fn back_batch(&self, batch: usize) -> Result<Self::BackedState> {
        if batch >= self.max_batch() {
            return Err(ModelError::BatchOutOfRange {
                batch,
                max: self.max_batch(),
            }
            .into());
        }

        let shape = self.shape();
        let shape = Shape::new(shape[0], shape[1], 1, 1);
        let map = self.context.tensor_init(shape);

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_tensor_batch(self, &map, batch)?;
        self.context.queue.submit(Some(encoder.finish()));

        let data = map.back_async().await.to_vec().into();
        Ok(BackedState { shape, data })
    }

    fn blit(&self, other: &Self) -> Result<(), TensorError> {
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_tensor(self, other)?;
        self.context.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn blit_batch(
        &self,
        other: &Self,
        from_batch: usize,
        to_batch: usize,
    ) -> Result<(), TensorError> {
        let op = TensorOp::blit(
            self.view(.., .., from_batch, ..)?,
            other.view(.., .., to_batch, ..)?,
        )?;
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        self.context.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BackedState {
    pub shape: Shape,
    pub data: Arc<Vec<f32>>,
}

impl FromBuilder for BackedState {
    type Builder<'a> = StateBuilder;
    type Error = Infallible;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error> {
        let StateBuilder {
            info, max_batch, ..
        } = builder;
        let shape = Shape::new(info.num_emb, 5 * info.num_layer, max_batch, 1);
        let data = (0..max_batch)
            .map(|_| {
                (0..info.num_layer)
                    .map(|_| {
                        [
                            vec![0.0; info.num_emb],
                            vec![0.0; info.num_emb],
                            vec![0.0; info.num_emb],
                            vec![f32::MIN; info.num_emb],
                            vec![0.0; info.num_emb],
                        ]
                        .concat()
                    })
                    .collect_vec()
                    .concat()
            })
            .collect_vec()
            .concat()
            .into();
        Ok(Self { shape, data })
    }
}

impl super::BackedState for BackedState {
    #[inline]
    fn max_batch(&self) -> usize {
        self.shape[2]
    }

    #[inline]
    fn num_layer(&self) -> usize {
        self.shape[1]
    }

    fn embed(&self, batch: usize, layer: usize) -> Vec<f32> {
        let num_emb = self.shape[0];
        let num_layer = self.shape[1];

        let start = ((batch * num_layer + layer) * 5 + 4) * num_emb;
        let end = start + num_emb;

        self.data[start..end].to_vec()
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
        let ModelBuilder {
            context,
            data,
            lora,
            quant,
            turbo,
            head_chunk_size,
            token_chunk_size,
        } = builder;

        if !head_chunk_size.is_power_of_two() {
            return Err(ModelError::InvalidChunkSize(head_chunk_size).into());
        }
        if !token_chunk_size.is_power_of_two() {
            return Err(ModelError::InvalidChunkSize(token_chunk_size).into());
        }

        let loader = Loader::new(&context, data, lora)?;
        let info = Loader::info(data)?;

        let rescale = turbo || quant.iter().any(|(_, quant)| matches!(quant, Quant::NF4));

        let embed = Embed {
            layer_norm: LayerNorm {
                w: loader.load_vector_f16("blocks.0.ln0.weight")?,
                b: loader.load_vector_f16("blocks.0.ln0.bias")?,
            },
            w: loader.load_embed()?,
        };

        let head = Head {
            layer_norm: LayerNorm {
                w: loader.load_vector_f16("ln_out.weight")?,
                b: loader.load_vector_f16("ln_out.bias")?,
            },
            w: loader.load_head(head_chunk_size)?,
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
                let time_decay = loader.load_vector_exp_f32(format!("{att}.time_decay"))?;
                let time_first = loader.load_vector_f32(format!("{att}.time_first"))?;
                let time_mix_k = loader.load_vector_f16(format!("{att}.time_mix_k"))?;
                let time_mix_v = loader.load_vector_f16(format!("{att}.time_mix_v"))?;
                let time_mix_r = loader.load_vector_f16(format!("{att}.time_mix_r"))?;

                let att = Att {
                    time_decay,
                    time_first,
                    time_mix_k,
                    time_mix_v,
                    time_mix_r,
                    w_k: load_matrix(format!("{att}.key.weight"), quant)?,
                    w_v: load_matrix(format!("{att}.value.weight"), quant)?,
                    w_r: load_matrix(format!("{att}.receptance.weight"), quant)?,
                    w_o: load_matrix_discount(format!("{att}.output.weight"), quant, discount)?,
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
            head_chunk_size,
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
    fn head_chunk_size(&self) -> usize {
        self.head_chunk_size
    }

    #[inline]
    fn head_shape(&self, num_batch: usize) -> Shape {
        Shape::new(self.info.num_vocab, 1, num_batch, 1)
    }
}

#[async_trait]
impl ModelSoftmax for Model<'_> {
    #[inline]
    fn request_softmax(&self, num_batch: usize) -> Arc<Softmax> {
        self.softmax_cache.request(num_batch, || {
            Softmax::new(&self.context, &self.info, num_batch)
        })
    }
}

#[async_trait]
impl ModelRun for Model<'_> {
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
        compute_head: Vec<bool>,
    ) -> Result<(Arc<Output>, Vec<Option<usize>>)> {
        let context = &self.context;
        let tensor = &self.tensor;

        let input: Vec<_> = tokens
            .into_iter()
            .map(|tokens| -> Result<_, TensorError> {
                let stack = TensorCpu::stack(
                    tokens
                        .into_iter()
                        .map(|token| tensor.embed.w.slice(.., token as usize, .., ..))
                        .try_collect()?,
                )
                .unwrap_or_else(|_| context.zeros(Shape::new(self.info.num_emb, 1, 0, 1)));
                stack.map(|x| x.to_f32()).reshape(
                    TensorDimension::Full,
                    TensorDimension::Auto,
                    TensorDimension::Dimension(1),
                    TensorDimension::Full,
                )
            })
            .try_collect()?;

        let input = TensorStack::try_from(input)?;
        let num_batch = input.num_batch();
        let num_token = input.num_token();
        assert_ne!(num_token, 0);

        let turbo = self.turbo && num_token == self.token_chunk_size;

        // collect batch output copy commands for later
        let mut redirect = vec![None; num_batch];
        let headers = input
            .cursors
            .iter()
            .filter(|cursor| cursor.len > 0)
            .filter(|cursor| compute_head[cursor.batch])
            .enumerate()
            .map(|(index, cursor)| {
                redirect[cursor.batch] = Some(index);
                cursor.token + cursor.len - 1
            })
            .collect_vec();
        let num_header = headers.len();

        let buffer = self.request_runtime(num_token);
        let output = self.request_output(num_header.max(1));

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

        let mut cursors = input.cursors.into_cursors();
        cursors.resize(self.token_chunk_size, 0);
        let cursors = context.tensor_from_data(buffer.cursors.shape(), cursors)?;

        buffer.input.load(&input.tensor)?;
        buffer.cursors.load(&cursors)?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let op = TensorOp::layer_norm(
            &tensor.embed.layer_norm.w,
            &tensor.embed.layer_norm.b,
            &buffer.input,
        )?;
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        for (index, layer) in tensor.layers.iter().enumerate() {
            encoder.copy_tensor(&buffer.input, &buffer.att_x)?;

            let ops = TensorOp::List(vec![
                TensorOp::layer_norm(
                    &layer.att_layer_norm.w,
                    &layer.att_layer_norm.b,
                    &buffer.att_x,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    layer.att.time_mix_k.view(.., .., .., ..)?,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_kx,
                    false,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    layer.att.time_mix_v.view(.., .., .., ..)?,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_vx,
                    false,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    layer.att.time_mix_r.view(.., .., .., ..)?,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_rx,
                    false,
                )?,
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
                TensorOp::time_mix(
                    &buffer.cursors,
                    &layer.att.time_decay,
                    &layer.att.time_first,
                    &buffer.att_k,
                    &buffer.att_v,
                    &buffer.att_r,
                    &buffer.att_x,
                    state.att(index)?,
                )?,
                layer.att.w_o.matmul_vec_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    buffer.att_x.view(.., .., .., ..)?,
                    buffer.att_o.view(.., .., .., ..)?,
                )?,
                TensorOp::add(
                    buffer.input.view(.., .., .., ..)?,
                    buffer.att_o.view(.., .., .., ..)?,
                )?,
            ]);

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.execute_tensor_op(&ops);
            drop(pass);

            encoder.copy_tensor(&buffer.att_o, &buffer.ffn_x)?;

            let ops = TensorOp::List(vec![
                TensorOp::layer_norm(
                    &layer.ffn_layer_norm.w,
                    &layer.ffn_layer_norm.b,
                    &buffer.ffn_x,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    layer.ffn.time_mix_k.view(.., .., .., ..)?,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                    &buffer.ffn_kx,
                    false,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    layer.ffn.time_mix_r.view(.., .., .., ..)?,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                    &buffer.ffn_rx,
                    false,
                )?,
                layer.ffn.w_k.matmul_op(
                    buffer.half_x.view(.., .., .., ..)?,
                    buffer.ffn_kx.view(.., .., .., ..)?,
                    buffer.ffn_k.view(.., .., .., ..)?,
                    turbo,
                )?,
                TensorOp::squared_relu(&buffer.ffn_k)?,
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
                TensorOp::channel_mix(
                    &buffer.cursors,
                    &buffer.ffn_r,
                    &buffer.ffn_v,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                )?,
                TensorOp::add(
                    buffer.att_o.view(.., .., .., ..)?,
                    buffer.ffn_x.view(.., .., .., ..)?,
                )?,
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
            let mut ops = vec![TensorOp::layer_norm(
                &tensor.head.layer_norm.w,
                &tensor.head.layer_norm.b,
                head_x,
            )?];

            for (chunk, matrix) in tensor.head.w.iter().enumerate() {
                let start = chunk * self.head_chunk_size;
                let end = start + matrix.shape()[1];
                let input = head_x.view(.., .., .., ..)?;
                let output = output.head_o.view(start..end, .., .., ..)?;
                ops.push(TensorOp::matmul_vec_fp16(matrix, input, output)?);
            }

            let ops = TensorOp::List(ops);

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.execute_tensor_op(&head_ops);
            pass.execute_tensor_op(&ops);
            drop(pass);

            encoder.copy_tensor(&output.head_o, &output.map)?;
        }

        context.queue.submit(Some(encoder.finish()));
        Ok((output.clone(), redirect))
    }
}
