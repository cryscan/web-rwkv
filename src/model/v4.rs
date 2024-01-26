use std::{convert::Infallible, sync::Arc};

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::{Deref, DerefMut};

use super::{
    run::{HookMap, ModelRunInternal, Output},
    softmax::{ModelSoftmaxInternal, Softmax},
    FromBuilder, ModelBase, ModelBuilder, ModelError, ModelInfo, PreparedModelBuilder, Quant,
    StateBuilder,
};
use crate::{
    context::Context,
    model::RESCALE_LAYER,
    tensor::{
        cache::ResourceCache,
        kind::{ReadBack, ReadWrite},
        matrix::Matrix,
        ops::{Activation, TensorCommand, TensorOp, TensorOpHook, TensorPass},
        shape::Shape,
        DeepClone, IntoPackedCursors, TensorCpu, TensorError, TensorGpu, TensorShape, TensorView,
    },
};

#[derive(Debug)]
pub struct Model<'a> {
    context: Context,
    info: ModelInfo,

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
pub struct ModelTensor<'a> {
    pub embed: Embed<'a>,
    pub head: Head,
    pub layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct LayerNorm {
    pub w: TensorGpu<f16, ReadWrite>,
    pub b: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug)]
pub struct Att {
    pub time_decay: TensorGpu<f32, ReadWrite>,
    pub time_first: TensorGpu<f32, ReadWrite>,

    pub time_mix_k: TensorGpu<f16, ReadWrite>,
    pub time_mix_v: TensorGpu<f16, ReadWrite>,
    pub time_mix_r: TensorGpu<f16, ReadWrite>,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
    pub w_o: Matrix,
}

#[derive(Debug)]
pub struct Ffn {
    pub time_mix_k: TensorGpu<f16, ReadWrite>,
    pub time_mix_r: TensorGpu<f16, ReadWrite>,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
}

#[derive(Debug)]
pub struct Layer {
    pub att_layer_norm: LayerNorm,
    pub ffn_layer_norm: LayerNorm,
    pub att: Att,
    pub ffn: Ffn,
}

#[derive(Debug)]
pub struct Embed<'a> {
    pub layer_norm: LayerNorm,
    pub w: TensorCpu<'a, f16>,
    pub u: Option<TensorGpu<f16, ReadWrite>>,
}

#[derive(Debug)]
pub struct Head {
    pub layer_norm: LayerNorm,
    pub w: Matrix,
}

/// Runtime buffers.
#[derive(Debug)]
pub struct Runtime {
    pub tokens: TensorGpu<u32, ReadWrite>,
    pub cursors: TensorGpu<u32, ReadWrite>,
    pub input: TensorGpu<f16, ReadWrite>,

    pub att_x: TensorGpu<f16, ReadWrite>,
    pub att_kx: TensorGpu<f16, ReadWrite>,
    pub att_vx: TensorGpu<f16, ReadWrite>,
    pub att_rx: TensorGpu<f16, ReadWrite>,
    pub att_k: TensorGpu<f32, ReadWrite>,
    pub att_v: TensorGpu<f32, ReadWrite>,
    pub att_r: TensorGpu<f32, ReadWrite>,
    pub att_o: TensorGpu<f16, ReadWrite>,

    pub ffn_x: TensorGpu<f16, ReadWrite>,
    pub ffn_kx: TensorGpu<f16, ReadWrite>,
    pub ffn_rx: TensorGpu<f16, ReadWrite>,
    pub ffn_k: TensorGpu<f16, ReadWrite>,
    pub ffn_v: TensorGpu<f16, ReadWrite>,
    pub ffn_r: TensorGpu<f16, ReadWrite>,

    pub aux_x: TensorGpu<f32, ReadWrite>,
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
            aux_x: context.tensor_init(shape),
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

impl TensorOpHook for Hook {}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct ModelState(TensorGpu<f32, ReadWrite>);

impl ModelState {
    #[inline]
    fn context(&self) -> &Context {
        self.0.context()
    }

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

impl super::ModelState for ModelState {
    type BackedState = BackedState;

    #[inline]
    fn max_batch(&self) -> usize {
        self.0.shape()[2]
    }

    fn load(&self, backed: &Self::BackedState) -> Result<()> {
        use super::BackedState;
        if backed.max_batch() != self.max_batch() {
            return Err(ModelError::BatchSize(backed.max_batch(), self.max_batch()).into());
        }
        let context = self.context();
        let host = context.tensor_from_data(self.shape(), &*backed.data)?;
        self.0.load(&host).map_err(|err| err.into())
    }

    fn load_batch(&self, backed: &Self::BackedState, batch: usize) -> Result<()> {
        use super::BackedState;
        if backed.max_batch() != 1 {
            return Err(ModelError::BatchSize(backed.max_batch(), 1).into());
        }
        let context = self.context();
        let shape = self.shape();
        let shape = Shape::new(shape[0], shape[1], 1, 1);
        let host = context.tensor_from_data(shape, &*backed.data)?;
        self.0.load_batch(&host, batch).map_err(|err| err.into())
    }

    async fn back(&self) -> Self::BackedState {
        let context = self.context();
        let shape = self.shape();
        let map = self.context().tensor_init(shape);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_tensor(self, &map).expect("back entire state");
        context.queue.submit(Some(encoder.finish()));

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

        let context = self.context();
        let shape = self.shape();
        let shape = Shape::new(shape[0], shape[1], 1, 1);
        let map = context.tensor_init(shape);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_tensor_batch(self, &map, batch)?;
        context.queue.submit(Some(encoder.finish()));

        let data = map.back_async().await.to_vec().into();
        Ok(BackedState { shape, data })
    }

    fn blit(&self, other: &Self) -> Result<(), TensorError> {
        let context = self.context();
        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_tensor(self, other)?;
        context.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn blit_batch(
        &self,
        other: &Self,
        from_batch: usize,
        to_batch: usize,
    ) -> Result<(), TensorError> {
        let context = self.context();
        let mut encoder = context.device.create_command_encoder(&Default::default());

        let op = TensorOp::blit(
            self.view(.., .., from_batch, ..)?,
            other.view(.., .., to_batch, ..)?,
        )?;

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        context.queue.submit(Some(encoder.finish()));
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
    pub const LN_EPS: f32 = 1.0e-5;
    pub const GN_EPS: f32 = 64.0e-5;
}

impl FromBuilder for Model<'_> {
    type Builder<'a> = ModelBuilder<'a>;
    type Error = anyhow::Error;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error> {
        let PreparedModelBuilder {
            context,
            info,
            loader,
            quant,
            embed_device,
            turbo,
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
                let discount = 2.0_f32.powi(-((layer / RESCALE_LAYER) as i32));
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
            turbo,
            token_chunk_size,
            tensor,
            runtime_cache: ResourceCache::new(1),
            output_cache: ResourceCache::new(1),
            softmax_cache: ResourceCache::new(1),
        })
    }
}

impl<'a> ModelBase for Model<'a> {
    type ModelTensor = ModelTensor<'a>;

    #[inline]
    fn context(&self) -> &Context {
        &self.context
    }

    #[inline]
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    #[inline]
    fn tensor(&self) -> &Self::ModelTensor {
        &self.tensor
    }
}

impl ModelSoftmaxInternal for Model<'_> {
    #[inline]
    fn request_softmax(&self, num_batch: usize) -> Arc<Softmax> {
        self.softmax_cache.request(num_batch, || {
            Softmax::new(&self.context, &self.info, num_batch)
        })
    }
}

impl ModelRunInternal for Model<'_> {
    type Hook = Hook;
    type Runtime = Runtime;
    type State = ModelState;

    #[inline]
    fn request_runtime(&self, num_token: usize) -> Arc<Runtime> {
        self.runtime_cache.request(num_token, || {
            Runtime::new(&self.context, &self.info, num_token, self.token_chunk_size)
        })
    }

    #[inline]
    fn request_output(&self, num_batch: usize) -> Arc<Output> {
        self.output_cache.request(num_batch, || {
            Output::new(&self.context, &self.info, num_batch)
        })
    }

    #[inline]
    fn token_chunk_size(&self) -> usize {
        self.token_chunk_size
    }

    #[inline]
    fn turbo(&self, num_token: usize) -> bool {
        self.turbo && num_token == self.token_chunk_size
    }

    fn run_internal(
        &self,
        tokens: Vec<Vec<u16>>,
        state: &ModelState,
        should_output: Vec<bool>,
        hooks: &HookMap<Self::Hook, Self, Self::State, Self::Runtime>,
    ) -> Result<(TensorGpu<f32, ReadBack>, Vec<Option<usize>>)> {
        let context = &self.context;
        let tensor = &self.tensor;

        let input = self.create_input(&tensor.embed.w, &tokens)?;
        let num_batch = input.num_batch();
        let num_token = input.num_token();
        assert_ne!(num_token, 0);

        let turbo = self.turbo(num_token);

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

        let hook_op = |hook: Hook| -> Result<TensorOp, TensorError> {
            hooks
                .get(&hook)
                .map(|f| f(self, state, &buffer))
                .unwrap_or_else(|| Ok(TensorOp::List(vec![])))
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
            hook_op(Hook::PostEmbedLoaded)?,
            TensorOp::layer_norm(
                &tensor.embed.layer_norm.w,
                &tensor.embed.layer_norm.b,
                &buffer.input,
                None,
                Self::LN_EPS,
            )?,
            hook_op(Hook::PostEmbedLayerNorm)?,
        ]);

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let ops = TensorOp::List(ops);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);

        for (index, layer) in tensor.layers.iter().enumerate() {
            encoder.copy_tensor(&buffer.input, &buffer.att_x)?;

            let ops = TensorOp::List(vec![
                hook_op(Hook::PreAtt(index))?,
                TensorOp::layer_norm(
                    &layer.att_layer_norm.w,
                    &layer.att_layer_norm.b,
                    &buffer.att_x,
                    None,
                    Self::LN_EPS,
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
                hook_op(Hook::PostAttTokenShift(index))?,
                hook_op(Hook::PreAttLinear(index))?,
                layer.att.w_k.matmul_op(
                    buffer.att_kx.view(.., .., .., ..)?,
                    buffer.att_k.view(.., .., .., ..)?,
                    Activation::None,
                    turbo,
                )?,
                layer.att.w_v.matmul_op(
                    buffer.att_vx.view(.., .., .., ..)?,
                    buffer.att_v.view(.., .., .., ..)?,
                    Activation::None,
                    turbo,
                )?,
                layer.att.w_r.matmul_op(
                    buffer.att_rx.view(.., .., .., ..)?,
                    buffer.att_r.view(.., .., .., ..)?,
                    Activation::None,
                    turbo,
                )?,
                hook_op(Hook::PostAttLinear(index))?,
                hook_op(Hook::PreAttTimeMix(index))?,
                TensorOp::blit(
                    buffer.att_x.view(.., .., .., ..)?,
                    buffer.aux_x.view(.., .., .., ..)?,
                )?,
                TensorOp::time_mix_v4(
                    &buffer.cursors,
                    &layer.att.time_decay,
                    &layer.att.time_first,
                    state.att(index)?,
                    &buffer.att_k,
                    &buffer.att_v,
                    &buffer.att_r,
                    &buffer.aux_x,
                )?,
                TensorOp::blit(
                    buffer.aux_x.view(.., .., .., ..)?,
                    buffer.att_x.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostAttTimeMix(index))?,
                hook_op(Hook::PreAttOut(index))?,
                layer.att.w_o.matmul_op(
                    buffer.att_x.view(.., .., .., ..)?,
                    buffer.att_o.view(.., .., .., ..)?,
                    Activation::None,
                    turbo,
                )?,
                hook_op(Hook::PostAttOut(index))?,
                TensorOp::add(
                    buffer.input.view(.., .., .., ..)?,
                    buffer.att_o.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostAtt(index))?,
            ]);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&ops);
            drop(pass);

            encoder.copy_tensor(&buffer.att_o, &buffer.ffn_x)?;

            let ops = TensorOp::List(vec![
                hook_op(Hook::PreFfn(index))?,
                TensorOp::layer_norm(
                    &layer.ffn_layer_norm.w,
                    &layer.ffn_layer_norm.b,
                    &buffer.ffn_x,
                    None,
                    Self::LN_EPS,
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
                    turbo,
                )?,
                hook_op(Hook::PostFfnActivate(index))?,
                layer.ffn.w_v.matmul_op(
                    buffer.ffn_k.view(.., .., .., ..)?,
                    buffer.ffn_v.view(.., .., .., ..)?,
                    Activation::None,
                    turbo,
                )?,
                layer.ffn.w_r.matmul_op(
                    buffer.ffn_rx.view(.., .., .., ..)?,
                    buffer.ffn_r.view(.., .., .., ..)?,
                    Activation::None,
                    turbo,
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

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&ops);
            drop(pass);

            if (index + 1) % RESCALE_LAYER == 0 {
                let op = TensorOp::discount(&buffer.ffn_x, 0.5)?;
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.execute_tensor_op(&op);
                drop(pass);
            }

            if index != self.info.num_layer - 1 {
                encoder.copy_tensor(&buffer.ffn_x, &buffer.input)?;
            }
        }

        if num_header > 0 {
            let ops = TensorOp::List(vec![
                hook_op(Hook::PreHead)?,
                TensorOp::layer_norm(
                    &tensor.head.layer_norm.w,
                    &tensor.head.layer_norm.b,
                    head_x,
                    None,
                    Self::LN_EPS,
                )?,
                hook_op(Hook::PostHeadLayerNorm)?,
                tensor.head.w.matmul_op(
                    head_x.view(.., .., .., ..)?,
                    output.head_o.view(.., .., .., ..)?,
                    Activation::None,
                    self.turbo(num_header),
                )?,
                hook_op(Hook::PostHead)?,
            ]);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&head_ops);
            pass.execute_tensor_op(&ops);
            drop(pass);

            encoder.copy_tensor(&output.head_o, &output.map)?;
        }

        context.queue.submit(Some(encoder.finish()));
        Ok((output.map.clone(), redirect))
    }
}
