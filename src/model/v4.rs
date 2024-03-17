use std::{convert::Infallible, marker::PhantomData};

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::{Deref, DerefMut};

use super::{
    loader::Reader,
    run::{Header, HookMap, ModelRunInternal},
    Build, BuildFuture, ModelBase, ModelBuilder, ModelInfo, OutputType, PreparedModelBuilder,
    Quant, StateBuilder, MIN_TOKEN_CHUNK_SIZE,
};
use crate::{
    context::Context,
    model::RESCALE_LAYER,
    num::{Float, Hom},
    tensor::{
        kind::ReadWrite,
        matrix::Matrix,
        ops::{Activation, TensorCommand, TensorOp, TensorPass},
        shape::Shape,
        DeepClone, IntoPackedCursors, TensorCpu, TensorError, TensorGpu, TensorGpuView,
        TensorShape,
    },
};

#[derive(Debug, Serialize)]
pub struct Model<'a, F: Float> {
    #[serde(skip)]
    context: Context,
    info: ModelInfo,

    /// Whether to use fp16 GEMM for matmul computations.
    turbo: bool,
    /// To prevent the GPU device from lost, this limits the maximum batch-token it processes one time.
    token_chunk_size: usize,

    tensor: ModelTensor<'a>,
    _phantom: PhantomData<F>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelTensor<'a> {
    pub embed: Embed<'a>,
    pub head: Head,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LayerNorm {
    pub w: TensorGpu<f16, ReadWrite>,
    pub b: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
pub struct Ffn {
    pub time_mix_k: TensorGpu<f16, ReadWrite>,
    pub time_mix_r: TensorGpu<f16, ReadWrite>,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Layer {
    pub att_layer_norm: LayerNorm,
    pub ffn_layer_norm: LayerNorm,
    pub att: Att,
    pub ffn: Ffn,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Embed<'a> {
    pub layer_norm: LayerNorm,
    pub w: TensorCpu<'a, f16>,
    pub u: Option<TensorGpu<f16, ReadWrite>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Head {
    pub layer_norm: LayerNorm,
    pub w: Matrix,
}

/// Runtime buffers.
#[derive(Debug)]
pub struct Runtime<F: Float> {
    pub tokens: TensorGpu<u32, ReadWrite>,
    pub cursors: TensorGpu<u32, ReadWrite>,
    pub input: TensorGpu<F, ReadWrite>,

    pub att_x: TensorGpu<F, ReadWrite>,
    pub att_kx: TensorGpu<F, ReadWrite>,
    pub att_vx: TensorGpu<F, ReadWrite>,
    pub att_rx: TensorGpu<F, ReadWrite>,
    pub att_k: TensorGpu<f32, ReadWrite>,
    pub att_v: TensorGpu<f32, ReadWrite>,
    pub att_r: TensorGpu<f32, ReadWrite>,
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

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct ModelState(TensorGpu<f32, ReadWrite>);

impl ModelState {
    #[inline]
    fn context(&self) -> &Context {
        self.0.context()
    }

    fn att(&self, layer: usize) -> Result<TensorGpuView<f32>, TensorError> {
        let start = 5 * layer;
        let end = start + 4;
        self.view(.., start..end, .., ..)
    }

    fn ffn(&self, layer: usize) -> Result<TensorGpuView<f32>, TensorError> {
        let start = 5 * layer + 4;
        self.view(.., start..=start, .., ..)
    }
}

impl DeepClone for ModelState {
    fn deep_clone(&self) -> Self {
        Self(self.0.deep_clone())
    }
}

impl Build<ModelState> for StateBuilder {
    type Error = Infallible;

    fn build(self) -> Result<ModelState, Self::Error> {
        let StateBuilder {
            context,
            info,
            num_batch,
            ..
        } = self;
        let data = (0..num_batch)
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
                Shape::new(info.num_emb, 5 * info.num_layer, num_batch, 1),
                data,
            )
            .unwrap();
        Ok(ModelState(state))
    }
}

impl super::ModelState for ModelState {
    type BackedState = BackedState;

    #[inline]
    fn num_batch(&self) -> usize {
        self.0.shape()[2]
    }

    fn load(&self, backed: &Self::BackedState) -> Result<(), TensorError> {
        use super::BackedState;
        if backed.num_batch() != self.num_batch() {
            return Err(TensorError::Batch(backed.num_batch(), self.num_batch()));
        }
        let context = self.context();
        let host = context.tensor_from_data(self.shape(), &*backed.data)?;
        self.0.load(&host)
    }

    fn load_batch(&self, backed: &Self::BackedState, batch: usize) -> Result<(), TensorError> {
        use super::BackedState;
        if backed.num_batch() != 1 {
            return Err(TensorError::Batch(backed.num_batch(), 1));
        }
        let context = self.context();
        let shape = self.shape();
        let shape = Shape::new(shape[0], shape[1], 1, 1);
        let host = context.tensor_from_data(shape, &*backed.data)?;
        self.0.load_batch(&host, batch)
    }

    async fn back(&self) -> Self::BackedState {
        let context = self.context();
        let shape = self.shape();
        let map = self.context().tensor_init(shape);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_tensor(self, &map).expect("back entire state");
        context.queue.submit(Some(encoder.finish()));

        let data = map.back_async().await.into();
        BackedState { shape, data }
    }

    async fn back_batch(&self, batch: usize) -> Result<Self::BackedState, TensorError> {
        let context = self.context();
        let shape = self.shape();
        let shape = Shape::new(shape[0], shape[1], 1, 1);
        let map = context.tensor_init(shape);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_tensor_batch(self, &map, batch)?;
        context.queue.submit(Some(encoder.finish()));

        let data = map.back_async().await.into();
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackedState {
    pub shape: Shape,
    pub data: Vec<f32>,
}

impl Build<BackedState> for StateBuilder {
    type Error = Infallible;

    fn build(self) -> Result<BackedState, Self::Error> {
        let StateBuilder {
            info, num_batch, ..
        } = self;
        let shape = Shape::new(info.num_emb, 5 * info.num_layer, num_batch, 1);
        let data = (0..num_batch)
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
        Ok(BackedState { shape, data })
    }
}

impl super::BackedState for BackedState {
    #[inline]
    fn num_batch(&self) -> usize {
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

impl<'a, F: Float> Model<'a, F> {
    pub const LN_EPS: f32 = 1.0e-5;
    pub const GN_EPS: f32 = 64.0e-5;
}

impl<'a, R: Reader, F: Float> BuildFuture<Model<'a, F>> for ModelBuilder<R> {
    type Error = anyhow::Error;

    async fn build(self) -> Result<Model<'a, F>, Self::Error> {
        let PreparedModelBuilder {
            context,
            info,
            loader,
            quant,
            embed_device,
            turbo,
            token_chunk_size,
        } = self.prepare().await?;

        let embed = Embed {
            layer_norm: LayerNorm {
                w: loader.load_vector_f16("blocks.0.ln0.weight").await?,
                b: loader.load_vector_f16("blocks.0.ln0.bias").await?,
            },
            w: loader.load_embed().await?,
            u: match embed_device {
                super::EmbedDevice::Cpu => None,
                super::EmbedDevice::Gpu => Some(loader.load_matrix_f16("emb.weight").await?),
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
            let discount = 2.0_f32.powi(-((layer / RESCALE_LAYER) as i32));

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
                .load_vector_exp_f32(format!("{att}.time_decay"))
                .await?;
            let time_first = loader.load_vector_f32(format!("{att}.time_first")).await?;
            let time_mix_k = loader.load_vector_f16(format!("{att}.time_mix_k")).await?;
            let time_mix_v = loader.load_vector_f16(format!("{att}.time_mix_v")).await?;
            let time_mix_r = loader.load_vector_f16(format!("{att}.time_mix_r")).await?;

            let att = Att {
                time_decay,
                time_first,
                time_mix_k,
                time_mix_v,
                time_mix_r,
                w_k: load_matrix(format!("{att}.key.weight"), quant).await?,
                w_v: load_matrix(format!("{att}.value.weight"), quant).await?,
                w_r: load_matrix(format!("{att}.receptance.weight"), quant).await?,
                w_o: load_matrix_discount(format!("{att}.output.weight"), quant, discount).await?,
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
        Ok(Model {
            context,
            info,
            turbo,
            token_chunk_size,
            tensor,
            _phantom: PhantomData,
        })
    }
}

impl<'a, F: Float> ModelBase for Model<'a, F> {
    #[inline]
    fn context(&self) -> &Context {
        &self.context
    }

    #[inline]
    fn info(&self) -> &ModelInfo {
        &self.info
    }
}

impl<'a, F: Float + Hom<f16>> ModelRunInternal for Model<'a, F> {
    type Hook = Hook;
    type State = ModelState;
    type Tensor = ModelTensor<'a>;
    type Runtime = Runtime<F>;
    type Header = Header<F>;

    #[inline]
    fn tensor(&self) -> &Self::Tensor {
        &self.tensor
    }

    #[inline]
    fn checkout_runtime(&self, num_token: usize) -> Self::Runtime {
        Runtime::new(&self.context, &self.info, num_token, self.token_chunk_size)
    }

    #[inline]
    fn checkout_header(&self, num_batch: usize) -> Self::Header {
        Header::new(&self.context, &self.info, num_batch)
    }

    #[inline]
    fn token_chunk_size(&self) -> usize {
        self.token_chunk_size
    }

    #[inline]
    fn turbo(&self, num_token: usize) -> bool {
        self.turbo && num_token % MIN_TOKEN_CHUNK_SIZE == 0
    }

    fn run_internal(
        &self,
        tokens: Vec<Vec<u16>>,
        state: &ModelState,
        outputs: Vec<Option<OutputType>>,
        hooks: &HookMap<Self::Hook, Self::Tensor, Self::State, Self::Runtime, Self::Header>,
    ) -> Result<(TensorGpu<f32, ReadWrite>, Vec<std::ops::Range<usize>>), TensorError> {
        let context = &self.context;
        let tensor = &self.tensor;

        let input = self.create_input(&tensor.embed.w, &tokens)?;
        let num_batch = input.num_batch();
        let num_token = input.num_token();
        assert_ne!(num_token, 0);

        let turbo = self.turbo(num_token);

        // collect batch output copy commands for later
        let mut redirect = vec![0..0; num_batch];
        let headers = input
            .cursors
            .iter()
            .filter(|cursor| cursor.len > 0)
            .fold((0, vec![]), |(index, mut header), cursor| {
                match outputs[cursor.batch] {
                    Some(OutputType::Last) => {
                        redirect[cursor.batch] = index..index + 1;
                        header.push(cursor.token + cursor.len - 1);
                        (index + 1, header)
                    }
                    Some(OutputType::Full) => {
                        redirect[cursor.batch] = index..index + cursor.len;
                        let r = (cursor.token..cursor.token + cursor.len).collect();
                        (index + cursor.len, [header, r].concat())
                    }
                    None => (index, header),
                }
            })
            .1;
        let num_header = headers.len();

        let buffer = self.checkout_runtime(num_token);
        let header = self.checkout_header(num_header.max(1));

        let hook_op = |hook: Hook| -> Result<TensorOp, TensorError> {
            hooks
                .get(&hook)
                .map(|f| f(&self.tensor, state, &buffer, &header))
                .unwrap_or_else(|| Ok(TensorOp::List(vec![])))
        };

        // collect and group copy operations
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
                    let output = header.head_x.view(.., start..end, .., ..)?;
                    ops.push(TensorOp::blit(input, output)?);

                    start = end;
                }
                end += 1;
            }
            (TensorOp::List(ops), &header.head_x)
        };

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
                    header.head_o.view(.., .., .., ..)?,
                    Activation::None,
                    self.turbo(num_header),
                )?,
                hook_op(Hook::PostHead)?,
            ]);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&head_ops);
            pass.execute_tensor_op(&ops);
            drop(pass);
        }

        context.queue.submit(Some(encoder.finish()));
        Ok((header.head_o.clone(), redirect))
    }
}
