use std::{convert::Infallible, marker::PhantomData};

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::DeserializeSeed;

use super::{
    loader::Reader,
    run::{Header, HookMap, ModelRunInternal},
    Build, BuildFuture, ModelBase, ModelBuilder, ModelInfo, OutputType, PreparedModelBuilder,
    Quant, StateBuilder, MIN_TOKEN_CHUNK_SIZE,
};
use crate::{
    context::Context,
    model::RESCALE_LAYER,
    num::Float,
    tensor::{
        kind::ReadWrite,
        matrix::Matrix,
        ops::{Activation, TensorCommand, TensorOp, TensorPass},
        shape::{Shape, TensorDimension},
        DeepClone, IntoPackedCursors, TensorCpu, TensorError, TensorGpu, TensorGpuView,
        TensorReshape, TensorShape,
    },
};

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct Model<'a, F: Float> {
    context: Context,
    info: ModelInfo,

    /// Whether to use fp16 GEMM for matmul computations.
    turbo: bool,
    /// To prevent the GPU device from lost, this limits the maximum batch-token it processes one time.
    token_chunk_size: usize,

    tensor: ModelTensor<'a>,
    _phantom: PhantomData<F>,
}

#[derive(Debug, Serialize, DeserializeSeed)]
pub struct ModelTensor<'a> {
    pub embed: Embed<'a>,
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
pub struct Embed<'a> {
    pub layer_norm: LayerNorm,
    pub w: TensorCpu<'a, f16>,
    pub u: Option<TensorGpu<f16, ReadWrite>>,
}

#[derive(Debug, Serialize, DeserializeSeed)]
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
    pub fn new(context: &Context, info: &ModelInfo, num_token: usize, max_token: usize) -> Self {
        let shape = Shape::new(info.num_emb, num_token, 1, 1);
        let tokens_shape = Shape::new(num_token, 1, 1, 1);
        let cursors_shape = Shape::new(max_token, 1, 1, 1);
        let hidden_shape = Shape::new(info.num_hidden, num_token, 1, 1);
        let time_mix_shape = Shape::new(info.num_emb, num_token, 5, 1);
        let time_mix_x_shape = Shape::new(Model::<F>::TIME_MIX_ADAPTER_SIZE, 5, num_token, 1);
        let time_mix_t_shape = Shape::new(Model::<F>::TIME_MIX_ADAPTER_SIZE, num_token, 5, 1);
        let time_decay_shape = Shape::new(Model::<F>::TIME_DECAY_ADAPTER_SIZE, num_token, 1, 1);

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

#[derive(Debug, Clone)]
pub struct ModelState {
    info: ModelInfo,
    num_batch: usize,
    chunk_size: usize,
    head_size: usize,
    state: Vec<TensorGpu<f32, ReadWrite>>,
}

impl ModelState {
    fn att(&self, layer: usize) -> Result<TensorGpuView<f32>, TensorError> {
        let chunk = layer / self.chunk_size;
        let offset = layer % self.chunk_size;
        let head_size = self.info.num_emb / self.info.num_head;

        let start = offset * (head_size + 2);
        let end = start + head_size + 1;
        self.state[chunk].view(.., start..end, .., ..)
    }

    fn ffn(&self, layer: usize) -> Result<TensorGpuView<f32>, TensorError> {
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

impl Build<ModelState> for StateBuilder {
    type Error = Infallible;

    fn build(self) -> Result<ModelState, Self::Error> {
        let StateBuilder {
            context,
            info,
            num_batch,
            chunk_size,
        } = self;
        let num_chunk = (info.num_layer + chunk_size - 1) / chunk_size;
        let head_size = info.num_emb / info.num_head;
        let state = (0..num_chunk)
            .map(|_| {
                let data = (0..num_batch)
                    .map(|_| vec![0.0; chunk_size * info.num_emb * (head_size + 2)])
                    .collect_vec()
                    .concat();
                context
                    .tensor_from_data(
                        Shape::new(info.num_emb, chunk_size * (head_size + 2), num_batch, 1),
                        data,
                    )
                    .expect("state creation")
            })
            .collect();
        Ok(ModelState {
            info,
            num_batch,
            chunk_size,
            head_size,
            state,
        })
    }
}

impl super::ModelState for ModelState {
    type BackedState = BackedState;

    #[inline]
    fn num_batch(&self) -> usize {
        self.num_batch
    }

    fn load(&self, backed: &BackedState) -> Result<(), TensorError> {
        use super::BackedState;
        if backed.num_batch() != self.num_batch() {
            return Err(TensorError::Batch(backed.num_batch(), self.num_batch()));
        }
        for (state, (shape, backed)) in self.state.iter().zip(backed.data.iter()) {
            let context = state.context();
            let host = context.tensor_from_data(*shape, backed)?;
            state.load(&host)?;
        }
        Ok(())
    }

    fn load_batch(&self, backed: &BackedState, batch: usize) -> Result<(), TensorError> {
        use super::BackedState;
        if backed.num_batch() != 1 {
            return Err(TensorError::Batch(backed.num_batch(), 1));
        }
        for (state, (_, backed)) in self.state.iter().zip(backed.data.iter()) {
            let context = state.context();
            let shape = state.shape();
            let shape = Shape::new(shape[0], shape[1], 1, 1);
            let host = context.tensor_from_data(shape, backed)?;
            state.load_batch(&host, batch)?;
        }
        Ok(())
    }

    async fn back(&self) -> BackedState {
        let num_batch = self.num_batch;
        let chunk_size = self.chunk_size;
        let head_size = self.head_size;

        let mut data = Vec::with_capacity(self.state.len());
        for state in self.state.iter() {
            let context = state.context();
            let shape = state.shape();
            let map = context.tensor_init(shape);

            let mut encoder = context.device.create_command_encoder(&Default::default());
            encoder.copy_tensor(state, &map).expect("back entire state");
            context.queue.submit(Some(encoder.finish()));

            let host = map.back().await;
            data.push((shape, host.to_vec()))
        }

        BackedState {
            num_batch,
            chunk_size,
            head_size,
            data,
        }
    }

    async fn back_batch(&self, batch: usize) -> Result<BackedState, TensorError> {
        let mut data = Vec::with_capacity(self.state.len());
        for state in self.state.iter() {
            let context = state.context();
            let shape = state.shape();
            let shape = Shape::new(shape[0], shape[1], 1, 1);
            let map = context.tensor_init(shape);

            let mut encoder = context.device.create_command_encoder(&Default::default());
            encoder.copy_tensor_batch(state, &map, batch)?;
            context.queue.submit(Some(encoder.finish()));

            let host = map.back().await;
            data.push((shape, host.to_vec()));
        }

        Ok(BackedState {
            num_batch: 1,
            chunk_size: self.chunk_size,
            head_size: self.head_size,
            data,
        })
    }

    fn blit(&self, other: &ModelState) -> Result<(), TensorError> {
        for (state, other) in self.state.iter().zip(other.state.iter()) {
            let context = state.context();
            state.check_shape(other.shape())?;
            let mut encoder = context.device.create_command_encoder(&Default::default());
            encoder.copy_tensor(state, other)?;
            context.queue.submit(Some(encoder.finish()));
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
            let context = state.context();
            let mut encoder = context.device.create_command_encoder(&Default::default());

            let op = TensorOp::blit(
                state.view(.., .., from_batch, ..)?,
                other.view(.., .., to_batch, ..)?,
            )?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
            drop(pass);

            context.queue.submit(Some(encoder.finish()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackedState {
    pub num_batch: usize,
    pub chunk_size: usize,
    pub head_size: usize,
    pub data: Vec<(Shape, Vec<f32>)>,
}

impl Build<BackedState> for StateBuilder {
    type Error = Infallible;

    fn build(self) -> Result<BackedState, Self::Error> {
        let StateBuilder {
            info,
            num_batch,
            chunk_size,
            ..
        } = self;
        let head_size = info.num_emb / info.num_head;
        let shape = Shape::new(info.num_emb, chunk_size * (head_size + 2), num_batch, 1);
        let data = (0..info.num_layer)
            .map(|_| {
                (0..num_batch)
                    .map(|_| vec![0.0; chunk_size * info.num_emb * (head_size + 2)])
                    .collect_vec()
                    .concat()
            })
            .map(|x| (shape, x))
            .collect_vec();
        Ok(BackedState {
            num_batch,
            chunk_size,
            head_size,
            data,
        })
    }
}

impl super::BackedState for BackedState {
    #[inline]
    fn num_batch(&self) -> usize {
        self.num_batch
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

impl<'a, F: Float> Model<'a, F> {
    const TIME_MIX_ADAPTER_SIZE: usize = 32;
    const TIME_DECAY_ADAPTER_SIZE: usize = 64;

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
            let time_decay = loader.load_vector_f16(format!("{att}.time_decay")).await?;
            let time_first = loader.load_vector_f32(format!("{att}.time_first")).await?;
            let time_mix_x = loader.load_vector_f16(format!("{att}.time_mix_x")).await?;
            let time_mix_w = loader.load_vector_f16(format!("{att}.time_mix_w")).await?;
            let time_mix_k = loader.load_vector_f16(format!("{att}.time_mix_k")).await?;
            let time_mix_v = loader.load_vector_f16(format!("{att}.time_mix_v")).await?;
            let time_mix_r = loader.load_vector_f16(format!("{att}.time_mix_r")).await?;
            let time_mix_g = loader.load_vector_f16(format!("{att}.time_mix_g")).await?;

            let time_decay_w1 = loader
                .load_matrix_f16(format!("{att}.time_decay_w1"))
                .await?;
            let time_decay_w2 = loader
                .load_matrix_f16(format!("{att}.time_decay_w2"))
                .await?;

            let time_mix_w1 = loader.load_matrix_f16(format!("{att}.time_mix_w1")).await?;
            let time_mix_w2 = loader.load_matrix_f16(format!("{att}.time_mix_w2")).await?;

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
                time_mix_x,
                time_mix_w,
                time_mix_k,
                time_mix_v,
                time_mix_r,
                time_mix_g,
                time_decay_w1: Matrix::Fp16(time_decay_w1),
                time_decay_w2: Matrix::Fp16(time_decay_w2),
                time_mix_w1: Matrix::Fp16(time_mix_w1),
                time_mix_w2: Matrix::Fp16(time_mix_w2),
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

impl<'a, F: Float> ModelRunInternal for Model<'a, F> {
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
        let head_size = self.info.num_emb / self.info.num_head;
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
                    Self::LN_EPS,
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
                    turbo,
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
                    turbo,
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
                layer.att.w_g.matmul_op(
                    buffer.att_gx.view(.., .., .., ..)?,
                    buffer.att_g.view(.., .., .., ..)?,
                    Activation::None,
                    turbo,
                )?,
                hook_op(Hook::PostAttLinear(index))?,
                hook_op(Hook::PreAttTimeDecayAdapt(index))?,
                layer.att.time_decay_w1.matmul_op(
                    buffer.att_wx.view(.., .., .., ..)?,
                    buffer.att_w.view(.., .., .., ..)?,
                    Activation::Tanh,
                    turbo,
                )?,
                hook_op(Hook::PostAttTimeDecayAdaptActivate(index))?,
                layer.att.time_decay_w2.matmul_op(
                    buffer.att_w.view(.., .., .., ..)?,
                    buffer.time_decay.view(.., .., .., ..)?,
                    Activation::None,
                    turbo,
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
                    Self::GN_EPS,
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
