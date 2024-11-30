use std::{convert::Infallible, marker::PhantomData};

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::DeserializeSeed;

use super::{
    loader::Reader,
    run::{Header, HookMap, ModelRunInternal},
    Build, BuildFuture, ModelBase, ModelBuilder, ModelInfo, PreparedModelBuilder, Quant,
    StateBuilder, MIN_TOKEN_CHUNK_SIZE,
};
use crate::{
    context::Context,
    model::{OutputType, RESCALE_LAYER},
    num::Float,
    tensor::{
        kind::ReadWrite,
        matrix::Matrix,
        ops::{Activation, TensorCommand, TensorOp},
        serialization::Seed,
        shape::{Shape, TensorDimension},
        DeepClone, IntoPackedCursors, TensorCpu, TensorError, TensorGpu, TensorGpuView,
        TensorReshape, TensorShape,
    },
};

#[derive(Debug, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Model<F: Float> {
    context: Context,
    info: ModelInfo,

    /// Whether to use fp16 GEMM for matmul computations.
    turbo: bool,
    /// To prevent the GPU device from lost, this limits the maximum batch-token it processes one time.
    token_chunk_size: usize,

    tensor: ModelTensor,
    _phantom: PhantomData<F>,
}

#[derive(Debug, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct ModelTensor {
    pub embed: Embed,
    pub head: Head,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct LayerNorm {
    pub w: TensorGpu<f16, ReadWrite>,
    pub b: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug, Serialize, DeserializeSeed)]
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

#[derive(Debug, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Ffn {
    pub time_mix_k: TensorGpu<f16, ReadWrite>,
    pub time_mix_r: TensorGpu<f16, ReadWrite>,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
}

#[derive(Debug, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Layer {
    pub att_layer_norm: LayerNorm,
    pub ffn_layer_norm: LayerNorm,
    pub att: Att,
    pub ffn: Ffn,
}

#[derive(Debug, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Embed {
    pub layer_norm: LayerNorm,
    pub w: TensorCpu<f16>,
    pub u: Option<TensorGpu<f16, ReadWrite>>,
}

#[derive(Debug, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
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
        let tokens_shape = Shape::new(num_token, 1, 1, 1);
        let cursors_shape = Shape::new(num_token, 1, 1, 1);
        let hidden_shape = Shape::new(info.num_hidden, num_token, 1, 1);

        Self {
            tokens: context.tensor_init(tokens_shape),
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
            let host = context.tensor_from_data(*shape, backed.clone())?;
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
            let host = context.tensor_from_data(shape, backed.clone())?;
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
            encoder.copy_tensor_batch(state, &map, batch, 0)?;
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
            let op = TensorOp::blit(
                state.view(.., .., from_batch, ..)?,
                other.view(.., .., to_batch, ..)?,
            )?;
            context.queue.submit(context.encode(&op));
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
            .collect();
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

impl<F: Float> Model<F> {
    pub const LN_EPS: f32 = 1.0e-5;
    pub const GN_EPS: f32 = 64.0e-5;
}

impl<R: Reader, F: Float> BuildFuture<Model<F>> for ModelBuilder<R> {
    type Error = anyhow::Error;

    async fn build(self) -> Result<Model<F>, Self::Error> {
        let PreparedModelBuilder {
            context,
            info,
            loader,
            quant,
            embed_device,
            turbo,
            token_chunk_size,
        } = self.prepare()?;

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

        let load_matrix = |name: String, quant: Quant| loader.load_matrix(name, quant);
        let load_matrix_discount = |name: String, quant: Quant, discount: f32| {
            loader.load_matrix_discount(name, quant, discount)
        };

        let mut layers = vec![];
        for layer in 0..info.num_layer {
            let quant = quant.get(&layer).copied().unwrap_or_default();
            let discount = 2.0_f32.powi(-((layer / RESCALE_LAYER) as i32));

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

impl<F: Float> ModelBase for Model<F> {
    #[inline]
    fn context(&self) -> &Context {
        &self.context
    }

    #[inline]
    fn info(&self) -> &ModelInfo {
        &self.info
    }
}

impl<F: Float> ModelRunInternal for Model<F> {
    type Hook = Hook;
    type State = ModelState;
    type Tensor = ModelTensor;
    type Runtime = Runtime<F>;
    type Header = Header<F>;

    #[inline]
    fn tensor(&self) -> &Self::Tensor {
        &self.tensor
    }

    #[inline]
    fn checkout_runtime(&self, num_token: usize) -> Self::Runtime {
        Runtime::new(&self.context, &self.info, num_token)
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

        context.maintain();

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
                .unwrap_or_else(|| Ok(TensorOp::empty()))
        };

        // collect and group copy operations
        let (head_ops, head_x) = if num_token == 1 || num_token == num_header {
            (TensorOp::empty(), &buffer.x)
        } else {
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
            (TensorOp::List(ops), &header.head_x)
        };

        let mut ops = vec![];

        let cursors = input.cursors.into_cursors();
        // cursors.resize(self.token_chunk_size, 0);

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
                Self::LN_EPS,
            )?,
            TensorOp::blit(
                buffer.input.view(.., .., .., ..)?,
                buffer.x.view(.., .., .., ..)?,
            )?,
            hook_op(Hook::PostEmbedLayerNorm)?,
        ]);

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

            ops.append(&mut vec![
                TensorOp::blit(
                    buffer.x.view(.., .., .., ..)?,
                    buffer.att_x.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PreAtt(index))?,
                TensorOp::layer_norm(
                    &layer.att_layer_norm.w,
                    &layer.att_layer_norm.b,
                    &buffer.att_x,
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
                    buffer.att_o.view(.., .., .., ..)?,
                    buffer.x.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostAtt(index))?,
            ]);

            ops.append(&mut vec![
                TensorOp::blit(
                    buffer.x.view(.., .., .., ..)?,
                    buffer.ffn_x.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PreFfn(index))?,
                TensorOp::layer_norm(
                    &layer.ffn_layer_norm.w,
                    &layer.ffn_layer_norm.b,
                    &buffer.ffn_x,
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
                    buffer.ffn_x.view(.., .., .., ..)?,
                    buffer.x.view(.., .., .., ..)?,
                )?,
                hook_op(Hook::PostFfn(index))?,
            ]);

            if (index + 1) % RESCALE_LAYER == 0 {
                ops.push(TensorOp::discount(&buffer.x, 0.5, 0.0)?);
            }
        }

        if num_header > 0 {
            ops.append(&mut vec![
                head_ops,
                hook_op(Hook::PreHead)?,
                TensorOp::layer_norm(
                    &tensor.head.layer_norm.w,
                    &tensor.head.layer_norm.b,
                    head_x,
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
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok((header.head_o.clone(), redirect))
    }
}
