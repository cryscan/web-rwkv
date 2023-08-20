use bitflags::bitflags;
use derive_getters::Getters;
use half::f16;

use crate::{
    context::Context,
    tensor::{ReadBack, ReadWrite, Shape, TensorCpu, TensorError, TensorGpu, Uniform},
};

#[derive(Getters)]
pub struct Model<'a> {
    pub(crate) info: ModelInfo,
    pub(crate) context: Context,
    #[getter(skip)]
    tensor: ModelTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub num_emb: usize,
    pub num_vocab: usize,

    /// The head matrix is too big for a storage buffer so it's divided into chunks.
    pub max_head_chunk: usize,
    /// To prevent the GPU device from lost, this limits the maximum batch-token it processes one time.
    pub max_batch_token_chunk: usize,
}

bitflags! {
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct LayerFlags: u64 {
    }
}

impl LayerFlags {
    pub fn from_layer(layer: u64) -> LayerFlags {
        LayerFlags::from_bits_retain(1 << layer)
    }

    pub fn contains_layer(&self, layer: u64) -> bool {
        self.contains(LayerFlags::from_layer(layer))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum Quantization {
    /// No quantization.
    #[default]
    None,
    /// Use int8 quantization, given layers to be quantized.
    Int8(LayerFlags),
}

enum Matrix<'a> {
    Fp16(TensorGpu<'a, f16, ReadWrite>),
    Int8 {
        w: Box<TensorGpu<'a, u8, ReadWrite>>,
        mx: Box<TensorGpu<'a, f16, ReadWrite>>,
        rx: Box<TensorGpu<'a, f16, ReadWrite>>,
        my: Box<TensorGpu<'a, f16, ReadWrite>>,
        ry: Box<TensorGpu<'a, f16, ReadWrite>>,
    },
}

struct ModelTensor<'a> {
    embed: Embed<'a>,
    head: Head<'a>,
    layers: Vec<Layer<'a>>,
}

struct LayerNorm<'a> {
    w: TensorGpu<'a, f32, ReadWrite>,
    b: TensorGpu<'a, f32, ReadWrite>,
}

struct Att<'a> {
    time_decay: TensorGpu<'a, f32, ReadWrite>,
    time_first: TensorGpu<'a, f32, ReadWrite>,

    time_mix_k: TensorGpu<'a, f16, ReadWrite>,
    time_mix_v: TensorGpu<'a, f16, ReadWrite>,
    time_mix_r: TensorGpu<'a, f16, ReadWrite>,

    w_k: Matrix<'a>,
    w_v: Matrix<'a>,
    w_r: Matrix<'a>,
    w_o: Matrix<'a>,
}

struct Ffn<'a> {
    time_mix_k: TensorGpu<'a, f16, ReadWrite>,
    time_mix_v: TensorGpu<'a, f16, ReadWrite>,

    w_k: Matrix<'a>,
    w_v: Matrix<'a>,
    w_r: Matrix<'a>,
}

struct Layer<'a> {
    att_layer_norm: LayerNorm<'a>,
    ffn_layer_norm: LayerNorm<'a>,
    att: Att<'a>,
    ffn: Ffn<'a>,
}

struct Embed<'a> {
    layer_norm: LayerNorm<'a>,
    w: TensorCpu<'a, f16, ReadWrite>,
}

struct Head<'a> {
    layer_norm: LayerNorm<'a>,
    w: Vec<TensorGpu<'a, f16, ReadWrite>>,
}

/// Runtime buffers.
pub struct ModelBuffer<'a> {
    info: ModelInfo,

    mask: TensorGpu<'a, f32, Uniform>,
    emb_x: TensorGpu<'a, f32, ReadWrite>,

    att_x: TensorGpu<'a, f32, ReadWrite>,
    att_kx: TensorGpu<'a, f32, ReadWrite>,
    att_vx: TensorGpu<'a, f32, ReadWrite>,
    att_k: TensorGpu<'a, f32, ReadWrite>,
    att_v: TensorGpu<'a, f32, ReadWrite>,
    att_r: TensorGpu<'a, f32, ReadWrite>,
    att_w: TensorGpu<'a, f32, ReadWrite>,
    att_o: TensorGpu<'a, f32, ReadWrite>,

    ffn_x: TensorGpu<'a, f32, ReadWrite>,
    ffn_kx: TensorGpu<'a, f32, ReadWrite>,
    ffn_rx: TensorGpu<'a, f32, ReadWrite>,
    ffn_k: TensorGpu<'a, f32, ReadWrite>,
    ffn_v: TensorGpu<'a, f32, ReadWrite>,
    ffn_r: TensorGpu<'a, f32, ReadWrite>,
    ffn_o: TensorGpu<'a, f32, ReadWrite>,

    head_x: TensorGpu<'a, f32, ReadWrite>,
    head_v: Vec<TensorGpu<'a, f32, ReadWrite>>,
    head_o: TensorGpu<'a, f32, ReadWrite>,

    softmax_o: TensorGpu<'a, f32, ReadWrite>,

    map: TensorGpu<'a, f32, ReadBack>,
}

impl<'a> ModelBuffer<'a> {
    pub fn new(
        context: &'a Context,
        info: ModelInfo,
        input: TensorCpu<'a, f32, ReadWrite>,
        mask: TensorCpu<'a, f32, Uniform>,
    ) -> Result<Self, TensorError> {
        let shape = input.shape();
        let ffn_shape = Shape::new(shape[0] * 4, shape[1], shape[2]);
        let head_shape = Shape::new(info.max_head_chunk, shape[1], shape[2]);
        let out_shape = Shape::new(info.num_vocab, shape[1], shape[2]);

        input.check_shape(Shape::new(info.num_emb, shape[1], shape[2]))?;

        let mask = TensorGpu::from(mask);
        let emb_x = TensorGpu::from(input);
        let att_x = emb_x.clone();

        let att_o: TensorGpu<_, _> = context.init_tensor(shape);
        let ffn_x = att_o.clone();

        let ffn_o: TensorGpu<_, _> = context.init_tensor(shape);
        let head_x = ffn_o.clone();

        let head_v = (0..(info.num_vocab + info.max_head_chunk - 1) / info.max_head_chunk)
            .map(|_| context.init_tensor(head_shape))
            .collect();

        Ok(Self {
            info,
            mask,
            emb_x,
            att_x,
            att_kx: context.init_tensor(shape),
            att_vx: context.init_tensor(shape),
            att_k: context.init_tensor(shape),
            att_v: context.init_tensor(shape),
            att_r: context.init_tensor(shape),
            att_w: context.init_tensor(shape),
            att_o,
            ffn_x,
            ffn_kx: context.init_tensor(shape),
            ffn_rx: context.init_tensor(shape),
            ffn_k: context.init_tensor(ffn_shape),
            ffn_v: context.init_tensor(shape),
            ffn_r: context.init_tensor(shape),
            ffn_o,
            head_x,
            head_v,
            head_o: context.init_tensor(out_shape),
            softmax_o: context.init_tensor(out_shape),
            map: context.init_tensor(out_shape),
        })
    }

    pub fn num_tokens(&self) -> usize {
        self.emb_x.shape()[1]
    }

    pub fn num_batches(&self) -> usize {
        self.emb_x.shape()[2]
    }
}
