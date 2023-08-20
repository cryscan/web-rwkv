use bitflags::bitflags;
use derive_getters::Getters;
use half::f16;
use wgpu::CommandEncoderDescriptor;

use crate::{
    context::Context,
    tensor::{
        ReadBack, ReadWrite, Shape, TensorCommand, TensorCpu, TensorError, TensorGpu, Uniform,
    },
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
    input: TensorGpu<'a, f32, ReadWrite>,

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

    head_x: Vec<TensorGpu<'a, f32, ReadWrite>>,
    head_o: TensorGpu<'a, f32, ReadWrite>,

    softmax: TensorGpu<'a, f32, ReadWrite>,

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

        let head_x = (0..(info.num_vocab + info.max_head_chunk - 1) / info.max_head_chunk)
            .map(|_| context.init_tensor(head_shape))
            .collect();

        Ok(Self {
            info,
            mask: TensorGpu::from(mask),
            input: TensorGpu::from(input),
            att_x: context.init_tensor(shape),
            att_kx: context.init_tensor(shape),
            att_vx: context.init_tensor(shape),
            att_k: context.init_tensor(shape),
            att_v: context.init_tensor(shape),
            att_r: context.init_tensor(shape),
            att_w: context.init_tensor(shape),
            att_o: context.init_tensor(shape),
            ffn_x: context.init_tensor(shape),
            ffn_kx: context.init_tensor(shape),
            ffn_rx: context.init_tensor(shape),
            ffn_k: context.init_tensor(ffn_shape),
            ffn_v: context.init_tensor(shape),
            ffn_r: context.init_tensor(shape),
            ffn_o: context.init_tensor(shape),
            head_x,
            head_o: context.init_tensor(out_shape),
            softmax: context.init_tensor(out_shape),
            map: context.init_tensor(out_shape),
        })
    }

    pub fn num_tokens(&self) -> usize {
        self.input.shape()[1]
    }

    pub fn num_batches(&self) -> usize {
        self.input.shape()[2]
    }
}

pub struct ModelState<'a> {
    pub info: ModelInfo,
    pub context: &'a Context,
    pub layers: Vec<(TensorGpu<'a, f32, ReadWrite>, TensorGpu<'a, f32, ReadWrite>)>,
}

impl<'a> ModelState<'a> {
    pub fn new(context: &'a Context, info: ModelInfo, num_batches: usize) -> Self {
        let layers = (0..info.num_layers)
            .map(|_| {
                let data = [
                    vec![0.0; info.num_emb],
                    vec![0.0; info.num_emb],
                    vec![0.0; info.num_emb],
                    vec![-f32::MIN; info.num_emb],
                ]
                .concat();
                let att = context
                    .tensor_from_data(Shape::new(info.num_emb, 4, num_batches), data)
                    .unwrap();
                let ffn = context.zeros(Shape::new(info.num_emb, 1, num_batches));
                (att, ffn)
            })
            .collect();
        Self {
            info,
            context,
            layers,
        }
    }

    pub fn att_shape(&self) -> Shape {
        self.layers[0].0.shape()
    }

    pub fn ffn_shape(&self) -> Shape {
        self.layers[0].1.shape()
    }
}

pub struct BackedState<'a> {
    pub info: ModelInfo,
    pub context: &'a Context,
    pub data: Vec<f32>,
}

impl<'a> From<BackedState<'a>> for ModelState<'a> {
    fn from(value: BackedState<'a>) -> Self {
        let BackedState {
            info,
            context,
            data,
        } = value;

        let layer_stride = info.num_emb * 5;
        let batch_stride = info.num_layers * layer_stride;
        let num_batch = data.len() / batch_stride;

        let att_shape = Shape::new(info.num_emb, 4, num_batch);
        let ffn_shape = Shape::new(info.num_emb, 1, num_batch);

        let mut layers = vec![];
        for batch in 0..num_batch {
            for layer in 0..info.num_layers {
                let index = batch * batch_stride + layer * layer_stride;
                let att = data[index..index + 4 * info.num_emb].to_owned();
                let ffn = data[index + 4 * info.num_emb..index + 5 * info.num_emb].to_owned();
                layers.push((
                    context.tensor_from_data(att_shape, att).unwrap(),
                    context.tensor_from_data(ffn_shape, ffn).unwrap(),
                ));
            }
        }

        Self {
            info,
            context,
            layers,
        }
    }
}

impl<'a> From<ModelState<'a>> for BackedState<'a> {
    fn from(value: ModelState<'a>) -> Self {
        let att_shape = value.att_shape();
        let ffn_shape = value.ffn_shape();

        let ModelState {
            info,
            context,
            layers,
        } = value;

        let att_map = context.init_tensor(att_shape);
        let ffn_map = context.init_tensor(ffn_shape);

        let data = layers
            .into_iter()
            .map(|(att, ffn)| {
                let mut encoder = context
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                encoder.copy_tensor(&att, &att_map).unwrap();
                encoder.copy_tensor(&ffn, &ffn_map).unwrap();
                context.queue.submit(Some(encoder.finish()));

                let att = Vec::from(TensorCpu::from(att_map.clone()));
                let ffn = Vec::from(TensorCpu::from(ffn_map.clone()));
                [att, ffn].concat()
            })
            .collect::<Vec<_>>()
            .concat();

        Self {
            info,
            context,
            data,
        }
    }
}
