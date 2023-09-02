use std::sync::Arc;

use anyhow::Result;
use bitflags::bitflags;
use derive_getters::Getters;
use half::f16;
use itertools::Itertools;
use safetensors::SafeTensors;
use web_rwkv_derive::{Deref, DerefMut};
use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor};

use crate::{
    context::Context,
    tensor::{
        cache::ResourceCache,
        ops::{TensorCommand, TensorOp, TensorPass},
        shape::{Shape, TensorDimension},
        IntoPackedCursors, ReadBack, ReadWrite, TensorCpu, TensorError, TensorExt, TensorGpu,
        TensorInit, TensorStack, TensorView,
    },
};

#[derive(Debug, Getters)]
pub struct Model<'a, 'b> {
    pub context: &'a Context,

    info: ModelInfo,
    quant: Quantization,

    /// The head matrix is too big for a storage buffer so it's divided into chunks.
    head_chunk_size: usize,
    /// To prevent the GPU device from lost, this limits the maximum batch-token it processes one time.
    token_chunk_size: usize,

    #[getter(skip)]
    tensor: ModelTensor<'a, 'b>,
    #[getter(skip)]
    buffer_cache: ResourceCache<(usize, usize), ModelBuffer<'a>>,
    #[getter(skip)]
    softmax_cache: ResourceCache<usize, SoftmaxBuffer<'a>>,
    #[getter(skip)]
    stack_cache: ResourceCache<usize, TensorGpu<'a, u32, ReadWrite>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub num_emb: usize,
    pub num_vocab: usize,
}

bitflags! {
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct LayerFlags: u64 {}
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

#[derive(Debug)]
enum Matrix<'a> {
    Fp16(TensorGpu<'a, f16, ReadWrite>),
    Int8 {
        w: Box<TensorGpu<'a, u8, ReadWrite>>,
        mx: Box<TensorGpu<'a, f32, ReadWrite>>,
        rx: Box<TensorGpu<'a, f32, ReadWrite>>,
        my: Box<TensorGpu<'a, f32, ReadWrite>>,
        ry: Box<TensorGpu<'a, f32, ReadWrite>>,
    },
}

impl<'a, 'b> Matrix<'a> {
    pub fn matmul_op(
        &'b self,
        input: TensorView<'a, f32>,
        output: TensorView<'a, f32>,
    ) -> Result<TensorOp<'b>, TensorError> {
        match self {
            Matrix::Fp16(matrix) => TensorOp::matmul(matrix, input, output),
            Matrix::Int8 { w, mx, rx, my, ry } => {
                TensorOp::matmul_int8(w, mx, rx, my, ry, input, output)
            }
        }
    }
}

#[derive(Debug)]
struct ModelTensor<'a, 'b> {
    embed: Embed<'a, 'b>,
    head: Head<'a>,
    layers: Vec<Layer<'a>>,
}

#[derive(Debug)]
struct LayerNorm<'a> {
    w: TensorGpu<'a, f16, ReadWrite>,
    b: TensorGpu<'a, f16, ReadWrite>,
}

#[derive(Debug)]
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

#[derive(Debug)]
struct Ffn<'a> {
    time_mix_k: TensorGpu<'a, f16, ReadWrite>,
    time_mix_r: TensorGpu<'a, f16, ReadWrite>,

    w_k: Matrix<'a>,
    w_v: Matrix<'a>,
    w_r: Matrix<'a>,
}

#[derive(Debug)]
struct Layer<'a> {
    att_layer_norm: LayerNorm<'a>,
    ffn_layer_norm: LayerNorm<'a>,
    att: Att<'a>,
    ffn: Ffn<'a>,
}

#[derive(Debug)]
struct Embed<'a, 'b> {
    layer_norm: LayerNorm<'a>,
    w: TensorCpu<'a, 'b, f16>,
}

#[derive(Debug)]
struct Head<'a> {
    layer_norm: LayerNorm<'a>,
    w: Vec<TensorGpu<'a, f16, ReadWrite>>,
}

/// Runtime buffers.
#[derive(Debug)]
pub struct ModelBuffer<'a> {
    cursors: TensorGpu<'a, u32, ReadWrite>,
    input: TensorGpu<'a, f32, ReadWrite>,

    att_x: TensorGpu<'a, f32, ReadWrite>,
    att_kx: TensorGpu<'a, f32, ReadWrite>,
    att_vx: TensorGpu<'a, f32, ReadWrite>,
    att_rx: TensorGpu<'a, f32, ReadWrite>,
    att_k: TensorGpu<'a, f32, ReadWrite>,
    att_v: TensorGpu<'a, f32, ReadWrite>,
    att_r: TensorGpu<'a, f32, ReadWrite>,
    att_o: TensorGpu<'a, f32, ReadWrite>,

    ffn_x: TensorGpu<'a, f32, ReadWrite>,
    ffn_kx: TensorGpu<'a, f32, ReadWrite>,
    ffn_rx: TensorGpu<'a, f32, ReadWrite>,
    ffn_k: TensorGpu<'a, f32, ReadWrite>,
    ffn_v: TensorGpu<'a, f32, ReadWrite>,
    ffn_r: TensorGpu<'a, f32, ReadWrite>,

    head_x: TensorGpu<'a, f32, ReadWrite>,
    head_o: TensorGpu<'a, f32, ReadWrite>,

    map: TensorGpu<'a, f32, ReadBack>,
}

impl<'a> ModelBuffer<'a> {
    pub fn new(context: &'a Context, info: &ModelInfo, max_batch: usize, num_token: usize) -> Self {
        let shape = Shape::new(info.num_emb, num_token, 1);
        let cursors_shape = Shape::new(num_token, 1, 1);
        let hidden_shape = Shape::new(info.num_emb << 2, num_token, 1);
        let head_shape = Shape::new(info.num_emb, 1, max_batch);
        let output_shape = Shape::new(info.num_vocab, 1, max_batch);

        Self {
            cursors: context.init_tensor(cursors_shape),
            input: context.init_tensor(shape),
            att_x: context.init_tensor(shape),
            att_kx: context.init_tensor(shape),
            att_vx: context.init_tensor(shape),
            att_rx: context.init_tensor(shape),
            att_k: context.init_tensor(shape),
            att_v: context.init_tensor(shape),
            att_r: context.init_tensor(shape),
            att_o: context.init_tensor(shape),
            ffn_x: context.init_tensor(shape),
            ffn_kx: context.init_tensor(shape),
            ffn_rx: context.init_tensor(shape),
            ffn_k: context.init_tensor(hidden_shape),
            ffn_v: context.init_tensor(shape),
            ffn_r: context.init_tensor(shape),
            head_x: context.init_tensor(head_shape),
            head_o: context.init_tensor(output_shape),
            map: context.init_tensor(output_shape),
        }
    }

    #[inline]
    pub fn max_batch(&self) -> usize {
        self.map.shape()[2]
    }

    #[inline]
    pub fn num_token(&self) -> usize {
        self.cursors.shape()[1]
    }
}

#[derive(Debug)]
pub struct SoftmaxBuffer<'a> {
    softmax: TensorGpu<'a, f32, ReadWrite>,
    map: TensorGpu<'a, f32, ReadBack>,
}

impl<'a> SoftmaxBuffer<'a> {
    pub fn new(context: &'a Context, info: &ModelInfo, max_batch: usize) -> Self {
        let shape = Shape::new(info.num_vocab, 1, max_batch);
        Self {
            softmax: context.init_tensor(shape),
            map: context.init_tensor(shape),
        }
    }
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct ModelState<'a>(pub TensorGpu<'a, f32, ReadWrite>);

impl<'a> ModelState<'a> {
    pub fn new(context: &'a Context, info: &ModelInfo, num_batches: usize) -> Self {
        let data = (0..num_batches)
            .map(|_| {
                (0..info.num_layers)
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
                Shape::new(info.num_emb, 5 * info.num_layers, num_batches),
                data,
            )
            .unwrap();
        Self(state)
    }

    pub fn load(&self, backed: &BackedState) -> Result<(), TensorError> {
        let host = TensorCpu::from_data(self.context, self.shape(), &backed.data)?;
        self.0.load(&host)
    }

    pub fn load_batch(&self, backed: &BackedState, batch: usize) -> Result<(), TensorError> {
        let shape = self.shape();
        let shape = Shape::new(shape[0], shape[1], 1);
        let host = TensorCpu::from_data(self.context, shape, &backed.data)?;
        self.0.load_batch(&host, batch)
    }

    pub fn back(&self) -> BackedState {
        let context = self.context;
        let shape = self.shape();
        let map = TensorGpu::init(self.context, shape);

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_tensor(self, &map).expect("back entire state");
        context.queue.submit(Some(encoder.finish()));

        let host = TensorCpu::from(map);
        BackedState {
            shape,
            data: host.to_vec(),
        }
    }

    pub fn back_batch(&self, batch: usize) -> Result<BackedState, TensorError> {
        let context = self.context;
        let shape = self.shape();
        let shape = Shape::new(shape[0], shape[1], 1);
        let map = TensorGpu::init(self.context, shape);

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_tensor_batch(self, &map, batch)?;
        context.queue.submit(Some(encoder.finish()));

        let host = TensorCpu::from(map);
        Ok(BackedState {
            shape,
            data: host.to_vec(),
        })
    }

    #[inline]
    pub fn max_batch(&self) -> usize {
        self.0.shape()[2]
    }

    fn att(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let start = 5 * layer;
        let end = start + 4;
        self.as_view((.., start..end, ..))
    }

    fn ffn(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let start = 5 * layer + 4;
        self.as_view((.., start..=start, ..))
    }
}

#[derive(Debug, Clone)]
pub struct BackedState {
    pub shape: Shape,
    pub data: Vec<f32>,
}

// impl<'a, 'b> BackedState<'a, 'b> {
//     pub fn repeat(self, repeat: usize) -> Self {
//         let state = self.0.repeat(2, repeat);
//         Self(state)
//     }

//     pub fn take(self, batch: usize) -> Result<Self, TensorError> {
//         let state = self.0.into_slice((.., .., batch))?;
//         Ok(Self(state))
//     }

//     pub fn split(self) -> Vec<Self> {
//         self.0
//             .split(2)
//             .expect("split backed state")
//             .into_iter()
//             .map(Self)
//             .collect()
//     }

//     pub fn stack(batches: Vec<Self>) -> Result<Self, TensorError> {
//         if batches.is_empty() {
//             return Err(TensorError::Empty);
//         }
//         let states: Vec<_> = batches.into_iter().map(|batch| batch.0).collect();
//         Ok(Self(TensorCpu::stack(states)?))
//     }
// }

// impl<'a> From<ModelState<'a>> for BackedState<'a, '_> {
//     fn from(value: ModelState<'a>) -> Self {
//         let context = value.context;
//         let map = context.init_tensor(value.shape());
//         let mut encoder = context
//             .device
//             .create_command_encoder(&CommandEncoderDescriptor::default());
//         encoder.copy_tensor(&value, &map).unwrap();
//         context.queue.submit(Some(encoder.finish()));

//         let state = TensorCpu::from(map);
//         Self(state)
//     }
// }

// impl<'a> From<BackedState<'a, '_>> for ModelState<'a> {
//     fn from(value: BackedState<'a, '_>) -> Self {
//         let state = TensorGpu::from(value.0);
//         Self(state)
//     }
// }

pub struct ModelBuilder<'a, 'b> {
    context: &'a Context,
    data: &'b [u8],
    quant: Quantization,
    head_chunk_size: usize,
    token_chunk_size: usize,
}

impl<'a, 'b> ModelBuilder<'a, 'b> {
    pub fn new(context: &'a Context, data: &'b [u8]) -> Self {
        Self {
            context,
            data,
            quant: Quantization::None,
            head_chunk_size: 4096,
            token_chunk_size: 32,
        }
    }

    pub fn with_quant(self, quant: Quantization) -> Self {
        Self { quant, ..self }
    }

    pub fn with_head_chunk_size(self, size: usize) -> Self {
        Self {
            head_chunk_size: size,
            ..self
        }
    }

    pub fn with_token_chunk_size(self, size: usize) -> Self {
        Self {
            token_chunk_size: size,
            ..self
        }
    }

    pub fn build<'c>(self) -> Result<Model<'a, 'c>> {
        let Self {
            context,
            data,
            quant,
            head_chunk_size,
            token_chunk_size,
        } = self;

        let model = SafeTensors::deserialize(data)?;
        let embed = model.tensor("emb.weight")?;

        let num_layers = {
            let mut r: usize = 0;
            for i in model.names() {
                const PREFIX: &str = "blocks.";
                if let Some(i) = i.strip_prefix(PREFIX) {
                    let i = &i[..i.find('.').unwrap_or(0)];
                    r = r.max(i.parse::<usize>()?)
                }
            }
            r + 1
        };
        let (num_emb, num_vocab) = {
            let num_emb = embed.shape()[1];
            let num_vocab = embed.shape()[0];
            (num_emb, num_vocab)
        };

        let info = ModelInfo {
            num_layers,
            num_emb,
            num_vocab,
        };

        let load_vector_f32 = |name: String| -> Result<TensorGpu<f32, ReadWrite>> {
            let tensor = model.tensor(&name)?;
            let data = bytemuck::pod_collect_to_vec::<_, f16>(tensor.data())
                .into_iter()
                .map(f16::to_f32)
                .collect_vec();
            let shape = Shape::new(data.len(), 1, 1);
            Ok(context.tensor_from_data(shape, data)?)
        };
        let load_vector_exp_f32 = |name: String| -> Result<TensorGpu<f32, ReadWrite>> {
            let tensor = model.tensor(&name)?;
            let data = bytemuck::pod_collect_to_vec::<_, f16>(tensor.data())
                .into_iter()
                .map(f16::to_f32)
                .map(|x| -x.exp())
                .collect_vec();
            let shape = Shape::new(data.len(), 1, 1);
            Ok(context.tensor_from_data(shape, data)?)
        };
        let load_vector_f16 = |name: String| -> Result<TensorGpu<f16, ReadWrite>> {
            let tensor = model.tensor(&name)?;
            let data = bytemuck::cast_slice(tensor.data());
            let shape = Shape::new(data.len(), 1, 1);
            Ok(context.tensor_from_data(shape, data)?)
        };
        let load_matrix_f16 = |name: String| -> Result<Matrix<'_>> {
            let tensor = model.tensor(&name)?;
            let shape = tensor.shape();
            let shape = Shape::new(shape[1], shape[0], 1);
            let w = context.tensor_from_data(shape, bytemuck::cast_slice(tensor.data()))?;
            Ok(Matrix::Fp16(w))
        };
        let load_matrix_u8 = |name: String| -> Result<Matrix<'a>> {
            let tensor = model.tensor(&name)?;
            let shape = tensor.shape();
            let shape = Shape::new(shape[1], shape[0], 1);
            let matrix: TensorGpu<f16, _> =
                context.tensor_from_data(shape, bytemuck::cast_slice(tensor.data()))?;
            let shape = matrix.shape();

            // let mx_f32 = context.init_tensor(Shape::new(shape[0], 1, 1));
            // let rx_f32 = context.init_tensor(Shape::new(shape[0], 1, 1));
            // let my_f32 = context.init_tensor(Shape::new(shape[1], 1, 1));
            // let ry_f32 = context.init_tensor(Shape::new(shape[1], 1, 1));

            let w = Box::new(context.init_tensor(matrix.shape()));

            let mx = Box::new(context.init_tensor(Shape::new(shape[0], 1, 1)));
            let rx = Box::new(context.init_tensor(Shape::new(shape[0], 1, 1)));
            let my = Box::new(context.init_tensor(Shape::new(shape[1], 1, 1)));
            let ry = Box::new(context.init_tensor(Shape::new(shape[1], 1, 1)));

            let ops = TensorOp::quantize_mat_int8(&matrix, &mx, &rx, &my, &ry, &w)?;

            // ops.push(TensorOp::quantize_vec_fp16(&mx_f32, &mx)?);
            // ops.push(TensorOp::quantize_vec_fp16(&rx_f32, &rx)?);
            // ops.push(TensorOp::quantize_vec_fp16(&my_f32, &my)?);
            // ops.push(TensorOp::quantize_vec_fp16(&ry_f32, &ry)?);

            let mut encoder = context
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            ops.iter().for_each(|op| pass.execute_tensor_op(op));
            drop(pass);

            context.queue.submit(Some(encoder.finish()));

            Ok(Matrix::Int8 { w, mx, rx, my, ry })
        };

        let embed = Embed {
            layer_norm: LayerNorm {
                w: load_vector_f16("blocks.0.ln0.weight".into())?,
                b: load_vector_f16("blocks.0.ln0.bias".into())?,
            },
            w: context.tensor_from_data(
                Shape::new(num_emb, num_vocab, 1),
                bytemuck::pod_collect_to_vec(embed.data()),
            )?,
        };

        let head = {
            let tensor = model.tensor("head.weight")?;
            let shape = tensor.shape();
            let shape = Shape::new(shape[1], shape[0], 1);
            let chunks = shape[1] / head_chunk_size;
            let data = bytemuck::cast_slice(tensor.data());

            let w = (0..chunks)
                .map(|chunk| {
                    let start = (chunk * head_chunk_size) * shape[0];
                    let end = start + head_chunk_size * shape[0];
                    context.tensor_from_data(
                        Shape::new(shape[0], head_chunk_size, 1),
                        &data[start..end],
                    )
                })
                .try_collect()?;

            Head {
                layer_norm: LayerNorm {
                    w: load_vector_f16("ln_out.weight".into())?,
                    b: load_vector_f16("ln_out.bias".into())?,
                },
                w,
            }
        };

        context.queue.submit(None);
        context.device.poll(wgpu::MaintainBase::Wait);

        let layers = (0..num_layers)
            .map(|layer| {
                let att_layer_norm = LayerNorm {
                    w: load_vector_f16(format!("blocks.{layer}.ln1.weight"))?,
                    b: load_vector_f16(format!("blocks.{layer}.ln1.bias"))?,
                };

                let att = format!("blocks.{layer}.att");
                let time_decay = load_vector_exp_f32(format!("{att}.time_decay"))?;
                let time_first = load_vector_f32(format!("{att}.time_first"))?;
                let time_mix_k = load_vector_f16(format!("{att}.time_mix_k"))?;
                let time_mix_v = load_vector_f16(format!("{att}.time_mix_v"))?;
                let time_mix_r = load_vector_f16(format!("{att}.time_mix_r"))?;

                let att = match quant {
                    Quantization::Int8(x) if x.contains_layer(layer as u64) => Att {
                        time_decay,
                        time_first,
                        time_mix_k,
                        time_mix_v,
                        time_mix_r,
                        w_k: load_matrix_u8(format!("{att}.key.weight"))?,
                        w_v: load_matrix_u8(format!("{att}.value.weight"))?,
                        w_r: load_matrix_u8(format!("{att}.receptance.weight"))?,
                        w_o: load_matrix_u8(format!("{att}.output.weight"))?,
                    },
                    _ => Att {
                        time_decay,
                        time_first,
                        time_mix_k,
                        time_mix_v,
                        time_mix_r,
                        w_k: load_matrix_f16(format!("{att}.key.weight"))?,
                        w_v: load_matrix_f16(format!("{att}.value.weight"))?,
                        w_r: load_matrix_f16(format!("{att}.receptance.weight"))?,
                        w_o: load_matrix_f16(format!("{att}.output.weight"))?,
                    },
                };

                let ffn_layer_norm = LayerNorm {
                    w: load_vector_f16(format!("blocks.{layer}.ln2.weight"))?,
                    b: load_vector_f16(format!("blocks.{layer}.ln2.bias"))?,
                };

                let ffn = format!("blocks.{layer}.ffn");
                let time_mix_k = load_vector_f16(format!("{ffn}.time_mix_k"))?;
                let time_mix_r = load_vector_f16(format!("{ffn}.time_mix_k"))?;

                let ffn = match quant {
                    Quantization::Int8(x) if x.contains_layer(layer as u64) => Ffn {
                        time_mix_k,
                        time_mix_r,
                        w_k: load_matrix_u8(format!("{ffn}.key.weight"))?,
                        w_v: load_matrix_u8(format!("{ffn}.value.weight"))?,
                        w_r: load_matrix_u8(format!("{ffn}.receptance.weight"))?,
                    },
                    _ => Ffn {
                        time_mix_k,
                        time_mix_r,
                        w_k: load_matrix_f16(format!("{ffn}.key.weight"))?,
                        w_v: load_matrix_f16(format!("{ffn}.value.weight"))?,
                        w_r: load_matrix_f16(format!("{ffn}.receptance.weight"))?,
                    },
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
        Ok(Model {
            context,
            info,
            quant,
            head_chunk_size,
            token_chunk_size,
            tensor,
            buffer_cache: ResourceCache::new(2),
            softmax_cache: ResourceCache::new(2),
            stack_cache: Default::default(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelError {
    BatchSize(usize, usize),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::BatchSize(batch, max) => {
                write!(f, "input batch size {batch} not match {max}")
            }
        }
    }
}

impl std::error::Error for ModelError {}

impl<'a, 'b> Model<'a, 'b> {
    #[inline]
    fn request_buffer(&self, max_batch: usize, num_token: usize) -> Arc<ModelBuffer> {
        self.buffer_cache.request((max_batch, num_token), || {
            ModelBuffer::new(self.context, &self.info, max_batch, num_token)
        })
    }

    fn request_softmax(&self, max_batch: usize) -> Arc<SoftmaxBuffer> {
        self.softmax_cache.request(max_batch, || {
            SoftmaxBuffer::new(self.context, &self.info, max_batch)
        })
    }

    #[inline]
    fn request_stack(&self, num_batch: usize) -> Arc<TensorGpu<u32, ReadWrite>> {
        self.stack_cache.request(num_batch, || {
            self.context.zeros(Shape::new(num_batch, 1, 1))
        })
    }

    #[inline]
    pub fn head_shape(&self, num_batch: usize) -> Shape {
        Shape::new(self.info.num_vocab, 1, num_batch)
    }

    /// Softmax of the input tensors.
    pub async fn softmax(&'a self, input: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        let max_batch = input.len();

        let mut redirect = vec![None; max_batch];
        let input: TensorCpu<_> = input
            .into_iter()
            .enumerate()
            .filter_map(|(index, data)| {
                TensorCpu::from_data(self.context, self.head_shape(1), data)
                    .map(|x| (index, x))
                    .ok()
            })
            .enumerate()
            .map(|(filtered_index, (index, x))| {
                redirect[index] = Some(filtered_index);
                x
            })
            .collect_vec()
            .try_into()?;

        let num_batch = input.shape()[2];
        let buffer = self.request_softmax(num_batch);
        buffer.softmax.load(&input)?;

        let op = TensorOp::softmax(&buffer.softmax)?;

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        encoder.copy_tensor(&buffer.softmax, &buffer.map)?;
        self.context.queue.submit(Some(encoder.finish()));

        let mut output = async {
            TensorCpu::from(buffer.map.clone())
                .split(2)
                .expect("split buffer map")
                .into_iter()
                .map(|tensor| tensor.to_vec())
                .collect_vec()
        }
        .await;

        let mut probs = vec![vec![]; max_batch];
        for (probs, redirect) in probs.iter_mut().zip_eq(redirect.into_iter()) {
            if let Some(redirect) = redirect {
                std::mem::swap(probs, &mut output[redirect]);
            }
        }

        Ok(probs)
    }

    /// Run the model for a batch of tokens as input.
    /// The length of `tokens` must match the number of batches in `state`.
    /// `tokens` may have slots with no tokens, for which `run` won't compute that batch and will return an empty vector in that corresponding slot.
    pub async fn run(
        &'a self,
        tokens: &mut Vec<Vec<u16>>,
        state: &ModelState<'a>,
    ) -> Result<Vec<Vec<f32>>> {
        let num_token: usize = tokens.iter().map(Vec::len).sum();
        let max_batch = state.shape()[2];

        if tokens.len() != max_batch {
            return Err(ModelError::BatchSize(tokens.len(), max_batch).into());
        }
        if num_token == 0 {
            return Ok(vec![vec![]; max_batch]);
        }

        // we only infer at most `token_chunk_size` tokens at a time
        let mut num_token = num_token.min(self.token_chunk_size);
        let mut inputs = vec![vec![]; max_batch];
        let mut last_batch = None;

        // take `num_token` tokens out of all the inputs and put into `input`
        for (index, (batch, input)) in tokens.iter_mut().zip(inputs.iter_mut()).enumerate() {
            let mid = batch.len().min(num_token);
            num_token -= mid;

            let (head, tail) = batch.split_at(mid);
            last_batch = (!tail.is_empty()).then_some(index);
            *input = head.to_vec();
            *batch = tail.to_vec();

            if num_token == 0 {
                break;
            }
        }

        let (buffer, redirect) = self.run_internal(inputs, state, last_batch)?;
        let output = async { TensorCpu::from(buffer.map.clone()) }.await;

        Ok(redirect
            .into_iter()
            .map(|index| match index {
                Some(index) => output
                    .as_slice((.., .., index))
                    .expect("this never happens")
                    .to_vec(),
                None => vec![],
            })
            .collect())
    }

    fn run_internal(
        &self,
        tokens: Vec<Vec<u16>>,
        state: &ModelState,
        last_batch: Option<usize>,
    ) -> Result<(Arc<ModelBuffer>, Vec<Option<usize>>)> {
        let context = self.context;
        let tensor = &self.tensor;

        let input: Vec<_> = tokens
            .into_iter()
            .map(|tokens| -> Result<_, TensorError> {
                let stack = TensorCpu::stack(
                    tokens
                        .into_iter()
                        .map(|token| tensor.embed.w.as_slice((.., token as usize, ..)))
                        .try_collect()?,
                )
                .unwrap_or_else(|_| context.zeros(Shape::new(self.info.num_emb, 1, 0)));
                stack.map(|x| x.to_f32()).reshape(
                    TensorDimension::Full,
                    TensorDimension::Auto,
                    TensorDimension::Dimension(1),
                )
            })
            .try_collect()?;

        let input = TensorStack::try_from(input)?;
        let max_batch = input.max_batch();
        let num_batch = input.num_batch();
        let num_token = input.num_token();

        let buffer = self.request_buffer(max_batch, num_token);
        let stack = self.request_stack(num_batch);

        // collect batch output copy commands for later
        let mut redirect = vec![None; max_batch];
        let head_ops: Vec<_> = input
            .cursors
            .iter()
            .filter(|cursor| cursor.len > 0)
            .filter(|cursor| !last_batch.is_some_and(|index| cursor.batch == index))
            .enumerate()
            .map(|(index, cursor)| -> Result<TensorOp<'_>, TensorError> {
                redirect[cursor.batch] = Some(index);
                let token = cursor.token + cursor.len - 1;
                let input = buffer.ffn_x.as_view((.., token, ..))?;
                let output = buffer.head_x.as_view((.., .., index))?;
                TensorOp::blit(input, output)
            })
            .try_collect()?;

        let stack_host =
            TensorCpu::from_data(context, stack.shape(), input.cursors.clone().into_stack())?;
        let cursors = TensorCpu::from_data(
            context,
            buffer.cursors.shape(),
            input.cursors.into_cursors(),
        )?;

        stack.load(&stack_host)?;
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

            let ops = vec![
                TensorOp::layer_norm(
                    &layer.att_layer_norm.w,
                    &layer.att_layer_norm.b,
                    &buffer.att_x,
                )?,
                TensorOp::token_shift_stack(
                    &buffer.cursors,
                    &layer.att.time_mix_k,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_kx,
                )?,
                TensorOp::token_shift_stack(
                    &buffer.cursors,
                    &layer.att.time_mix_v,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_vx,
                )?,
                TensorOp::token_shift_stack(
                    &buffer.cursors,
                    &layer.att.time_mix_r,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_rx,
                )?,
                layer
                    .att
                    .w_k
                    .matmul_op(buffer.att_kx.clone().into(), buffer.att_k.clone().into())?,
                layer
                    .att
                    .w_v
                    .matmul_op(buffer.att_vx.clone().into(), buffer.att_v.clone().into())?,
                layer
                    .att
                    .w_r
                    .matmul_op(buffer.att_rx.clone().into(), buffer.att_r.clone().into())?,
                TensorOp::token_mix_stack(
                    &stack,
                    &layer.att.time_decay,
                    &layer.att.time_first,
                    &buffer.att_k,
                    &buffer.att_v,
                    &buffer.att_r,
                    &buffer.att_x,
                    state.att(index)?,
                )?,
                layer
                    .att
                    .w_o
                    .matmul_op(buffer.att_x.clone().into(), buffer.att_o.clone().into())?,
                TensorOp::add(&buffer.input, &buffer.att_o)?,
            ];

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            ops.iter().for_each(|op| pass.execute_tensor_op(op));
            drop(pass);

            encoder.copy_tensor(&buffer.att_o, &buffer.ffn_x)?;

            let ops = vec![
                TensorOp::layer_norm(
                    &layer.ffn_layer_norm.w,
                    &layer.ffn_layer_norm.b,
                    &buffer.ffn_x,
                )?,
                TensorOp::token_shift_stack(
                    &buffer.cursors,
                    &layer.ffn.time_mix_k,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                    &buffer.ffn_kx,
                )?,
                TensorOp::token_shift_stack(
                    &buffer.cursors,
                    &layer.ffn.time_mix_r,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                    &buffer.ffn_rx,
                )?,
                layer
                    .ffn
                    .w_k
                    .matmul_op(buffer.ffn_kx.clone().into(), buffer.ffn_k.clone().into())?,
                TensorOp::squared_relu(&buffer.ffn_k)?,
                layer
                    .ffn
                    .w_v
                    .matmul_op(buffer.ffn_k.clone().into(), buffer.ffn_v.clone().into())?,
                layer
                    .ffn
                    .w_r
                    .matmul_op(buffer.ffn_rx.clone().into(), buffer.ffn_r.clone().into())?,
                TensorOp::channel_mix_stack(
                    &buffer.cursors,
                    &buffer.ffn_r,
                    &buffer.ffn_v,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                )?,
                TensorOp::add(&buffer.att_o, &buffer.ffn_x)?,
            ];

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            ops.iter().for_each(|op| pass.execute_tensor_op(op));
            drop(pass);

            if index != self.info.num_layers - 1 {
                encoder.copy_tensor(&buffer.ffn_x, &buffer.input)?;
            }
        }

        let mut ops = vec![TensorOp::layer_norm(
            &tensor.head.layer_norm.w,
            &tensor.head.layer_norm.b,
            &buffer.head_x,
        )?];

        for (chunk, matrix) in tensor.head.w.iter().enumerate() {
            let start = chunk * self.head_chunk_size;
            let end = start + self.head_chunk_size;
            let input = buffer.head_x.as_view((.., .., ..head_ops.len()))?;
            let output = buffer.head_o.as_view((start..end, .., ..head_ops.len()))?;
            ops.push(TensorOp::matmul(matrix, input, output)?);
        }

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        head_ops.iter().for_each(|op| pass.execute_tensor_op(op));
        ops.iter().for_each(|op| pass.execute_tensor_op(op));
        drop(pass);

        encoder.copy_tensor(&buffer.head_o, &buffer.map)?;

        context.queue.submit(Some(encoder.finish()));
        Ok((buffer, redirect))
    }
}
