use anyhow::Result;
use bitflags::bitflags;
use derive_getters::Getters;
use half::f16;
use safetensors::SafeTensors;
use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor};

use crate::{
    context::Context,
    tensor::{
        ReadBack, ReadWrite, ResourceCache, Shape, TensorCommand, TensorCpu, TensorError,
        TensorExt, TensorGpu, TensorOp, TensorPass, TensorQueue, TensorView, Uniform,
    },
};

#[derive(Debug, Getters)]
pub struct Model<'a, 'b> {
    pub context: &'a Context,

    info: ModelInfo,
    /// The head matrix is too big for a storage buffer so it's divided into chunks.
    head_chunk_size: usize,
    /// To prevent the GPU device from lost, this limits the maximum batch-token it processes one time.
    max_batch_token_chunk_size: usize,

    #[getter(skip)]
    tensor: ModelTensor<'a, 'b>,
    #[getter(skip)]
    buffer_cache: ResourceCache<usize, ModelBuffer<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub num_emb: usize,
    pub num_vocab: usize,
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

#[derive(Debug)]
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

    head: TensorGpu<'a, f32, ReadWrite>,
    softmax: TensorGpu<'a, f32, ReadWrite>,

    map: TensorGpu<'a, f32, ReadBack>,
}

impl<'a> ModelBuffer<'a> {
    pub fn new(
        context: &'a Context,
        info: ModelInfo,
        input: TensorCpu<'a, '_, f32>,
        mask: TensorCpu<'a, '_, f32>,
    ) -> Result<Self, TensorError> {
        let shape = input.shape();
        let ffn_shape = Shape::new(shape[0] * 4, shape[1], shape[2]);
        let out_shape = Shape::new(info.num_vocab, shape[1], shape[2]);

        input.check_shape(Shape::new(info.num_emb, shape[1], shape[2]))?;

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
            head: context.init_tensor(out_shape),
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

#[derive(Debug, Clone)]
pub struct ModelState<'a> {
    pub info: ModelInfo,
    pub context: &'a Context,
    pub state: TensorGpu<'a, f32, ReadWrite>,
}

impl<'a> ModelState<'a> {
    pub fn new(context: &'a Context, info: ModelInfo, num_batches: usize) -> Self {
        let data = (0..num_batches)
            .map(|_| {
                (0..info.num_layers)
                    .map(|_| {
                        [
                            vec![0.0; info.num_emb],
                            vec![0.0; info.num_emb],
                            vec![0.0; info.num_emb],
                            vec![-f32::MIN; info.num_emb],
                            vec![0.0; info.num_emb],
                        ]
                        .concat()
                    })
                    .collect::<Vec<_>>()
                    .concat()
            })
            .collect::<Vec<_>>()
            .concat();
        let state = context
            .tensor_from_data(
                Shape::new(info.num_emb, 5 * info.num_layers, num_batches),
                data,
            )
            .unwrap();
        Self {
            info,
            context,
            state,
        }
    }

    pub fn load(&self, backed: &BackedState<'a, '_>) -> Result<(), TensorError> {
        self.context.queue.write_tensor(&backed.data, &self.state)
    }

    fn att(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let start = 5 * layer;
        let end = start + 4;
        self.state.as_view((.., start..end, ..))
    }

    fn ffn(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let start = 5 * layer + 4;
        self.state.as_view((.., start..=start, ..))
    }
}

#[derive(Debug, Clone)]
pub struct BackedState<'a, 'b> {
    pub info: ModelInfo,
    pub context: &'a Context,
    pub data: TensorCpu<'a, 'b, f32>,
}

impl<'a, 'b> From<ModelState<'a>> for BackedState<'a, 'b> {
    fn from(value: ModelState<'a>) -> Self {
        let ModelState {
            info,
            context,
            state,
        } = value;
        let map = context.init_tensor(state.shape());
        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_tensor(&state, &map).unwrap();
        Self {
            info,
            context,
            data: TensorCpu::from(map),
        }
    }
}

impl<'a, 'b> From<BackedState<'a, 'b>> for ModelState<'a> {
    fn from(value: BackedState<'a, 'b>) -> Self {
        let BackedState {
            info,
            context,
            data,
        } = value;
        let state = TensorGpu::from(data);
        Self {
            info,
            context,
            state,
        }
    }
}

pub struct ModelBuilder<'a, 'b> {
    context: &'a Context,
    data: &'b [u8],
    quant: Quantization,
    head_chunk_size: usize,
    max_batch_token_chunk_size: usize,
}

impl<'a, 'b> ModelBuilder<'a, 'b> {
    pub fn new(context: &'a Context, data: &'b [u8]) -> Self {
        Self {
            context,
            data,
            quant: Quantization::None,
            head_chunk_size: 4096,
            max_batch_token_chunk_size: 32,
        }
    }

    pub fn with_quant(self, quant: Quantization) -> Self {
        Self { quant, ..self }
    }

    pub fn with_max_head_chunk(self, size: usize) -> Self {
        Self {
            head_chunk_size: size,
            ..self
        }
    }

    pub fn with_max_token_batch_chunk(self, size: usize) -> Self {
        Self {
            max_batch_token_chunk_size: size,
            ..self
        }
    }

    pub fn build(self) -> Result<Model<'a, 'b>> {
        let Self {
            context,
            data,
            quant,
            head_chunk_size,
            max_batch_token_chunk_size,
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
            let data: Vec<_> = bytemuck::pod_collect_to_vec::<_, f16>(tensor.data())
                .into_iter()
                .map(f16::to_f32)
                .collect();
            let shape = Shape::new(data.len(), 1, 1);
            Ok(context.tensor_from_data(shape, data)?)
        };
        let load_vector_exp_f32 = |name: String| -> Result<TensorGpu<f32, ReadWrite>> {
            let tensor = model.tensor(&name)?;
            let data: Vec<_> = bytemuck::pod_collect_to_vec::<_, f16>(tensor.data())
                .into_iter()
                .map(f16::to_f32)
                .map(|x| -x.exp())
                .collect();
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

            let mx_f32 = context.init_tensor(Shape::new(shape[0], 1, 1));
            let rx_f32 = context.init_tensor(Shape::new(shape[0], 1, 1));
            let my_f32 = context.init_tensor(Shape::new(shape[1], 1, 1));
            let ry_f32 = context.init_tensor(Shape::new(shape[1], 1, 1));

            let w = Box::new(context.init_tensor(matrix.shape()));

            let mut ops =
                TensorOp::quantize_mat_int8(&matrix, &mx_f32, &rx_f32, &my_f32, &ry_f32, &w)?;

            let mx = Box::new(context.init_tensor(Shape::new(shape[0], 1, 1)));
            let rx = Box::new(context.init_tensor(Shape::new(shape[0], 1, 1)));
            let my = Box::new(context.init_tensor(Shape::new(shape[1], 1, 1)));
            let ry = Box::new(context.init_tensor(Shape::new(shape[1], 1, 1)));

            ops.push(TensorOp::quantize_vec_fp16(&mx_f32, &mx)?);
            ops.push(TensorOp::quantize_vec_fp16(&rx_f32, &rx)?);
            ops.push(TensorOp::quantize_vec_fp16(&my_f32, &my)?);
            ops.push(TensorOp::quantize_vec_fp16(&ry_f32, &ry)?);

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
            let tensor = model.tensor("head_weight")?;
            let chunks = tensor.shape()[1] / head_chunk_size;

            let shape = tensor.shape();
            let shape = Shape::new(shape[1], shape[0], 1);
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
                .collect::<Result<Vec<_>, TensorError>>()?;

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
                        w_k: load_matrix_u8(format!("{att}.key,weight"))?,
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
                        w_k: load_matrix_f16(format!("{att}.key,weight"))?,
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
            head_chunk_size,
            max_batch_token_chunk_size,
            tensor,
            buffer_cache: Default::default(),
        })
    }
}

impl<'a, 'b> Model<'a, 'b> {}
