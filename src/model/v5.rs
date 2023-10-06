use std::sync::Arc;

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use safetensors::SafeTensors;
use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor};

use super::{
    matrix::Matrix, BackedStateExt, Lora, ModelError, ModelExt, ModelInfo, ModelStateExt,
    ModelVersion, Quantization,
};
use crate::{
    context::Context,
    tensor::{
        cache::ResourceCache,
        ops::{TensorCommand, TensorOp, TensorPass},
        shape::{Shape, TensorDimension},
        IntoPackedCursors, ReadBack, ReadWrite, TensorCpu, TensorError, TensorGpu, TensorInit,
        TensorReshape, TensorShape, TensorStack, TensorView,
    },
};

#[derive(Debug)]
pub struct Model<'a> {
    context: Context,
    info: ModelInfo,

    /// The head matrix is too big for a storage buffer so it's divided into chunks.
    head_chunk_size: usize,
    /// To prevent the GPU device from lost, this limits the maximum batch-token it processes one time.
    token_chunk_size: usize,

    tensor: ModelTensor<'a>,
    runtime_cache: ResourceCache<usize, Runtime>,
    output_cache: ResourceCache<usize, Output>,
    softmax_cache: ResourceCache<usize, Softmax>,
    stack_cache: ResourceCache<usize, TensorGpu<u32, ReadWrite>>,
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
    att_gx: TensorGpu<f32, ReadWrite>,
    att_k: TensorGpu<f32, ReadWrite>,
    att_v: TensorGpu<f32, ReadWrite>,
    att_r: TensorGpu<f32, ReadWrite>,
    att_g: TensorGpu<f32, ReadWrite>,
    att_o: TensorGpu<f32, ReadWrite>,

    ffn_x: TensorGpu<f32, ReadWrite>,
    ffn_kx: TensorGpu<f32, ReadWrite>,
    ffn_rx: TensorGpu<f32, ReadWrite>,
    ffn_k: TensorGpu<f32, ReadWrite>,
    ffn_v: TensorGpu<f32, ReadWrite>,
    ffn_r: TensorGpu<f32, ReadWrite>,
}

impl Runtime {
    pub fn new(context: &Context, info: &ModelInfo, num_token: usize) -> Self {
        let shape = Shape::new(info.num_emb, num_token, 1, 1);
        let cursors_shape = Shape::new(num_token, 1, 1, 1);
        let hidden_shape = Shape::new(info.num_hidden, num_token, 1, 1);

        Self {
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
        }
    }
}

#[derive(Debug)]
struct Output {
    head_x: TensorGpu<f32, ReadWrite>,
    head_o: TensorGpu<f32, ReadWrite>,
    map: TensorGpu<f32, ReadBack>,
}

impl Output {
    pub fn new(context: &Context, info: &ModelInfo, num_batch: usize) -> Self {
        let head_shape = Shape::new(info.num_emb, num_batch, 1, 1);
        let output_shape = Shape::new(info.num_vocab, num_batch, 1, 1);

        Self {
            head_x: context.tensor_init(head_shape),
            head_o: context.tensor_init(output_shape),
            map: context.tensor_init(output_shape),
        }
    }
}

#[derive(Debug)]
struct Softmax {
    buffer: TensorGpu<f32, ReadWrite>,
    map: TensorGpu<f32, ReadBack>,
}

impl Softmax {
    pub fn new(context: &Context, info: &ModelInfo, num_batch: usize) -> Self {
        let shape = Shape::new(info.num_vocab, 1, num_batch, 1);
        Self {
            buffer: context.tensor_init(shape),
            map: context.tensor_init(shape),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelState {
    info: ModelInfo,
    max_batch: usize,
    chunk_size: usize,
    state: Vec<TensorGpu<f32, ReadWrite>>,
}

impl ModelState {
    /// Create a model state.
    /// - `max_batch`: The maximum number of runtime slots.
    /// - `chunk_size`: Internally, the state is split into chunks of layers, since there is a size limit on one GPU buffer (128 MB).
    /// If there is only one batch, it is recommended to set `chunk_size` to `info.num_layers()`.
    pub fn new(context: &Context, info: &ModelInfo, max_batch: usize, chunk_size: usize) -> Self {
        let num_chunk = (info.num_layers + chunk_size - 1) / chunk_size;
        let state = (0..num_chunk)
            .map(|_| {
                let data = (0..max_batch)
                    .map(|_| vec![0.0; chunk_size * info.num_emb * (info.head_size + 2)])
                    .collect_vec()
                    .concat();
                context
                    .tensor_from_data(
                        Shape::new(
                            info.num_emb,
                            chunk_size * (info.head_size + 2),
                            max_batch,
                            1,
                        ),
                        data,
                    )
                    .expect("state creation")
            })
            .collect();
        Self {
            info: info.clone(),
            max_batch,
            chunk_size,
            state,
        }
    }

    fn att(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let chunk = layer / self.chunk_size;
        let offset = layer % self.chunk_size;

        let start = offset * (self.info.head_size + 2);
        let end = start + self.info.head_size + 1;
        self.state[chunk].view(.., start..end, .., ..)
    }

    fn ffn(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let chunk = layer / self.chunk_size;
        let offset = layer % self.chunk_size;

        let start = offset * (self.info.head_size + 2) + self.info.head_size + 1;
        self.state[chunk].view(.., start..=start, .., ..)
    }
}

impl ModelStateExt for ModelState {
    type BackedState = BackedState;

    #[inline]
    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn load(&self, backed: &BackedState) -> Result<()> {
        if self.max_batch() != backed.max_batch() {
            return Err(ModelError::BatchSize(self.max_batch(), backed.max_batch()).into());
        }
        for (state, (shape, backed)) in self.state.iter().zip(backed.data.iter()) {
            let host = state.context.tensor_from_data(*shape, backed)?;
            state.load(&host)?;
        }
        Ok(())
    }

    fn load_batch(&self, backed: &BackedState, batch: usize) -> Result<()> {
        if self.max_batch() != backed.max_batch() {
            return Err(ModelError::BatchSize(self.max_batch(), backed.max_batch()).into());
        }
        for (state, (shape, backed)) in self.state.iter().zip(backed.data.iter()) {
            state.check_shape(*shape)?;
            let shape = state.shape();
            let shape = Shape::new(shape[0], shape[1], 1, 1);
            let host = state.context.tensor_from_data(shape, backed)?;
            state.load_batch(&host, batch)?;
        }
        Ok(())
    }

    fn back(&self) -> BackedState {
        let data = self
            .state
            .iter()
            .map(|state| {
                let shape = state.shape();
                let map = state.context.tensor_init(shape);

                let mut encoder = state
                    .context
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                encoder.copy_tensor(state, &map).expect("back entire state");
                state.context.queue.submit(Some(encoder.finish()));

                let host = TensorCpu::from(map);
                (shape, host.to_vec())
            })
            .collect();
        BackedState {
            max_batch: self.max_batch,
            data,
        }
    }

    fn back_batch(&self, batch: usize) -> Result<BackedState> {
        if batch >= self.max_batch() {
            return Err(ModelError::BatchOutOfRange {
                batch,
                max: self.max_batch(),
            }
            .into());
        }

        let data: Result<Vec<_>, _> = self
            .state
            .iter()
            .map(|state| -> Result<_, TensorError> {
                let shape = state.shape();
                let shape = Shape::new(shape[0], shape[1], 1, 1);
                let map = state.context.tensor_init(shape);

                let mut encoder = state
                    .context
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                encoder.copy_tensor_batch(state, &map, batch)?;
                state.context.queue.submit(Some(encoder.finish()));

                let host = TensorCpu::from(map);
                Ok((shape, host.to_vec()))
            })
            .collect();
        Ok(BackedState {
            max_batch: self.max_batch,
            data: data?,
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
            state.check_shape(other.shape())?;
            let op: TensorOp<'_> = TensorOp::blit(
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
    pub data: Vec<(Shape, Vec<f32>)>,
}

impl BackedState {
    pub fn new(info: &ModelInfo, max_batch: usize, chunk_size: usize) -> Self {
        let shape = Shape::new(
            info.num_emb,
            chunk_size * (info.head_size + 2),
            max_batch,
            1,
        );
        let data = (0..info.num_layers)
            .map(|_| {
                (0..max_batch)
                    .map(|_| vec![0.0; chunk_size * info.num_emb * (info.head_size + 2)])
                    .collect_vec()
                    .concat()
            })
            .map(|x| (shape, x))
            .collect();
        Self { max_batch, data }
    }
}

impl BackedStateExt for BackedState {
    #[inline]
    fn max_batch(&self) -> usize {
        self.max_batch
    }
}

pub struct ModelBuilder<'a> {
    context: Context,
    data: &'a [u8],
    lora: Vec<Lora<'a>>,
    quant: Quantization,
    head_chunk_size: usize,
    token_chunk_size: usize,
}

impl<'a> ModelBuilder<'a> {
    pub fn new(context: &Context, data: &'a [u8]) -> Self {
        Self {
            context: context.clone(),
            data,
            lora: vec![],
            quant: Quantization::None,
            head_chunk_size: 4096,
            token_chunk_size: 32,
        }
    }

    pub fn with_quant(self, quant: Quantization) -> Self {
        Self { quant, ..self }
    }

    pub fn add_lora(mut self, lora: Lora<'a>) -> Self {
        self.lora.push(lora);
        self
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

    pub fn build<'b>(self) -> Result<Model<'b>> {
        let Self {
            context,
            data,
            lora,
            quant,
            head_chunk_size,
            token_chunk_size,
        } = self;

        let model = SafeTensors::deserialize(data)?;
        let embed = model.tensor("emb.weight")?;
        let ffn = model.tensor("blocks.0.ffn.key.weight")?;

        let lora_tensors: Vec<_> = lora
            .iter()
            .map(|lora| SafeTensors::deserialize(lora.data))
            .try_collect()?;

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

        let info = {
            let num_emb = embed.shape()[1];
            let num_hidden = ffn.shape()[0];
            let num_vocab = embed.shape()[0];
            let head_size = if num_emb <= 2048 { 64 } else { 128 };

            ModelInfo {
                version: ModelVersion::V5,
                num_layers,
                num_emb,
                num_hidden,
                num_vocab,
                head_size,
            }
        };

        let lora_vectors = |name: &str| -> Vec<(TensorGpu<f32, ReadWrite>, f32)> {
            lora.iter()
                .zip_eq(lora_tensors.iter())
                .filter_map(|(lora, data)| {
                    // find the last blend that matches the name while the tensor exists in the data
                    lora.blend
                        .clone()
                        .into_patterns()
                        .into_iter()
                        .filter(|blend| blend.pattern.is_match(name))
                        .last()
                        .and_then(|blend| {
                            data.tensor(name).ok().and_then(|tensor| {
                                let tensor = TensorCpu::<f16>::from_safetensors(&context, tensor)
                                    .ok()?
                                    .map(|x| x.to_f32());
                                log::info!("loaded lora {}, alpha: {}", name, blend.alpha);
                                Some((tensor.into(), blend.alpha))
                            })
                        })
                })
                .collect()
        };
        let lora_matrices =
            |name: &str| -> Vec<(TensorGpu<f32, ReadWrite>, f32, usize)> {
                lora.iter()
                    .zip_eq(lora_tensors.iter())
                    .filter_map(|(lora, data)| {
                        // find the last blend that matches the name while the tensor exists in the data
                        lora.blend
                            .clone()
                            .into_patterns()
                            .into_iter()
                            .filter(|blend| blend.pattern.is_match(name))
                            .last()
                            .and_then(|blend| {
                                let a = data.tensor(&format!("{name}.lora_a")).ok().and_then(
                                    |tensor| TensorGpu::from_safetensors(&context, tensor).ok(),
                                )?;
                                let b = data.tensor(&format!("{name}.lora_b")).ok().and_then(
                                    |tensor| TensorGpu::from_safetensors(&context, tensor).ok(),
                                )?;
                                let output = TensorGpu::init(
                                    &context,
                                    Shape::new(a.shape()[1], b.shape()[1], 1, 1),
                                );

                                let mut encoder = context
                                    .device
                                    .create_command_encoder(&CommandEncoderDescriptor::default());

                                let op = TensorOp::matmul_mat_fp16(
                                    b.view(.., .., .., ..).ok()?,
                                    a.view(.., .., .., ..).ok()?,
                                    output.view(.., .., .., ..).ok()?,
                                )
                                .ok()?;
                                let mut pass =
                                    encoder.begin_compute_pass(&ComputePassDescriptor::default());
                                pass.execute_tensor_op(&op);
                                drop(pass);

                                context.queue.submit(Some(encoder.finish()));

                                log::info!("loaded lora {}, alpha: {}", name, blend.alpha);
                                Some((output, blend.alpha, a.shape()[0]))
                            })
                    })
                    .collect()
            };

        let load_vector_f32 = |name: String| -> Result<TensorGpu<f32, ReadWrite>> {
            use TensorDimension::{Auto, Dimension};
            let tensor = model.tensor(&name)?;
            let tensor = TensorCpu::<f16>::from_safetensors(&context, tensor)?
                .map(|x| x.to_f32())
                .reshape(Auto, Dimension(1), Dimension(1), Dimension(1))?
                .into();

            let mut encoder = context
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());

            for (lora, alpha) in lora_vectors(&name) {
                let factor = vec![alpha, 1.0 - alpha, 0.0, 0.0];
                let factor = TensorGpu::from_data(&context, Shape::new(4, 1, 1, 1), &factor)?;
                let op = TensorOp::blend(&factor, &lora, &tensor)?;
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.execute_tensor_op(&op);
            }

            context.queue.submit(Some(encoder.finish()));
            Ok(tensor)
        };
        let load_vector_exp_exp_f32 = |name: String| -> Result<TensorGpu<f32, ReadWrite>> {
            use TensorDimension::{Auto, Dimension};
            let tensor = model.tensor(&name)?;
            let tensor = TensorCpu::<f16>::from_safetensors(&context, tensor)?
                .map(|x| -x.to_f32().exp())
                .map(|x| x.exp())
                .reshape(Auto, Dimension(1), Dimension(1), Dimension(1))?
                .into();

            let mut encoder = context
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());

            for (lora, alpha) in lora_vectors(&name) {
                let factor = vec![alpha, 1.0 - alpha, 0.0, 0.0];
                let factor = TensorGpu::from_data(&context, Shape::new(4, 1, 1, 1), &factor)?;
                let op = TensorOp::blend(&factor, &lora, &tensor)?;
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.execute_tensor_op(&op);
            }

            context.queue.submit(Some(encoder.finish()));
            Ok(tensor)
        };
        let load_vector_f16 = |name: String| -> Result<TensorGpu<f16, ReadWrite>> {
            use TensorDimension::{Auto, Dimension};
            let lora = lora_vectors(&name);
            let tensor = model.tensor(&name)?;
            let tensor = if lora.is_empty() {
                TensorGpu::from_safetensors(&context, tensor)?.reshape(
                    Auto,
                    Dimension(1),
                    Dimension(1),
                    Dimension(1),
                )?
            } else {
                let tensor_f32 = TensorCpu::<f16>::from_safetensors(&context, tensor)?
                    .map(|x| x.to_f32())
                    .reshape(Auto, Dimension(1), Dimension(1), Dimension(1))?;
                let tensor_f32 = TensorGpu::from(tensor_f32);
                let tensor_f16 = context.tensor_init(tensor_f32.shape());

                let mut encoder = context
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor::default());

                for (lora, alpha) in lora {
                    let factor = vec![alpha, 1.0 - alpha, 0.0, 0.0];
                    let factor = TensorGpu::from_data(&context, Shape::new(4, 1, 1, 1), &factor)?;
                    let op = TensorOp::blend(&factor, &lora, &tensor_f32)?;
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.execute_tensor_op(&op);
                }

                let op = TensorOp::quantize_fp16(&tensor_f32, &tensor_f16)?;
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.execute_tensor_op(&op);
                drop(pass);

                context.queue.submit(Some(encoder.finish()));
                tensor_f16
            };
            Ok(tensor)
        };
        let load_matrix_f16 = |name: String| -> Result<TensorGpu<f16, ReadWrite>> {
            use TensorDimension::{Dimension, Full};
            let lora = lora_matrices(&name);
            let tensor = model.tensor(&name)?;
            let tensor = if lora.is_empty() {
                TensorGpu::from_safetensors(&context, tensor)?.reshape(
                    Full,
                    Full,
                    Dimension(1),
                    Dimension(1),
                )?
            } else {
                let tensor_f32 = TensorCpu::<f16>::from_safetensors(&context, tensor)?
                    .map(|x| x.to_f32())
                    .reshape(Full, Full, Dimension(1), Dimension(1))?;
                let tensor_f32 = TensorGpu::from(tensor_f32);
                let tensor_f16 = context.tensor_init(tensor_f32.shape());

                let mut encoder = context
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor::default());

                for (lora, alpha, dim) in lora {
                    let factor = vec![alpha / dim as f32, 1.0, 0.0, 0.0];
                    let factor = TensorGpu::from_data(&context, Shape::new(4, 1, 1, 1), &factor)?;
                    let op = TensorOp::blend(&factor, &lora, &tensor_f32)?;
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.execute_tensor_op(&op);
                }

                let op = TensorOp::quantize_fp16(&tensor_f32, &tensor_f16)?;
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.execute_tensor_op(&op);
                drop(pass);

                context.queue.submit(Some(encoder.finish()));
                tensor_f16
            };
            Ok(tensor)
        };

        let embed = Embed {
            layer_norm: LayerNorm {
                w: load_vector_f16("blocks.0.ln0.weight".into())?,
                b: load_vector_f16("blocks.0.ln0.bias".into())?,
            },
            w: context.tensor_from_data(
                Shape::new(info.num_emb, info.num_vocab, 1, 1),
                bytemuck::pod_collect_to_vec(embed.data()),
            )?,
        };

        let head = {
            let tensor = model.tensor("head.weight")?;
            let shape = tensor.shape();
            let shape = Shape::new(shape[1], shape[0], 1, 1);
            let chunks = shape[1] / head_chunk_size;
            let data = bytemuck::cast_slice(tensor.data());

            let w = (0..chunks)
                .map(|chunk| {
                    let start = (chunk * head_chunk_size) * shape[0];
                    let end = start + head_chunk_size * shape[0];
                    context.tensor_from_data(
                        Shape::new(shape[0], head_chunk_size, 1, 1),
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
                let time_decay = load_vector_exp_exp_f32(format!("{att}.time_decay"))?;
                let time_first = load_vector_f32(format!("{att}.time_first"))?;
                let time_mix_k = load_vector_f16(format!("{att}.time_mix_k"))?;
                let time_mix_v = load_vector_f16(format!("{att}.time_mix_v"))?;
                let time_mix_r = load_vector_f16(format!("{att}.time_mix_r"))?;
                let time_mix_g = load_vector_f16(format!("{att}.time_mix_g"))?;

                let w_k = load_matrix_f16(format!("{att}.key.weight"))?;
                let w_v = load_matrix_f16(format!("{att}.value.weight"))?;
                let w_r = load_matrix_f16(format!("{att}.receptance.weight"))?;
                let w_g = load_matrix_f16(format!("{att}.gate.weight"))?;
                let w_o = load_matrix_f16(format!("{att}.output.weight"))?;

                let group_norm = LayerNorm {
                    w: load_vector_f16(format!("{att}.ln_x.weight"))?.reshape(
                        TensorDimension::Dimension(info.head_size),
                        TensorDimension::Auto,
                        TensorDimension::Dimension(1),
                        TensorDimension::Dimension(1),
                    )?,
                    b: load_vector_f16(format!("{att}.ln_x.bias"))?.reshape(
                        TensorDimension::Dimension(info.head_size),
                        TensorDimension::Auto,
                        TensorDimension::Dimension(1),
                        TensorDimension::Dimension(1),
                    )?,
                };

                let att = match quant {
                    Quantization::Int8(x) if x.contains_layer(layer as u64) => Att {
                        time_decay,
                        time_first,
                        time_mix_k,
                        time_mix_v,
                        time_mix_r,
                        time_mix_g,
                        w_k: Matrix::quant_u8(w_k)?,
                        w_v: Matrix::quant_u8(w_v)?,
                        w_r: Matrix::quant_u8(w_r)?,
                        w_g: Matrix::quant_u8(w_g)?,
                        w_o: Matrix::quant_u8(w_o)?,
                        group_norm,
                    },
                    _ => Att {
                        time_decay,
                        time_first,
                        time_mix_k,
                        time_mix_v,
                        time_mix_r,
                        time_mix_g,
                        w_k: Matrix::Fp16(w_k),
                        w_v: Matrix::Fp16(w_v),
                        w_r: Matrix::Fp16(w_r),
                        w_g: Matrix::Fp16(w_g),
                        w_o: Matrix::Fp16(w_o),
                        group_norm,
                    },
                };

                let ffn_layer_norm = LayerNorm {
                    w: load_vector_f16(format!("blocks.{layer}.ln2.weight"))?,
                    b: load_vector_f16(format!("blocks.{layer}.ln2.bias"))?,
                };

                let ffn = format!("blocks.{layer}.ffn");
                let time_mix_k = load_vector_f16(format!("{ffn}.time_mix_k"))?;
                let time_mix_r = load_vector_f16(format!("{ffn}.time_mix_k"))?;

                let w_k = load_matrix_f16(format!("{ffn}.key.weight"))?;
                let w_v = load_matrix_f16(format!("{ffn}.value.weight"))?;
                let w_r = load_matrix_f16(format!("{ffn}.receptance.weight"))?;

                let ffn = match quant {
                    Quantization::Int8(x) if x.contains_layer(layer as u64) => Ffn {
                        time_mix_k,
                        time_mix_r,
                        w_k: Matrix::quant_u8(w_k)?,
                        w_v: Matrix::quant_u8(w_v)?,
                        w_r: Matrix::quant_u8(w_r)?,
                    },
                    _ => Ffn {
                        time_mix_k,
                        time_mix_r,
                        w_k: Matrix::Fp16(w_k),
                        w_v: Matrix::Fp16(w_v),
                        w_r: Matrix::Fp16(w_r),
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
            token_chunk_size,
            tensor,
            runtime_cache: ResourceCache::new(1),
            output_cache: ResourceCache::new(1),
            softmax_cache: ResourceCache::new(1),
            stack_cache: Default::default(),
        })
    }
}

impl<'a> Model<'a> {
    #[inline]
    fn request_runtime(&self, num_token: usize) -> Arc<Runtime> {
        self.runtime_cache.request(num_token, || {
            Runtime::new(&self.context, &self.info, num_token)
        })
    }

    #[inline]
    fn request_output(&self, num_batch: usize) -> Arc<Output> {
        self.output_cache.request(num_batch, || {
            Output::new(&self.context, &self.info, num_batch)
        })
    }

    #[inline]
    fn request_softmax(&self, num_batch: usize) -> Arc<Softmax> {
        self.softmax_cache.request(num_batch, || {
            Softmax::new(&self.context, &self.info, num_batch)
        })
    }

    #[inline]
    fn request_stack(&self, num_batch: usize) -> Arc<TensorGpu<u32, ReadWrite>> {
        self.stack_cache.request(num_batch, || {
            self.context.zeros(Shape::new(num_batch, 1, 1, 1))
        })
    }

    #[inline]
    fn head_shape(&self, num_batch: usize) -> Shape {
        Shape::new(self.info.num_vocab, 1, num_batch, 1)
    }

    fn run_internal(
        &self,
        tokens: Vec<Vec<u16>>,
        state: &ModelState,
        last: Option<usize>,
    ) -> Result<(Arc<Output>, Vec<Option<usize>>), TensorError> {
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
        let max_batch = input.max_batch();
        let num_batch = input.num_batch();
        let num_token = input.num_token();
        let head_size = state.info.head_size;
        assert_ne!(num_token, 0);
        assert_ne!(num_batch, 0);

        // collect batch output copy commands for later
        let mut redirect = vec![None; max_batch];
        let headers = input
            .cursors
            .iter()
            .filter(|cursor| cursor.len > 0)
            .filter(|cursor| !last.is_some_and(|index| cursor.batch == index))
            .enumerate()
            .map(|(index, cursor)| {
                redirect[cursor.batch] = Some(index);
                cursor.token + cursor.len - 1
            })
            .collect_vec();
        let num_header = headers.len();

        let buffer = self.request_runtime(num_token);
        let output = self.request_output(num_header.max(1));
        let stack = self.request_stack(num_batch);

        // gather and group copy operations
        let (head_ops, head_x) = if num_token == 1 || num_token == num_header {
            (vec![], &buffer.ffn_x)
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
            (ops, &output.head_x)
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

        let stack_host =
            context.tensor_from_data(stack.shape(), input.cursors.clone().into_stack())?;
        let cursors =
            context.tensor_from_data(buffer.cursors.shape(), input.cursors.into_cursors())?;

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

            let ops = vec![
                TensorOp::layer_norm(
                    &layer.att_layer_norm.w,
                    &layer.att_layer_norm.b,
                    &buffer.att_x,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.att.time_mix_k,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_kx,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.att.time_mix_v,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_vx,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.att.time_mix_r,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_rx,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.att.time_mix_g,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_gx,
                )?,
                layer.att.w_k.matmul_op(
                    buffer.att_kx.view(.., .., .., ..)?,
                    buffer.att_k.view(.., .., .., ..)?,
                )?,
                layer.att.w_v.matmul_op(
                    buffer.att_vx.view(.., .., .., ..)?,
                    buffer.att_v.view(.., .., .., ..)?,
                )?,
                layer.att.w_r.matmul_op(
                    buffer.att_rx.view(.., .., .., ..)?,
                    buffer.att_r.view(.., .., .., ..)?,
                )?,
                layer.att.w_g.matmul_op(
                    buffer.att_gx.view(.., .., .., ..)?,
                    buffer.att_g.view(.., .., .., ..)?,
                )?,
                TensorOp::time_mix_v5(
                    &stack,
                    &time_decay,
                    &time_first,
                    &att_k,
                    &att_v,
                    &att_r,
                    &att_x,
                    state.att(index)?,
                )?,
                TensorOp::group_norm(&layer.att.group_norm.w, &layer.att.group_norm.b, &att_x)?,
                TensorOp::silu(&buffer.att_g, &buffer.att_x)?,
                layer.att.w_o.matmul_op(
                    buffer.att_x.view(.., .., .., ..)?,
                    buffer.att_o.view(.., .., .., ..)?,
                )?,
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
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.ffn.time_mix_k,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                    &buffer.ffn_kx,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.ffn.time_mix_r,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                    &buffer.ffn_rx,
                )?,
                layer.ffn.w_k.matmul_op(
                    buffer.ffn_kx.view(.., .., .., ..)?,
                    buffer.ffn_k.view(.., .., .., ..)?,
                )?,
                TensorOp::squared_relu(&buffer.ffn_k)?,
                layer.ffn.w_v.matmul_op(
                    buffer.ffn_k.view(.., .., .., ..)?,
                    buffer.ffn_v.view(.., .., .., ..)?,
                )?,
                layer.ffn.w_r.matmul_op(
                    buffer.ffn_rx.view(.., .., .., ..)?,
                    buffer.ffn_r.view(.., .., .., ..)?,
                )?,
                TensorOp::channel_mix(
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

        if num_header > 0 {
            let mut ops = vec![TensorOp::layer_norm(
                &tensor.head.layer_norm.w,
                &tensor.head.layer_norm.b,
                head_x,
            )?];

            for (chunk, matrix) in tensor.head.w.iter().enumerate() {
                let start = chunk * self.head_chunk_size;
                let end = start + self.head_chunk_size;
                let input = head_x.view(.., .., .., ..)?;
                let output = output.head_o.view(start..end, .., .., ..)?;
                ops.push(TensorOp::matmul_vec_fp16(matrix, input, output)?);
            }

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            head_ops.iter().for_each(|op| pass.execute_tensor_op(op));
            ops.iter().for_each(|op| pass.execute_tensor_op(op));
            drop(pass);

            encoder.copy_tensor(&output.head_o, &output.map)?;
        }

        context.queue.submit(Some(encoder.finish()));
        Ok((output, redirect))
    }
}

impl ModelExt for Model<'_> {
    type ModelState = ModelState;

    #[inline]
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn softmax(&self, input: Vec<Option<Vec<f32>>>) -> Result<Vec<Option<Vec<f32>>>> {
        let max_batch = input.len();

        let mut redirect = vec![None; max_batch];
        let input: Vec<_> = input
            .into_iter()
            .enumerate()
            .filter_map(|(batch, data)| data.map(|data| (batch, data)))
            .map(|(batch, data)| {
                TensorCpu::from_data(&self.context, self.head_shape(1), data)
                    .map(|tensor| (batch, tensor))
            })
            .try_collect()?;
        let input = TensorCpu::stack(
            input
                .into_iter()
                .enumerate()
                .map(|(index, (batch, tensor))| {
                    redirect[batch] = Some(index);
                    tensor
                })
                .collect_vec(),
        )?;

        let num_batch = input.shape()[2];
        let softmax = self.request_softmax(num_batch);
        softmax.buffer.load(&input)?;

        let op = TensorOp::softmax(&softmax.buffer)?;

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        encoder.copy_tensor(&softmax.buffer, &softmax.map)?;
        self.context.queue.submit(Some(encoder.finish()));

        let mut output = TensorCpu::from(softmax.map.clone())
            .split(2)
            .expect("split buffer map")
            .into_iter()
            .map(|tensor| Some(tensor.to_vec()))
            .collect_vec();

        let mut probs = vec![None; max_batch];
        for (probs, redirect) in probs.iter_mut().zip_eq(redirect.into_iter()) {
            if let Some(redirect) = redirect {
                std::mem::swap(probs, &mut output[redirect]);
            }
        }

        Ok(probs)
    }

    fn run(
        &self,
        tokens: &mut Vec<Vec<u16>>,
        state: &Self::ModelState,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let num_token: usize = tokens.iter().map(Vec::len).sum();
        let max_batch = state.max_batch;

        if tokens.len() != max_batch {
            return Err(ModelError::BatchSize(tokens.len(), max_batch).into());
        }
        if num_token == 0 {
            return Ok(vec![None; max_batch]);
        }

        // we only infer at most `token_chunk_size` tokens at a time
        let mut num_token = num_token.min(self.token_chunk_size);
        let mut inputs = vec![vec![]; max_batch];
        let mut last = None;

        // take `num_token` tokens out of all the inputs and put into `input`
        for (index, (batch, input)) in tokens.iter_mut().zip(inputs.iter_mut()).enumerate() {
            let mid = batch.len().min(num_token);
            num_token -= mid;

            let (head, tail) = batch.split_at(mid);
            last = (!tail.is_empty()).then_some(index);
            *input = head.to_vec();
            *batch = tail.to_vec();

            if num_token == 0 {
                break;
            }
        }

        let (output, redirect) = self.run_internal(inputs, state, last)?;
        let output = TensorCpu::from(output.map.clone());

        Ok(redirect
            .into_iter()
            .map(|index| {
                index.map(|index| {
                    output
                        .slice(.., index, .., ..)
                        .expect("this never happens")
                        .to_vec()
                })
            })
            .collect())
    }
}
