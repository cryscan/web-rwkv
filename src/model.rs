use anyhow::Result;
use bytemuck::{cast_slice, pod_collect_to_vec};
use derive_getters::Getters;
use half::prelude::*;
use safetensors::SafeTensors;
use std::{borrow::Cow, cell::RefCell, num::NonZeroU64};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, Buffer, BufferBinding, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
};

use crate::Environment;

#[derive(Getters)]
pub struct Model {
    info: ModelInfo,
    env: Environment,
    #[getter(skip)]
    tensor: ModelTensor,
    #[getter(skip)]
    pipeline: ModelPipeline,
    #[getter(skip)]
    buffer: RefCell<ModelBuffer>,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub num_emb: usize,
    pub num_vocab: usize,
}

impl ModelInfo {
    pub const HEAD_CHUNK_SIZE: usize = 8192;
    pub const TOKEN_CHUNK_SIZE: usize = 16;
}

struct ModelTensor {
    dim: Buffer,
    embed: Embed,
    head: Head,
    layers: Vec<Layer>,
}

struct LayerNorm {
    w: Buffer,
    b: Buffer,
}

struct Att {
    time_decay: Buffer,
    time_first: Buffer,

    dims: Buffer,

    time_mix_k: Buffer,
    time_mix_v: Buffer,
    time_mix_r: Buffer,

    w_k: Buffer,
    w_v: Buffer,
    w_r: Buffer,
    w_o: Buffer,
}

struct Ffn {
    time_mix_k: Buffer,
    time_mix_r: Buffer,

    dims_k: Buffer,
    dims_v: Buffer,
    dims_r: Buffer,

    w_k: Buffer,
    w_v: Buffer,
    w_r: Buffer,
}

struct Layer {
    att_layer_norm: LayerNorm,
    ffn_layer_norm: LayerNorm,
    att: Att,
    ffn: Ffn,
}

struct Embed {
    layer_norm: LayerNorm,
    w: Vec<f16>,
}

struct Head {
    layer_norm: LayerNorm,

    dims: Buffer,
    w: Vec<Buffer>,
}

struct ModelPipeline {
    layer_norm: ComputePipeline,
    token_shift: ComputePipeline,
    matmul: ComputePipeline,
    token_mix: ComputePipeline,
    activation: ComputePipeline,
    channel_mix: ComputePipeline,
    add: ComputePipeline,
}

pub struct ModelBuffer {
    num_tokens_host: usize,
    num_tokens: Buffer,

    emb_x: Buffer,
    emb_o: Buffer,

    att_x: Buffer,
    att_kx: Buffer,
    att_vx: Buffer,
    att_rx: Buffer,
    att_k: Buffer,
    att_v: Buffer,
    att_r: Buffer,
    att_w: Buffer,
    att_o: Buffer,

    ffn_x: Buffer,
    ffn_kx: Buffer,
    ffn_vx: Buffer,
    ffn_rx: Buffer,
    ffn_k: Buffer,
    ffn_v: Buffer,
    ffn_r: Buffer,
    ffn_o: Buffer,

    head_x: Buffer,
    head_r: Buffer,
    head_o: Buffer,

    map: Buffer,
}

impl ModelBuffer {
    fn new(env: &Environment, info: ModelInfo, input: &[f32]) -> Self {
        let device = &env.device;

        let create_buffer_f32 = |capacity: usize| -> Buffer {
            device.create_buffer(&BufferDescriptor {
                label: None,
                size: 4 * capacity as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let load_buffer_f32 = |data: &[f32]| -> Buffer {
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            })
        };
        let create_uniform_u32 = |values: &[u32]| -> Buffer {
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(values),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            })
        };

        let num_emb = info.num_emb;
        let num_vocab = info.num_vocab;
        let num_tokens = input.len() / num_emb;
        let capacity = num_tokens * num_emb;

        let map = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4 * num_vocab as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        ModelBuffer {
            num_tokens_host: num_tokens,
            num_tokens: create_uniform_u32(&[num_tokens as u32]),
            emb_x: load_buffer_f32(input),
            emb_o: create_buffer_f32(capacity),
            att_x: create_buffer_f32(capacity),
            att_kx: create_buffer_f32(capacity),
            att_vx: create_buffer_f32(capacity),
            att_rx: create_buffer_f32(capacity),
            att_k: create_buffer_f32(capacity),
            att_v: create_buffer_f32(capacity),
            att_r: create_buffer_f32(capacity),
            att_w: create_buffer_f32(capacity),
            att_o: create_buffer_f32(capacity),
            ffn_x: create_buffer_f32(capacity),
            ffn_kx: create_buffer_f32(capacity),
            ffn_vx: create_buffer_f32(4 * capacity),
            ffn_rx: create_buffer_f32(capacity),
            ffn_k: create_buffer_f32(4 * capacity),
            ffn_v: create_buffer_f32(capacity),
            ffn_r: create_buffer_f32(capacity),
            ffn_o: create_buffer_f32(capacity),
            head_x: create_buffer_f32(num_emb),
            head_r: create_buffer_f32(num_emb),
            head_o: create_buffer_f32(num_vocab),
            map,
        }
    }

    fn reload(&self, env: &Environment, info: ModelInfo, input: &[f32]) {
        assert_eq!(self.num_tokens_host, input.len() / info.num_emb);
        env.queue.write_buffer(&self.emb_x, 0, cast_slice(input));
    }
}

pub struct ModelState {
    env: Environment,
    info: ModelInfo,
    layers: Vec<LayerState>,
}

struct LayerState {
    att: Buffer,
    ffn: Buffer,
}

#[derive(Clone)]
pub struct BackedModelState(Vec<Vec<f32>>);

#[derive(Debug)]
pub enum ModelStateError {
    LayerCountNotMatch,
    EmbedSizeNotMatch,
}

impl std::fmt::Display for ModelStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            ModelStateError::LayerCountNotMatch => write!(f, "layer count not match"),
            ModelStateError::EmbedSizeNotMatch => write!(f, "embed size not match"),
        }
    }
}

impl std::error::Error for ModelStateError {}

impl ModelState {
    fn new(env: Environment, info: ModelInfo) -> Self {
        let device = &env.device;

        let ModelInfo {
            num_layers,
            num_emb,
            ..
        } = info;

        let create_buffer_f32 = |data: &[f32]| -> Buffer {
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };

        let layers = (0..num_layers)
            .map(|_| {
                let mut att = vec![0.0f32; 4 * num_emb];
                att[3 * num_emb..4 * num_emb]
                    .iter_mut()
                    .for_each(|x| *x = -1.0e30);

                let ffn = vec![0.0f32; num_emb];

                LayerState {
                    att: create_buffer_f32(&att),
                    ffn: create_buffer_f32(&ffn),
                }
            })
            .collect();

        Self { env, info, layers }
    }

    pub fn back(&self) -> Result<BackedModelState> {
        let device = &self.env.device;
        let queue = &self.env.queue;

        let num_emb = self.info.num_emb;

        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let map = device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: 20 * num_emb as u64,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor::default());
                encoder.copy_buffer_to_buffer(&layer.att, 0, &map, 0, 16 * num_emb as u64);
                encoder.copy_buffer_to_buffer(
                    &layer.ffn,
                    0,
                    &map,
                    16 * num_emb as u64,
                    4 * num_emb as u64,
                );
                queue.submit(Some(encoder.finish()));

                let (sender, receiver) = async_channel::bounded(1);
                let slice = map.slice(..);
                slice.map_async(wgpu::MapMode::Read, move |v| {
                    sender.send_blocking(v).unwrap();
                });

                device.poll(wgpu::MaintainBase::Wait);
                match receiver.recv_blocking().unwrap() {
                    Ok(_) => {
                        let data = {
                            let data = slice.get_mapped_range();
                            cast_slice(&data).to_vec()
                        };
                        map.unmap();
                        Ok(data)
                    }
                    Err(err) => Err(err.into()),
                }
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(BackedModelState(layers))
    }

    pub fn load(&self, backed: &BackedModelState) -> std::result::Result<(), ModelStateError> {
        let queue = &self.env.queue;

        let ModelInfo {
            num_layers,
            num_emb,
            ..
        } = self.info;

        if backed.0.len() != num_layers {
            return Err(ModelStateError::LayerCountNotMatch);
        }

        for (backed_layer, layer) in backed.0.iter().zip(self.layers.iter()) {
            queue.write_buffer(&layer.att, 0, cast_slice(&backed_layer[..4 * num_emb]));
            queue.write_buffer(&layer.ffn, 0, cast_slice(&backed_layer[4 * num_emb..]));
        }

        Ok(())
    }
}

struct ModelBindGroup {
    embed: EmbedBindGroup,
    head: HeadBindGroup,
    layers: Vec<LayerBindGroup>,
}

struct EmbedBindGroup {
    layer_norm: BindGroup,
}

struct HeadBindGroup {
    layer_norm: BindGroup,
    matmul: Vec<BindGroup>,
}

struct LayerBindGroup {
    att_layer_norm: BindGroup,
    att_token_shift_k: BindGroup,
    att_token_shift_v: BindGroup,
    att_token_shift_r: BindGroup,
    att_matmul_k: BindGroup,
    att_matmul_v: BindGroup,
    att_matmul_r: BindGroup,
    att_token_mix: BindGroup,
    att_matmul_o: BindGroup,
    att_add: BindGroup,

    ffn_layer_norm: BindGroup,
    ffn_token_shift_k: BindGroup,
    ffn_token_shift_r: BindGroup,
    ffn_matmul_k: BindGroup,
    ffn_activation: BindGroup,
    ffn_matmul_v: BindGroup,
    ffn_matmul_r: BindGroup,
    ffn_channel_mix: BindGroup,
    ffn_add: BindGroup,
}

impl Environment {
    pub fn create_model_from_bytes(&self, data: &[u8]) -> Result<Model> {
        let device = &self.device;
        let queue = &self.queue;
        let model = SafeTensors::deserialize(data)?;

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
            let emb = model.tensor("emb.weight")?;
            let num_emb = emb.shape()[1];
            let num_vocab = emb.shape()[0];
            (num_emb, num_vocab)
        };

        let info = ModelInfo {
            num_layers,
            num_emb,
            num_vocab,
        };

        let load_tensor_f32 = |name: String| -> Result<Buffer> {
            let tensor = model.tensor(&name)?.data();
            let tensor: Vec<_> = pod_collect_to_vec::<_, f16>(tensor)
                .into_iter()
                .map(f16::to_f32)
                .collect();
            let buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&name),
                contents: cast_slice(&tensor),
                usage: BufferUsages::STORAGE,
            });
            Ok(buffer)
        };
        let load_tensor_exp_f32 = |name: String| -> Result<Buffer> {
            let tensor = model.tensor(&name)?.data();
            let tensor: Vec<_> = pod_collect_to_vec::<_, f16>(tensor)
                .into_iter()
                .map(f16::to_f32)
                .map(|x| -x.exp())
                .collect();
            let buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&name),
                contents: cast_slice(&tensor),
                usage: BufferUsages::STORAGE,
            });
            Ok(buffer)
        };
        let load_tensor_f16 = |name: String| -> Result<Buffer> {
            let tensor = model.tensor(&name)?.data();
            let buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&name),
                contents: cast_slice(tensor),
                usage: BufferUsages::STORAGE,
            });
            Ok(buffer)
        };
        let create_uniform_u32 = |values: &[u32]| -> Buffer {
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(values),
                usage: BufferUsages::UNIFORM,
            })
        };

        let embed = Embed {
            layer_norm: LayerNorm {
                w: load_tensor_f16("blocks.0.ln0.weight".into())?,
                b: load_tensor_f16("blocks.0.ln0.bias".into())?,
            },
            w: pod_collect_to_vec(model.tensor("emb.weight")?.data()),
        };
        let head = {
            let chunk_size = ModelInfo::HEAD_CHUNK_SIZE;
            let w: Vec<f16> = pod_collect_to_vec(model.tensor("head.weight")?.data());
            let w = (0..num_vocab / chunk_size)
                .map(|chunk| {
                    let begin = chunk_size * chunk * num_emb;
                    let end = begin + chunk_size * num_emb;
                    device.create_buffer_init(&BufferInitDescriptor {
                        label: None,
                        contents: cast_slice(&w[begin..end]),
                        usage: BufferUsages::STORAGE,
                    })
                })
                .collect();

            Head {
                layer_norm: LayerNorm {
                    w: load_tensor_f16("ln_out.weight".into())?,
                    b: load_tensor_f16("ln_out.bias".into())?,
                },
                dims: create_uniform_u32(&[num_emb as u32, num_vocab as u32]),
                w,
            }
        };
        queue.submit(None);
        device.poll(wgpu::MaintainBase::Wait);

        let mut layers = vec![];
        for layer in 0..num_layers {
            let att_layer_norm = LayerNorm {
                w: load_tensor_f16(format!("blocks.{layer}.ln1.weight"))?,
                b: load_tensor_f16(format!("blocks.{layer}.ln1.bias"))?,
            };

            let att = format!("blocks.{layer}.att");
            let att = Att {
                time_decay: load_tensor_exp_f32(format!("{att}.time_decay"))?,
                time_first: load_tensor_f32(format!("{att}.time_first"))?,
                time_mix_k: load_tensor_f16(format!("{att}.time_mix_k"))?,
                time_mix_v: load_tensor_f16(format!("{att}.time_mix_v"))?,
                time_mix_r: load_tensor_f16(format!("{att}.time_mix_r"))?,
                dims: create_uniform_u32(&[num_emb as u32, num_emb as u32]),
                w_k: load_tensor_f16(format!("{att}.key.weight"))?,
                w_v: load_tensor_f16(format!("{att}.value.weight"))?,
                w_r: load_tensor_f16(format!("{att}.receptance.weight"))?,
                w_o: load_tensor_f16(format!("{att}.output.weight"))?,
            };

            let ffn_layer_norm = LayerNorm {
                w: load_tensor_f16(format!("blocks.{layer}.ln2.weight"))?,
                b: load_tensor_f16(format!("blocks.{layer}.ln2.bias"))?,
            };

            let ffn = format!("blocks.{layer}.ffn");
            let ffn = Ffn {
                time_mix_k: load_tensor_f16(format!("{ffn}.time_mix_k"))?,
                time_mix_r: load_tensor_f16(format!("{ffn}.time_mix_r"))?,
                dims_k: create_uniform_u32(&[num_emb as u32, 4 * num_emb as u32]),
                dims_v: create_uniform_u32(&[4 * num_emb as u32, num_emb as u32]),
                dims_r: create_uniform_u32(&[num_emb as u32, num_emb as u32]),
                w_k: load_tensor_f16(format!("{ffn}.key.weight"))?,
                w_v: load_tensor_f16(format!("{ffn}.value.weight"))?,
                w_r: load_tensor_f16(format!("{ffn}.receptance.weight"))?,
            };

            queue.submit(None);
            device.poll(wgpu::MaintainBase::Wait);
            layers.push(Layer {
                att_layer_norm,
                ffn_layer_norm,
                att,
                ffn,
            });
        }

        let dim = create_uniform_u32(&[num_emb as u32]);
        let tensor = ModelTensor {
            dim,
            embed,
            head,
            layers,
        };

        let create_pipeline = |shader: &str, entry_point: &str| -> ComputePipeline {
            let module = &device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(Cow::Borrowed(shader)),
            });
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module,
                entry_point,
            })
        };

        let pipeline = ModelPipeline {
            layer_norm: create_pipeline(include_str!("shaders/layer_norm.wgsl"), "layer_norm"),
            token_shift: create_pipeline(include_str!("shaders/token_shift.wgsl"), "token_shift"),
            matmul: create_pipeline(include_str!("shaders/matmul.wgsl"), "matmul"),
            token_mix: create_pipeline(include_str!("shaders/token_mix.wgsl"), "token_mix"),
            activation: create_pipeline(include_str!("shaders/activation.wgsl"), "activation"),
            channel_mix: create_pipeline(include_str!("shaders/channel_mix.wgsl"), "channel_mix"),
            add: create_pipeline(include_str!("shaders/add.wgsl"), "add"),
        };

        let input = vec![0.0; num_emb];
        let buffer = RefCell::new(ModelBuffer::new(self, info, &input));

        queue.submit(None);
        device.poll(wgpu::MaintainBase::Wait);
        Ok(Model {
            env: self.clone(),
            info,
            tensor,
            pipeline,
            buffer,
        })
    }
}

impl Model {
    pub fn embedding(&self, tokens: &[u16]) -> Vec<f32> {
        let num_tokens = tokens.len();
        let num_emb = self.info.num_emb;
        let capacity = num_tokens * num_emb;

        let mut input = vec![];
        input.reserve(capacity);
        for token in tokens {
            let index = *token as usize;
            let mut embed: Vec<_> = self.tensor.embed.w[index * num_emb..(index + 1) * num_emb]
                .iter()
                .copied()
                .map(f16::to_f32)
                .collect();
            input.append(&mut embed);
        }
        input
    }

    pub fn create_state(&self) -> ModelState {
        ModelState::new(self.env.clone(), self.info)
    }

    fn create_bind_group(&self, buffer: &ModelBuffer, state: &ModelState) -> ModelBindGroup {
        let device = &self.env.device;
        let pipeline = &self.pipeline;

        let [layer_norm_layout, token_shift_layout, matmul_layout, token_mix_layout, activation_layout, channel_mix_layout, add_layout] =
            [
                &pipeline.layer_norm,
                &pipeline.token_shift,
                &pipeline.matmul,
                &pipeline.token_mix,
                &pipeline.activation,
                &pipeline.channel_mix,
                &pipeline.add,
            ]
            .map(|pipeline| pipeline.get_bind_group_layout(0));

        let embed = {
            let layer_norm = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &layer_norm_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.tensor.dim.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: buffer.emb_x.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.tensor.embed.layer_norm.w.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.tensor.embed.layer_norm.b.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: buffer.emb_o.as_entire_binding(),
                    },
                ],
            });
            EmbedBindGroup { layer_norm }
        };

        let head = {
            let layer_norm = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &layer_norm_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.tensor.dim.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: buffer.head_x.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.tensor.head.layer_norm.w.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.tensor.head.layer_norm.b.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: buffer.head_r.as_entire_binding(),
                    },
                ],
            });
            let matmul = self
                .tensor
                .head
                .w
                .iter()
                .enumerate()
                .map(|(chunk, w)| {
                    let chunk_size = ModelInfo::HEAD_CHUNK_SIZE as u64;
                    let offset = 4 * chunk as u64 * chunk_size;
                    device.create_bind_group(&BindGroupDescriptor {
                        label: None,
                        layout: &matmul_layout,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: self.tensor.head.dims.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: w.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 2,
                                resource: buffer.head_r.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Buffer(BufferBinding {
                                    buffer: &buffer.head_o,
                                    offset,
                                    size: NonZeroU64::new(4 * chunk_size),
                                }),
                            },
                        ],
                    })
                })
                .collect();
            HeadBindGroup { layer_norm, matmul }
        };

        let layers = (0..self.info.num_layers)
            .map(|layer| {
                let state = &state.layers[layer];
                let layer = &self.tensor.layers[layer];

                let att_layer_norm = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &layer_norm_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.emb_o.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: layer.att_layer_norm.w.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: layer.att_layer_norm.b.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: buffer.att_x.as_entire_binding(),
                        },
                    ],
                });
                let att_token_shift_k = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &token_shift_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.att.time_mix_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: state.att.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: buffer.att_kx.as_entire_binding(),
                        },
                    ],
                });
                let att_token_shift_v = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &token_shift_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.att.time_mix_v.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: state.att.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: buffer.att_vx.as_entire_binding(),
                        },
                    ],
                });
                let att_token_shift_r = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &token_shift_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.att.time_mix_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: state.att.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: buffer.att_rx.as_entire_binding(),
                        },
                    ],
                });
                let att_matmul_k = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &matmul_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: layer.att.dims.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.att.w_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.att_kx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: buffer.att_k.as_entire_binding(),
                        },
                    ],
                });
                let att_matmul_v = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &matmul_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: layer.att.dims.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.att.w_v.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.att_vx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: buffer.att_v.as_entire_binding(),
                        },
                    ],
                });
                let att_matmul_r = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &matmul_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: layer.att.dims.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.att.w_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.att_rx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: buffer.att_r.as_entire_binding(),
                        },
                    ],
                });
                let att_token_mix = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &token_mix_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.num_tokens.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: layer.att.time_decay.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: layer.att.time_first.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: buffer.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: buffer.att_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 6,
                            resource: buffer.att_v.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 7,
                            resource: buffer.att_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 8,
                            resource: state.att.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 9,
                            resource: buffer.att_w.as_entire_binding(),
                        },
                    ],
                });
                let att_matmul_o = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &matmul_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: layer.att.dims.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.att.w_o.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.att_w.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: buffer.att_o.as_entire_binding(),
                        },
                    ],
                });
                let att_add = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &add_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.emb_o.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.att_o.as_entire_binding(),
                        },
                    ],
                });

                let ffn_layer_norm = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &layer_norm_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.att_o.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: layer.ffn_layer_norm.w.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: layer.ffn_layer_norm.b.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: buffer.ffn_x.as_entire_binding(),
                        },
                    ],
                });
                let ffn_token_shift_k = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &token_shift_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.ffn.time_mix_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.ffn_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: state.ffn.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: buffer.ffn_kx.as_entire_binding(),
                        },
                    ],
                });
                let ffn_token_shift_r = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &token_shift_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.ffn.time_mix_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.ffn_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: state.ffn.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: buffer.ffn_rx.as_entire_binding(),
                        },
                    ],
                });
                let ffn_matmul_k = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &matmul_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: layer.ffn.dims_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.ffn.w_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.ffn_kx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: buffer.ffn_k.as_entire_binding(),
                        },
                    ],
                });
                let ffn_activation = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &activation_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.ffn_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.ffn_vx.as_entire_binding(),
                        },
                    ],
                });
                let ffn_matmul_v = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &matmul_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: layer.ffn.dims_v.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.ffn.w_v.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.ffn_vx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: buffer.ffn_v.as_entire_binding(),
                        },
                    ],
                });
                let ffn_matmul_r = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &matmul_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: layer.ffn.dims_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.ffn.w_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.ffn_rx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: buffer.ffn_r.as_entire_binding(),
                        },
                    ],
                });
                let ffn_channel_mix = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &channel_mix_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.ffn_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.ffn_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: buffer.ffn_v.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: state.ffn.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: buffer.ffn_o.as_entire_binding(),
                        },
                    ],
                });
                let ffn_add = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &add_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.tensor.dim.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.att_o.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.ffn_o.as_entire_binding(),
                        },
                    ],
                });

                LayerBindGroup {
                    att_layer_norm,
                    att_token_shift_k,
                    att_token_shift_v,
                    att_token_shift_r,
                    att_matmul_k,
                    att_matmul_v,
                    att_matmul_r,
                    att_token_mix,
                    att_matmul_o,
                    att_add,
                    ffn_layer_norm,
                    ffn_token_shift_k,
                    ffn_token_shift_r,
                    ffn_matmul_k,
                    ffn_activation,
                    ffn_matmul_v,
                    ffn_matmul_r,
                    ffn_channel_mix,
                    ffn_add,
                }
            })
            .collect();

        ModelBindGroup {
            embed,
            head,
            layers,
        }
    }

    fn reload_buffer(&self, tokens: &[u16]) {
        let mut buffer = self.buffer.borrow_mut();
        let input = self.embedding(tokens);
        if buffer.num_tokens_host != tokens.len() {
            *buffer = ModelBuffer::new(&self.env, self.info, &input);
        } else {
            buffer.reload(&self.env, self.info, &input);
        }
    }

    fn run_internal(&self, tokens: &[u16], state: &ModelState, output: bool) {
        let device = &self.env.device;
        let queue = &self.env.queue;

        self.reload_buffer(tokens);
        let buffer = self.buffer.borrow();

        let bind_group = self.create_bind_group(&buffer, state);
        let pipeline = &self.pipeline;

        let ModelInfo {
            num_emb, num_vocab, ..
        } = self.info;

        let num_tokens = buffer.num_tokens_host as u32;
        let num_emb_vec4 = num_emb as u32 / 4;
        let num_emb_blocks = (num_emb_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let chunk_size_vec4 = ModelInfo::HEAD_CHUNK_SIZE as u32 / 4;
        const BLOCK_SIZE: u32 = 256;

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_pipeline(&pipeline.layer_norm);
            pass.set_bind_group(0, &bind_group.embed.layer_norm, &[]);
            pass.dispatch_workgroups(1, num_tokens, 1);
        }

        for layer in &bind_group.layers {
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

                pass.set_pipeline(&pipeline.layer_norm);
                pass.set_bind_group(0, &layer.att_layer_norm, &[]);
                pass.dispatch_workgroups(1, num_tokens, 1);

                pass.set_pipeline(&pipeline.token_shift);
                pass.set_bind_group(0, &layer.att_token_shift_k, &[]);
                pass.dispatch_workgroups(num_emb_blocks, num_tokens, 1);

                pass.set_bind_group(0, &layer.att_token_shift_v, &[]);
                pass.dispatch_workgroups(num_emb_blocks, num_tokens, 1);

                pass.set_bind_group(0, &layer.att_token_shift_r, &[]);
                pass.dispatch_workgroups(num_emb_blocks, num_tokens, 1);

                pass.set_pipeline(&pipeline.matmul);
                pass.set_bind_group(0, &layer.att_matmul_k, &[]);
                pass.dispatch_workgroups(1, num_emb_vec4, num_tokens);

                pass.set_bind_group(0, &layer.att_matmul_v, &[]);
                pass.dispatch_workgroups(1, num_emb_vec4, num_tokens);

                pass.set_bind_group(0, &layer.att_matmul_r, &[]);
                pass.dispatch_workgroups(1, num_emb_vec4, num_tokens);

                pass.set_pipeline(&pipeline.token_mix);
                pass.set_bind_group(0, &layer.att_token_mix, &[]);
                pass.dispatch_workgroups(num_emb_blocks, 1, 1);

                pass.set_pipeline(&pipeline.matmul);
                pass.set_bind_group(0, &layer.att_matmul_o, &[]);
                pass.dispatch_workgroups(1, num_emb_vec4, num_tokens);

                pass.set_pipeline(&pipeline.add);
                pass.set_bind_group(0, &layer.att_add, &[]);
                pass.dispatch_workgroups(num_emb_blocks, num_tokens, 1);

                pass.set_pipeline(&pipeline.layer_norm);
                pass.set_bind_group(0, &layer.ffn_layer_norm, &[]);
                pass.dispatch_workgroups(1, num_tokens, 1);

                pass.set_pipeline(&pipeline.token_shift);
                pass.set_bind_group(0, &layer.ffn_token_shift_k, &[]);
                pass.dispatch_workgroups(num_emb_blocks, num_tokens, 1);

                pass.set_bind_group(0, &layer.ffn_token_shift_r, &[]);
                pass.dispatch_workgroups(num_emb_blocks, num_tokens, 1);

                pass.set_pipeline(&pipeline.matmul);
                pass.set_bind_group(0, &layer.ffn_matmul_k, &[]);
                pass.dispatch_workgroups(1, 4 * num_emb_vec4, num_tokens);

                pass.set_bind_group(0, &layer.ffn_matmul_r, &[]);
                pass.dispatch_workgroups(1, num_emb_vec4, num_tokens);

                pass.set_pipeline(&pipeline.activation);
                pass.set_bind_group(0, &layer.ffn_activation, &[]);
                pass.dispatch_workgroups(4 * num_emb_blocks, num_tokens, 1);

                pass.set_pipeline(&pipeline.matmul);
                pass.set_bind_group(0, &layer.ffn_matmul_v, &[]);
                pass.dispatch_workgroups(1, num_emb_vec4, num_tokens);

                pass.set_pipeline(&pipeline.channel_mix);
                pass.set_bind_group(0, &layer.ffn_channel_mix, &[]);
                pass.dispatch_workgroups(num_emb_blocks, num_tokens, 1);

                pass.set_pipeline(&pipeline.add);
                pass.set_bind_group(0, &layer.ffn_add, &[]);
                pass.dispatch_workgroups(num_emb_blocks, num_tokens, 1);
            }

            encoder.copy_buffer_to_buffer(
                &buffer.ffn_o,
                0,
                &buffer.emb_o,
                0,
                4 * num_emb as u64 * num_tokens as u64,
            );
        }

        if output {
            encoder.copy_buffer_to_buffer(
                &buffer.ffn_o,
                4 * (num_tokens - 1) as u64 * num_emb as u64,
                &buffer.head_x,
                0,
                4 * num_emb as u64,
            );

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

                pass.set_pipeline(&pipeline.layer_norm);
                pass.set_bind_group(0, &bind_group.head.layer_norm, &[]);
                pass.dispatch_workgroups(1, 1, 1);

                pass.set_pipeline(&pipeline.matmul);
                for matmul in &bind_group.head.matmul {
                    pass.set_bind_group(0, matmul, &[]);
                    pass.dispatch_workgroups(1, chunk_size_vec4, 1);
                }
            }

            encoder.copy_buffer_to_buffer(&buffer.head_o, 0, &buffer.map, 0, 4 * num_vocab as u64);
        }
        queue.submit(Some(encoder.finish()));
    }

    pub fn run(&self, tokens: &[u16], state: &ModelState) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Ok(vec![0.0; self.info.num_vocab]);
        }

        let device = &self.env.device;

        let chunk_size = ModelInfo::TOKEN_CHUNK_SIZE;
        let mut tokens = tokens.to_vec();

        for _ in 0..(tokens.len() - 1) / chunk_size {
            self.run_internal(&tokens[..chunk_size], state, false);
            device.poll(wgpu::MaintainBase::Wait);
            tokens = tokens[chunk_size..].to_vec();
        }
        self.run_internal(&tokens, state, true);

        let buffer = self.buffer.borrow();
        let (sender, receiver) = async_channel::bounded(1);
        let slice = buffer.map.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send_blocking(v).unwrap();
        });

        device.poll(wgpu::MaintainBase::Wait);
        match receiver.recv_blocking().unwrap() {
            Ok(_) => {
                let data = {
                    let data = slice.get_mapped_range();
                    cast_slice(&data).to_vec()
                };
                buffer.map.unmap();
                Ok(data)
            }
            Err(err) => Err(err.into()),
        }
    }
}
