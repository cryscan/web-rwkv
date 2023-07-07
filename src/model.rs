use anyhow::Result;
use bytemuck::{cast_slice, pod_collect_to_vec};
use half::prelude::*;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::{borrow::Cow, fs::File, path::PathBuf, sync::Arc};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages,
    ComputePipeline, ComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
};

use crate::Environment;

#[derive(Clone)]
pub struct Model {
    pub env: Arc<Environment>,
    pub info: ModelInfo,
    pub tensor: Arc<ModelTensor>,
    pub pipeline: Arc<ModelPipeline>,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub num_emb: usize,
    pub num_vocab: usize,
}

pub struct ModelTensor {
    pub dim: Buffer,
    pub embed: Embed,
    pub head: Head,
    pub layers: Vec<Layer>,
}

pub struct LayerNorm {
    pub w: Buffer,
    pub b: Buffer,
}

pub struct Att {
    pub time_decay: Buffer,
    pub time_first: Buffer,

    pub dims: Buffer,

    pub time_mix_k: Buffer,
    pub time_mix_v: Buffer,
    pub time_mix_r: Buffer,

    pub w_k: Buffer,
    pub w_v: Buffer,
    pub w_r: Buffer,
    pub w_o: Buffer,
}

pub struct Ffn {
    pub time_mix_k: Buffer,
    pub time_mix_r: Buffer,

    pub dims_k: Buffer,
    pub dims_v: Buffer,
    pub dims_r: Buffer,

    pub w_k: Buffer,
    pub w_v: Buffer,
    pub w_r: Buffer,
}

pub struct Layer {
    pub att_layer_norm: LayerNorm,
    pub ffn_layer_norm: LayerNorm,
    pub att: Att,
    pub ffn: Ffn,
}

pub struct Embed {
    pub layer_norm: LayerNorm,
    pub w: Vec<f16>,
}

pub struct Head {
    pub layer_norm: LayerNorm,

    pub dims: Buffer,
    pub w: Buffer,
}

pub struct ModelPipeline {
    pub layer_norm: ComputePipeline,
    pub token_shift: ComputePipeline,
    pub matmul: ComputePipeline,
    pub token_mix: ComputePipeline,
    pub activation: ComputePipeline,
    pub channel_mix: ComputePipeline,
    pub add: ComputePipeline,
}

pub struct ModelBuffer {
    pub num_tokens: Buffer,

    pub emb_x: Buffer,
    pub emb_o: Buffer,

    pub att_x: Buffer,
    pub att_kx: Buffer,
    pub att_vx: Buffer,
    pub att_rx: Buffer,
    pub att_k: Buffer,
    pub att_v: Buffer,
    pub att_r: Buffer,
    pub att_w: Buffer,
    pub att_o: Buffer,

    pub ffn_x: Buffer,
    pub ffn_kx: Buffer,
    pub ffn_vx: Buffer,
    pub ffn_rx: Buffer,
    pub ffn_k: Buffer,
    pub ffn_v: Buffer,
    pub ffn_r: Buffer,
    pub ffn_o: Buffer,

    pub head_x: Buffer,
    pub head_r: Buffer,
    pub head_o: Buffer,

    pub map: Buffer,
}

pub struct ModelState(pub Vec<LayerState>);

pub struct LayerState {
    pub att_x: Buffer,
    pub att_a: Buffer,
    pub att_b: Buffer,
    pub att_p: Buffer,
    pub ffn_x: Buffer,
}

pub struct ModelBindGroup {
    pub embed: EmbedBindGroup,
    pub head: HeadBindGroup,
    pub layers: Vec<LayerBindGroup>,
}

pub struct EmbedBindGroup {
    pub layer_norm: BindGroup,
}

pub struct HeadBindGroup {
    pub layer_norm: BindGroup,
    pub matmul: BindGroup,
}

pub struct LayerBindGroup {
    pub att_layer_norm: BindGroup,
    pub att_token_shift_k: BindGroup,
    pub att_token_shift_v: BindGroup,
    pub att_token_shift_r: BindGroup,
    pub att_matmul_k: BindGroup,
    pub att_matmul_v: BindGroup,
    pub att_matmul_r: BindGroup,
    pub att_token_mix: BindGroup,
    pub att_matmul_o: BindGroup,
    pub att_add: BindGroup,

    pub ffn_layer_norm: BindGroup,
    pub ffn_token_shift_k: BindGroup,
    pub ffn_token_shift_r: BindGroup,
    pub ffn_matmul_k: BindGroup,
    pub ffn_squared_relu: BindGroup,
    pub ffn_matmul_v: BindGroup,
    pub ffn_matmul_r: BindGroup,
    pub ffn_channel_mix: BindGroup,
    pub ffn_add: BindGroup,
}

impl Model {
    pub fn from_file<'a>(path: PathBuf, env: Arc<Environment>) -> Result<Self> {
        let device = &env.device;

        let file = File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        let model = SafeTensors::deserialize(&map)?;

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
                w: load_tensor_f32("blocks.0.ln0.weight".into())?,
                b: load_tensor_f32("blocks.0.ln0.bias".into())?,
            },
            w: pod_collect_to_vec(model.tensor("emb.weight")?.data()),
        };
        let head = Head {
            layer_norm: LayerNorm {
                w: load_tensor_f32("ln_out.weight".into())?,
                b: load_tensor_f32("ln_out.bias".into())?,
            },
            dims: create_uniform_u32(&[num_emb as u32, num_vocab as u32]),
            w: load_tensor_f16("head.weight".into())?,
        };

        let mut layers = vec![];
        for layer in 0..num_layers {
            let att_layer_norm = LayerNorm {
                w: load_tensor_f32(format!("blocks.{layer}.ln1.weight"))?,
                b: load_tensor_f32(format!("blocks.{layer}.ln1.bias"))?,
            };

            let att = format!("blocks.{layer}.att");
            let att = Att {
                time_decay: load_tensor_exp_f32(format!("{att}.time_decay"))?,
                time_first: load_tensor_f32(format!("{att}.time_first"))?,
                time_mix_k: load_tensor_f32(format!("{att}.time_mix_k"))?,
                time_mix_v: load_tensor_f32(format!("{att}.time_mix_v"))?,
                time_mix_r: load_tensor_f32(format!("{att}.time_mix_r"))?,
                dims: create_uniform_u32(&[num_emb as u32, num_emb as u32]),
                w_k: load_tensor_f16(format!("{att}.key.weight"))?,
                w_v: load_tensor_f16(format!("{att}.value.weight"))?,
                w_r: load_tensor_f16(format!("{att}.receptance.weight"))?,
                w_o: load_tensor_f16(format!("{att}.output.weight"))?,
            };

            let ffn_layer_norm = LayerNorm {
                w: load_tensor_f32(format!("blocks.{layer}.ln2.weight"))?,
                b: load_tensor_f32(format!("blocks.{layer}.ln2.bias"))?,
            };

            let ffn = format!("blocks.{layer}.ffn");
            let ffn = Ffn {
                time_mix_k: load_tensor_f32(format!("{ffn}.time_mix_k"))?,
                time_mix_r: load_tensor_f32(format!("{ffn}.time_mix_r"))?,
                dims_k: create_uniform_u32(&[num_emb as u32, 4 * num_emb as u32]),
                dims_v: create_uniform_u32(&[4 * num_emb as u32, num_emb as u32]),
                dims_r: create_uniform_u32(&[num_emb as u32, num_emb as u32]),
                w_k: load_tensor_f16(format!("{ffn}.key.weight"))?,
                w_v: load_tensor_f16(format!("{ffn}.value.weight"))?,
                w_r: load_tensor_f16(format!("{ffn}.receptance.weight"))?,
            };

            layers.push(Layer {
                att_layer_norm,
                ffn_layer_norm,
                att,
                ffn,
            });
        }

        let dim = create_uniform_u32(&[num_emb as u32]);
        let tensor = Arc::new(ModelTensor {
            dim,
            embed,
            head,
            layers,
        });

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

        let pipeline = Arc::new(ModelPipeline {
            layer_norm: create_pipeline(include_str!("shaders/layer_norm.wgsl"), "layer_norm"),
            token_shift: create_pipeline(include_str!("shaders/token_shift.wgsl"), "token_shift"),
            matmul: create_pipeline(include_str!("shaders/matmul.wgsl"), "matmul"),
            token_mix: create_pipeline(include_str!("shaders/token_mix.wgsl"), "token_mix"),
            activation: create_pipeline(include_str!("shaders/activation.wgsl"), "activation"),
            channel_mix: create_pipeline(include_str!("shaders/channel_mix.wgsl"), "channel_mix"),
            add: create_pipeline(include_str!("shaders/add.wgsl"), "add"),
        });

        Ok(Self {
            env,
            info,
            tensor,
            pipeline,
        })
    }

    pub fn create_buffer(&self, tokens: &[u16]) -> ModelBuffer {
        let device = &self.env.device;

        let create_buffer_f32 = |capacity: usize| -> Buffer {
            let data = vec![0.0f32; capacity];
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(&data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };
        let load_buffer_f32 = |data: &[f32]| -> Buffer {
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(data),
                usage: BufferUsages::STORAGE,
            })
        };
        let create_uniform_u32 = |values: &[u32]| -> Buffer {
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(values),
                usage: BufferUsages::UNIFORM,
            })
        };

        let num_tokens = tokens.len();
        let num_emb = self.info.num_emb;
        let num_vocab = self.info.num_vocab;
        let capacity = num_tokens * num_emb;

        let input = {
            let mut input = vec![];
            input.reserve(capacity);
            for token in tokens {
                let index = *token as usize;
                let begin = index * num_emb;
                let end = begin + num_emb;
                let mut embed: Vec<_> = self.tensor.embed.w[begin..end]
                    .iter()
                    .copied()
                    .map(f16::to_f32)
                    .collect();
                input.append(&mut embed);
            }
            input
        };

        let map = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4 * num_emb as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        ModelBuffer {
            num_tokens: create_uniform_u32(&[num_tokens as u32]),
            emb_x: load_buffer_f32(&input),
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

    pub fn create_state(&self) -> ModelState {
        let device = &self.env.device;

        let ModelInfo {
            num_layers,
            num_emb,
            ..
        } = self.info;

        let create_buffer_f32 = |value: f32| -> Buffer {
            let data = vec![value; num_emb];
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: cast_slice(&data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };

        let mut layers = vec![];
        for _ in 0..num_layers {
            let layer = LayerState {
                att_x: create_buffer_f32(0.0),
                att_a: create_buffer_f32(0.0),
                att_b: create_buffer_f32(0.0),
                att_p: create_buffer_f32(-1.0e30),
                ffn_x: create_buffer_f32(0.0),
            };
            layers.push(layer);
        }

        ModelState(layers)
    }

    pub fn create_bind_group(&self, buffer: &ModelBuffer, state: &ModelState) -> ModelBindGroup {
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
            .map(|pipeline| pipeline.get_bind_group_layout(1));

        let embed = {
            let layer_norm = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &layer_norm_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: buffer.emb_x.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.tensor.embed.layer_norm.w.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.tensor.embed.layer_norm.b.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
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
                        resource: buffer.head_x.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.tensor.head.layer_norm.w.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.tensor.head.layer_norm.b.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: buffer.head_r.as_entire_binding(),
                    },
                ],
            });
            let matmul = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &matmul_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.tensor.head.dims.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.tensor.head.w.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: buffer.head_r.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: buffer.head_o.as_entire_binding(),
                    },
                ],
            });
            HeadBindGroup { layer_norm, matmul }
        };

        let layers = (0..self.info.num_layers)
            .map(|layer| {
                let state = &state.0[layer];
                let layer = &self.tensor.layers[layer];

                let att_layer_norm = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &layer_norm_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: buffer.emb_o.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.att_layer_norm.w.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: layer.att_layer_norm.b.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
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
                            resource: layer.att.time_mix_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: state.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
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
                            resource: layer.att.time_mix_v.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: state.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
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
                            resource: layer.att.time_mix_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: state.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
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
                            resource: buffer.num_tokens.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.att.time_decay.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: layer.att.time_first.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: buffer.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: buffer.att_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: buffer.att_v.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 6,
                            resource: buffer.att_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 7,
                            resource: state.att_a.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 8,
                            resource: state.att_b.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 9,
                            resource: state.att_p.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 10,
                            resource: state.att_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 11,
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
                            resource: buffer.emb_o.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
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
                            resource: buffer.att_o.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: layer.ffn_layer_norm.w.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: layer.ffn_layer_norm.b.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
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
                            resource: layer.ffn.time_mix_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.ffn_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: state.ffn_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
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
                            resource: layer.ffn.time_mix_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.ffn_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: state.ffn_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
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
                let ffn_squared_relu = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &activation_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: buffer.ffn_k.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
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
                            resource: buffer.ffn_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.ffn_r.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: buffer.ffn_v.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: state.ffn_x.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
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
                            resource: buffer.att_o.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
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
                    ffn_squared_relu,
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
}
