use std::{borrow::Cow, collections::HashMap, str::FromStr, sync::Arc};

use web_rwkv_derive::{Deref, DerefMut, Id};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, Backends, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer, BufferUsages,
    ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Features, Limits,
    PipelineLayoutDescriptor, PowerPreference, Queue, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderStages,
};

use crate::tensor::{
    cache::ResourceCache,
    shape::{IntoBytes, Shape},
    TensorError, View,
};

#[derive(Deref)]
pub struct Instance(wgpu::Instance);

impl Default for Instance {
    fn default() -> Self {
        Self::new()
    }
}

impl Instance {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(Default::default());
        Self(instance)
    }

    pub fn select_adapter(
        &self,
        backends: Backends,
        selection: usize,
    ) -> Result<Adapter, CreateEnvironmentError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.enumerate_adapters(backends)
                .nth(selection)
                .ok_or(CreateEnvironmentError::RequestAdapterFailed)
        }
        #[cfg(target_arch = "wasm32")]
        {
            unimplemented!()
        }
    }

    pub async fn adapter(
        &self,
        power_preference: PowerPreference,
    ) -> Result<Adapter, CreateEnvironmentError> {
        self.request_adapter(&RequestAdapterOptions {
            power_preference,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok_or(CreateEnvironmentError::RequestAdapterFailed)
    }
}

#[derive(Debug, Clone, Copy, Deref, DerefMut, Id, PartialEq, Eq, Hash)]
pub struct ContextId(usize);

#[derive(Debug)]
pub struct ContextInner {
    pub id: ContextId,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,

    pipelines: HashMap<String, ComputePipeline>,

    shape_cache: ResourceCache<Shape, Buffer>,
    view_cache: ResourceCache<View, Buffer>,
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct Context(Arc<ContextInner>);

pub struct ContextBuilder<'a> {
    adapter: Adapter,
    features: Features,
    limits: Limits,
    pipelines: HashMap<&'a str, (&'a str, &'a str, Option<&'a [BindGroupLayoutEntry]>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateEnvironmentError {
    RequestAdapterFailed,
    RequestDeviceFailed,
}

impl std::fmt::Display for CreateEnvironmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CreateEnvironmentError::RequestAdapterFailed => write!(f, "failed to request adaptor"),
            CreateEnvironmentError::RequestDeviceFailed => write!(f, "failed to request device"),
        }
    }
}

impl std::error::Error for CreateEnvironmentError {}

impl<'a> ContextBuilder<'a> {
    pub fn new(adapter: Adapter) -> Self {
        Self {
            adapter,
            pipelines: HashMap::new(),
            features: Features::empty(),
            limits: Default::default(),
        }
    }

    pub async fn build(self) -> Result<Context, CreateEnvironmentError> {
        let (device, queue) = self
            .adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features: self.features,
                    limits: self.limits,
                },
                None,
            )
            .await
            .map_err(|_| CreateEnvironmentError::RequestDeviceFailed)?;
        let pipelines = self
            .pipelines
            .into_iter()
            .map(|(name, (shader, entry_point, layout))| {
                let module = &device.create_shader_module(ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(Cow::from(shader)),
                });
                let layout = layout.map(|entries| {
                    let layout = &device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                        label: None,
                        entries,
                    });
                    device.create_pipeline_layout(&PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[layout],
                        push_constant_ranges: &[],
                    })
                });
                let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some(name),
                    layout: layout.as_ref(),
                    module,
                    entry_point,
                });
                (String::from_str(name).expect("bad pipeline name"), pipeline)
            })
            .collect();
        Ok(Context(
            ContextInner {
                id: ContextId::new(),
                adapter: self.adapter,
                device,
                queue,
                pipelines,
                shape_cache: Default::default(),
                view_cache: Default::default(),
            }
            .into(),
        ))
    }

    pub fn with_limits(self, limits: Limits) -> Self {
        Self { limits, ..self }
    }

    pub fn with_features(self, features: Features) -> Self {
        Self { features, ..self }
    }

    pub fn with_pipeline(
        self,
        name: &'a str,
        shader: &'a str,
        entry_point: &'a str,
        layout: Option<&'a [BindGroupLayoutEntry]>,
    ) -> Self {
        let mut pipelines = self.pipelines;
        pipelines.insert(name, (shader, entry_point, layout));
        Self { pipelines, ..self }
    }

    pub fn with_default_pipelines(self) -> Self {
        self.with_core_pipelines()
            .with_util_pipelines()
            .with_quant_pipelines()
    }

    fn with_core_pipelines(self) -> Self {
        self.with_pipeline(
            "layer_norm",
            include_str!("shaders/layer_norm.wgsl"),
            "layer_norm",
            None,
        )
        .with_pipeline(
            "group_norm",
            include_str!("shaders/group_norm.wgsl"),
            "group_norm",
            None,
        )
        .with_pipeline(
            "matmul_vec_fp16",
            include_str!("shaders/matmul_vec_fp16.wgsl"),
            "matmul",
            None,
        )
        .with_pipeline(
            "matmul_vec_int8",
            include_str!("shaders/matmul_vec_int8.wgsl"),
            "matmul",
            None,
        )
        .with_pipeline(
            "matmul_vec_nf4",
            include_str!("shaders/matmul_vec_nf4.wgsl"),
            "matmul",
            None,
        )
        .with_pipeline(
            "matmul_mat_fp16",
            include_str!("shaders/matmul_mat_fp16.wgsl"),
            "matmul",
            None,
        )
        .with_pipeline(
            "matmul_mat_int8",
            include_str!("shaders/matmul_mat_int8.wgsl"),
            "matmul",
            None,
        )
        .with_pipeline(
            "matmul_mat_nf4",
            include_str!("shaders/matmul_mat_nf4.wgsl"),
            "matmul",
            None,
        )
        .with_pipeline(
            "token_shift_fp16",
            include_str!("shaders/token_shift.wgsl"),
            "token_shift_fp16",
            None,
        )
        .with_pipeline(
            "token_shift_rev_fp16",
            include_str!("shaders/token_shift.wgsl"),
            "token_shift_rev_fp16",
            None,
        )
        .with_pipeline(
            "token_shift_fp32",
            include_str!("shaders/token_shift.wgsl"),
            "token_shift_fp32",
            None,
        )
        .with_pipeline(
            "token_shift_rev_fp32",
            include_str!("shaders/token_shift.wgsl"),
            "token_shift_rev_fp32",
            None,
        )
        .with_pipeline(
            "time_mix_v4",
            include_str!("shaders/time_mix_v4.wgsl"),
            "time_mix",
            None,
        )
        .with_pipeline(
            "time_mix_v5",
            include_str!("shaders/time_mix_v5.wgsl"),
            "time_mix",
            None,
        )
        .with_pipeline(
            "time_mix_v6",
            include_str!("shaders/time_mix_v6.wgsl"),
            "time_mix",
            None,
        )
        .with_pipeline(
            "add_fp32",
            include_str!("shaders/add.wgsl"),
            "add_fp32",
            None,
        )
        .with_pipeline(
            "add_fp16",
            include_str!("shaders/add.wgsl"),
            "add_fp16",
            None,
        )
        .with_pipeline("silu", include_str!("shaders/silu.wgsl"), "silu", None)
        .with_pipeline(
            "tanh",
            include_str!("shaders/activation.wgsl"),
            "activation_tanh",
            None,
        )
        .with_pipeline(
            "stable_exp",
            include_str!("shaders/activation.wgsl"),
            "stable_exp",
            None,
        )
        .with_pipeline(
            "squared_relu",
            include_str!("shaders/activation.wgsl"),
            "squared_relu",
            None,
        )
        .with_pipeline(
            "channel_mix",
            include_str!("shaders/channel_mix.wgsl"),
            "channel_mix",
            None,
        )
        .with_pipeline(
            "softmax",
            include_str!("shaders/softmax.wgsl"),
            "softmax",
            None,
        )
    }

    fn with_util_pipelines(self) -> Self {
        self.with_pipeline("blit", include_str!("shaders/blit.wgsl"), "blit", None)
            .with_pipeline(
                "transpose",
                include_str!("shaders/blit.wgsl"),
                "transpose",
                None,
            )
            .with_pipeline("blend", include_str!("shaders/blend.wgsl"), "blend", None)
            .with_pipeline(
                "blend_lora",
                include_str!("shaders/blend_lora.wgsl"),
                "blend_lora",
                None,
            )
            .with_pipeline(
                "half",
                include_str!("shaders/discount.wgsl"),
                "discount_half",
                None,
            )
    }

    fn with_quant_pipelines(self) -> Self {
        let shader = include_str!("shaders/quant_mat_int8.wgsl");
        let entries = &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        let layout: Option<&[BindGroupLayoutEntry]> = Some(entries);
        let context = self
            .with_pipeline("quant_mat_int8", shader, "quantize", layout)
            .with_pipeline("quant_mat_int8_mx", shader, "compute_mx", layout)
            .with_pipeline("quant_mat_int8_my", shader, "compute_my", layout)
            .with_pipeline("quant_mat_int8_rx", shader, "compute_rx", layout)
            .with_pipeline("quant_mat_int8_ry", shader, "compute_ry", layout);

        let shader = include_str!("shaders/quant_mat_nf4.wgsl");
        let entries = &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        let layout: Option<&[BindGroupLayoutEntry]> = Some(entries);
        let context = context
            .with_pipeline("quant_mat_nf4_absmax", shader, "compute_absmax", layout)
            .with_pipeline("quant_mat_nf4", shader, "quantize", layout);

        context.with_pipeline(
            "quant_fp16",
            include_str!("shaders/quant_fp16.wgsl"),
            "quantize",
            None,
        )
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Context {}

impl Context {
    pub fn pipeline(&self, name: &'static str) -> Result<&ComputePipeline, TensorError> {
        self.pipelines.get(name).ok_or(TensorError::Pipeline(name))
    }

    pub fn request_shape_uniform(&self, shape: Shape) -> Arc<Buffer> {
        self.shape_cache.request(shape, || {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: &shape.into_bytes(),
                usage: BufferUsages::UNIFORM,
            })
        })
    }

    pub fn request_view_uniform(&self, view: View) -> Arc<Buffer> {
        self.view_cache.request(view, || {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: &view.into_bytes(),
                usage: BufferUsages::UNIFORM,
            })
        })
    }
}
