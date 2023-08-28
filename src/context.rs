use std::{borrow::Cow, collections::HashMap, str::FromStr, sync::Arc};

use web_rwkv_derive::{Deref, DerefMut, Id};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, Backends, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer, BufferUsages,
    ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Dx12Compiler,
    InstanceDescriptor, PipelineLayoutDescriptor, PowerPreference, Queue, RequestAdapterOptions,
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
    pub const BACKENDS: Backends = Backends::PRIMARY;

    pub fn new() -> Self {
        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: Self::BACKENDS,
            dx12_shader_compiler: Dx12Compiler::Dxc {
                dxil_path: None,
                dxc_path: None,
            },
        });
        Self(instance)
    }

    pub fn adapters(&self) -> Vec<String> {
        self.enumerate_adapters(Self::BACKENDS)
            .map(|adapter| {
                let info = adapter.get_info();
                format!("{} ({:?})", info.name, info.backend)
            })
            .collect()
    }

    pub fn select_adapter(&self, selection: usize) -> Result<Adapter, CreateEnvironmentError> {
        self.enumerate_adapters(Self::BACKENDS)
            .nth(selection)
            .ok_or(CreateEnvironmentError::RequestAdapterFailed)
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
pub struct Context {
    pub id: ContextId,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,

    pipelines: HashMap<String, ComputePipeline>,
    shapes: ResourceCache<Shape, Buffer>,
    views: ResourceCache<View, Buffer>,
}

pub struct ContextBuilder<'a> {
    adapter: Adapter,
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
        }
    }

    pub async fn build(self) -> Result<Context, CreateEnvironmentError> {
        let (device, queue) = self
            .adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
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
                (String::from_str(name).expect("Bad pipeline name"), pipeline)
            })
            .collect();
        Ok(Context {
            id: ContextId::new(),
            adapter: self.adapter,
            device,
            queue,
            pipelines,
            shapes: Default::default(),
            views: Default::default(),
        })
    }

    pub fn with_pipeline(
        self,
        name: &'a str,
        shader: &'a str,
        entry_point: &'a str,
        layout: Option<&'a [BindGroupLayoutEntry]>,
    ) -> Self {
        let Self {
            adapter,
            mut pipelines,
        } = self;
        pipelines.insert(name, (shader, entry_point, layout));
        Self { adapter, pipelines }
    }

    pub fn with_default_pipelines(self) -> Self {
        self.with_pipeline(
            "layer_norm",
            include_str!("shaders/layer_norm.wgsl"),
            "layer_norm",
            None,
        )
        .with_pipeline(
            "matmul",
            include_str!("shaders/matmul.wgsl"),
            "matmul",
            None,
        )
        .with_pipeline(
            "matmul_int8",
            include_str!("shaders/matmul_int8.wgsl"),
            "matmul",
            None,
        )
        .with_pipeline(
            "token_shift",
            include_str!("shaders/token_shift.wgsl"),
            "token_shift",
            None,
        )
        .with_pipeline(
            "token_mix",
            include_str!("shaders/token_mix.wgsl"),
            "token_mix",
            None,
        )
        .with_pipeline("add", include_str!("shaders/add.wgsl"), "add", None)
        .with_pipeline(
            "squared_relu",
            include_str!("shaders/squared_relu.wgsl"),
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
        .with_pipeline("blit", include_str!("shaders/blit.wgsl"), "blit", None)
    }

    pub fn with_quant_pipelines(self) -> Self {
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

        self.with_pipeline("quant_mat_int8", shader, "quantize", layout)
            .with_pipeline("quant_mat_int8_mx", shader, "compute_mx", layout)
            .with_pipeline("quant_mat_int8_my", shader, "compute_my", layout)
            .with_pipeline("quant_mat_int8_rx", shader, "compute_rx", layout)
            .with_pipeline("quant_mat_int8_ry", shader, "compute_ry", layout)
            .with_pipeline(
                "quant_vec_fp16",
                include_str!("shaders/quant_vec_fp16.wgsl"),
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
        self.shapes.request(shape, || {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: &shape.into_bytes(),
                usage: BufferUsages::UNIFORM,
            })
        })
    }

    pub fn request_view_uniform(&self, view: View) -> Arc<Buffer> {
        self.views.request(view, || {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: &view.into_bytes(),
                usage: BufferUsages::UNIFORM,
            })
        })
    }
}
