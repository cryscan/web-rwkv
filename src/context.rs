use derive_getters::Getters;
use std::{
    borrow::Cow,
    collections::HashMap,
    str::FromStr,
    sync::{Arc, RwLock},
};
use uid::Id;
use wgpu::{
    Adapter, Backends, BindGroupLayoutDescriptor, BindGroupLayoutEntry, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Dx12Compiler, InstanceDescriptor,
    PipelineLayout, PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderStages,
};

#[derive(Clone)]
pub struct Instance(pub Arc<wgpu::Instance>);

impl std::ops::Deref for Instance {
    type Target = wgpu::Instance;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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
        Self(Arc::new(instance))
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextId(usize);

#[derive(Debug, Clone, Getters)]
pub struct Context {
    pub(crate) id: Id<ContextId>,
    pub(crate) adapter: Arc<Adapter>,
    pub(crate) device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) pipelines: Arc<RwLock<HashMap<String, ComputePipeline>>>,
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

impl Context {
    pub async fn new(adapter: Adapter) -> Result<Self, CreateEnvironmentError> {
        let (device, queue) = adapter
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

        Ok(Self {
            id: Id::new(),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines: Default::default(),
        })
    }

    pub fn add_pipeline(&mut self, name: &str, shader: &str, layout: Option<&PipelineLayout>) {
        let module = &self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(Cow::from(shader)),
        });
        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(name),
                layout,
                module,
                entry_point: name,
            });

        let mut pipelines = self.pipelines.write().unwrap();
        pipelines.insert(String::from_str(name).expect("Bad pipeline name"), pipeline);
    }

    pub fn with_pipeline(
        mut self,
        name: &str,
        shader: &str,
        layout: Option<&PipelineLayout>,
    ) -> Self {
        self.add_pipeline(name, shader, layout);
        self
    }

    pub fn with_default_pipelines(self) -> Self {
        self.with_pipeline("layer_norm", include_str!("shaders/layer_norm.wgsl"), None)
            .with_pipeline("matmul", include_str!("shaders/matmul.wgsl"), None)
            .with_pipeline(
                "matmul_int8",
                include_str!("shaders/matmul_int8.wgsl"),
                None,
            )
            .with_pipeline(
                "token_shift",
                include_str!("shaders/token_shift.wgsl"),
                None,
            )
            .with_pipeline("token_mix", include_str!("shaders/token_mix.wgsl"), None)
            .with_pipeline("add", include_str!("shaders/add.wgsl"), None)
            .with_pipeline(
                "squared_relu",
                include_str!("shaders/squared_relu.wgsl"),
                None,
            )
            .with_pipeline(
                "channel_mix",
                include_str!("shaders/channel_mix.wgsl"),
                None,
            )
            .with_pipeline("softmax", include_str!("shaders/softmax.wgsl"), None)
    }

    pub fn with_quantize_pipelines(self) -> Self {
        let shader = include_str!("shaders/quantize_int8.wgsl");
        let bind_group_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
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
                ],
            });
        let layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let layout = Some(&layout);

        self.with_pipeline("quantize", shader, layout)
            .with_pipeline("compute_mx", shader, layout)
            .with_pipeline("compute_my", shader, layout)
            .with_pipeline("compute_rx", shader, layout)
            .with_pipeline("compute_ry", shader, layout)
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Context {}
