use std::sync::Arc;
use wgpu::{Adapter, Backends, Device, DeviceDescriptor, Dx12Compiler, InstanceDescriptor, Queue};

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

#[derive(Debug, Clone)]
pub struct Context {
    pub id: uid::Id<ContextId>,
    pub adapter: Arc<Adapter>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
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
            id: uid::Id::new(),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }

    fn ne(&self, other: &Self) -> bool {
        self.id != other.id
    }
}

impl Eq for Context {}
