use anyhow::Result;
use std::sync::Arc;
use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, Dx12Compiler, Instance, InstanceDescriptor, Queue,
    RequestAdapterOptions,
};

#[derive(Clone)]
pub struct Environment {
    pub instance: Arc<Instance>,
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

impl Environment {
    pub async fn create() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            dx12_shader_compiler: Dx12Compiler::Dxc {
                dxil_path: None,
                dxc_path: None,
            },
        });
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .ok_or(CreateEnvironmentError::RequestAdapterFailed)?;
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
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}
