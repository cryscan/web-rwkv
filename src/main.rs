use anyhow::Result;
use std::sync::Arc;
use wgpu::{
    Adapter, Backends, Device, Dx12Compiler, Instance, InstanceDescriptor, Queue,
    RequestAdapterOptions,
};

mod model;

pub struct Environment {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
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
        let instance = wgpu::Instance::new(InstanceDescriptor {
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
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 12,
                        ..Default::default()
                    },
                },
                None,
            )
            .await
            .map_err(|_| CreateEnvironmentError::RequestDeviceFailed)?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

async fn run() -> Result<()> {
    let env = Arc::new(Environment::create().await?);

    let model = model::Model::from_file(
        "assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st".into(),
        env,
    )?;
    println!("{:#?}", model.info);

    Ok(())
}

fn main() {
    pollster::block_on(run()).unwrap();
}
