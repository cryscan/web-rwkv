use std::{collections::HashMap, future::Future, sync::Arc};

use thiserror::Error;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::num::Scalar;

pub trait Device: Send + Sync {
    /// Type of buffer on the device.
    type Data: ?Sized;
    /// Extra parameters for buffer allocation.
    type Params;

    /// Allocate buffer on the device.
    fn alloc<T: Scalar>(
        &self,
        len: usize,
        params: Self::Params,
    ) -> impl Future<Output = Arc<Self::Data>> + Send;

    /// Free the allocated data. The device would potentially recycle it for future use.
    #[inline]
    fn dealloc(&self, _data: Arc<Self::Data>) {}
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId;

/// A CPU device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpu(uid::Id<DeviceId>);

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for Cpu {
    type Data = [u8];
    type Params = ();

    #[inline]
    async fn alloc<T: Scalar>(&self, len: usize, _params: Self::Params) -> Arc<Self::Data> {
        let data = vec![T::zero(); len].into_boxed_slice();
        let data = Box::leak(data);
        unsafe {
            let data = bytemuck::cast_slice_mut(data);
            Arc::from_raw(data)
        }
    }
}

impl Cpu {
    pub fn new() -> Self {
        Self(uid::Id::new())
    }
}

/// A WebGPU device.
#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Gpu {
    /// The unique identifier of the device.
    id: uid::Id<DeviceId>,
    /// Handle to a WebGPU compute device.
    device: wgpu::Device,
    /// The WebGPU command queue.
    queue: wgpu::Queue,
    /// GPU buffer read back events.
    #[cfg(not(target_arch = "wasm32"))]
    event: flume::Sender<GpuEvent>,
}

impl PartialEq for Gpu {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Gpu {}

impl Device for Gpu {
    type Data = wgpu::Buffer;
    type Params = wgpu::BufferUsages;

    #[inline]
    #[cfg(target_arch = "wasm32")]
    async fn alloc<T: Scalar>(&self, len: usize, params: Self::Params) -> Arc<Self::Data> {
        let size = len * size_of::<T>();
        self.device
            .create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size as u64,
                usage: params,
                mapped_at_creation: false,
            })
            .into()
    }

    #[inline]
    #[cfg(not(target_arch = "wasm32"))]
    async fn alloc<T: Scalar>(&self, len: usize, params: Self::Params) -> Arc<Self::Data> {
        let (sender, receiver) = flume::bounded(1);
        let size = (len * size_of::<T>()) as u64;
        let _ = self.event.send(GpuEvent::Alloc {
            size,
            params,
            sender,
        });
        receiver
            .recv_async()
            .await
            .expect("failed to allocate buffer")
    }

    #[inline]
    #[cfg(not(target_arch = "wasm32"))]
    fn dealloc(&self, data: Arc<Self::Data>) {
        let _ = self.event.send(GpuEvent::Dealloc(data));
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub enum GpuEvent {
    Back {
        buffer: Arc<<Gpu as Device>::Data>,
        sender: flume::Sender<Box<<Cpu as Device>::Data>>,
    },
    Alloc {
        size: u64,
        params: <Gpu as Device>::Params,
        sender: flume::Sender<Arc<<Gpu as Device>::Data>>,
    },
    Dealloc(Arc<<Gpu as Device>::Data>),
    Cleanup,
}

#[derive(Debug, Clone)]
pub struct GpuBuilder {
    pub adapter: wgpu::Adapter,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
    #[cfg(not(target_arch = "wasm32"))]
    pub threads: usize,
}

#[wasm_bindgen]
#[derive(Debug, Error)]
pub enum GpuBuildError {
    #[error("failed to request adaptor")]
    RequestAdapterFailed,
    #[error("failed to request device")]
    RequestDeviceFailed,
}

impl GpuBuilder {
    pub fn new(adapter: wgpu::Adapter) -> Self {
        let features = wgpu::Features::empty();
        #[cfg(feature = "subgroup-ops")]
        let features = features | wgpu::Features::SUBGROUP;
        Self {
            adapter,
            features,
            limits: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            threads: 4,
        }
    }

    pub async fn build(self) -> Result<Gpu, GpuBuildError> {
        let Self {
            adapter,
            features,
            limits,
            #[cfg(not(target_arch = "wasm32"))]
            threads,
        } = self;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|_| GpuBuildError::RequestDeviceFailed)?;

        #[cfg(not(target_arch = "wasm32"))]
        let (event, receiver) = flume::unbounded();

        let id = uid::Id::new();
        let device = Gpu {
            id,
            device,
            queue,
            event,
        };

        // start threads for buffer services
        #[cfg(not(target_arch = "wasm32"))]
        for _ in 0..threads {
            let id = device.id;
            let device = device.device.clone();
            let receiver = receiver.clone();
            std::thread::spawn(move || handle_buffer_events(id, device, receiver));
        }

        Ok(device)
    }

    pub fn limits(&mut self, limits: wgpu::Limits) -> &mut Self {
        self.limits = limits;
        self
    }

    pub fn features(&mut self, features: wgpu::Features) -> &mut Self {
        self.features = features;
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn threads(&mut self, threads: usize) -> &mut Self {
        self.threads = threads;
        self
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_back_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Box<[u8]> {
    let (sender, receiver) = flume::bounded(1);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::MaintainBase::Wait);
    receiver
        .recv()
        .expect("failed to receive read back buffer")
        .expect("failed to map buffer");

    let data = {
        let map = slice.get_mapped_range();
        let len = map.len();
        let size = std::mem::size_of::<u32>();
        let data = vec![0u32; len.div_ceil(size)].into_boxed_slice();
        unsafe {
            let data = Box::leak(data);
            let data: &mut [u8] = bytemuck::cast_slice_mut(data);
            data.copy_from_slice(&map);
            Box::from_raw(data)
        }
    };
    buffer.unmap();
    data
}

#[cfg(not(target_arch = "wasm32"))]
fn handle_buffer_events(
    id: uid::Id<DeviceId>,
    device: wgpu::Device,
    receiver: flume::Receiver<GpuEvent>,
) {
    let mut cache = HashMap::new();
    while let Ok(event) = receiver.recv() {
        match event {
            GpuEvent::Back { buffer, sender } => {
                #[cfg(feature = "trace")]
                let _span = tracing::trace_span!("device").entered();
                let data = read_back_buffer(&device, &buffer);
                let _ = sender.send(data);
            }
            GpuEvent::Alloc {
                size,
                params,
                sender,
            } => {
                let buffer = match cache
                    .get_mut(&(size, params))
                    .and_then(|buffers: &mut Vec<_>| buffers.pop())
                {
                    Some(buffer) => buffer,
                    None => device
                        .create_buffer(&wgpu::BufferDescriptor {
                            label: None,
                            size,
                            usage: params,
                            mapped_at_creation: false,
                        })
                        .into(),
                };
                let _ = sender.send(buffer);
            }
            GpuEvent::Dealloc(buffer) => {
                let key = (buffer.size(), buffer.usage());
                let mut buffers = cache.remove(&key).unwrap_or_default();
                buffers.push(buffer);
                cache.insert(key, buffers);
            }
            GpuEvent::Cleanup => cache.clear(),
        }
    }
    log::info!("device dropped: {id}");
}

#[cfg(test)]
mod tests {
    use super::{GpuBuildError, GpuBuilder};

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn test_alloc() -> anyhow::Result<()> {
        use crate::loom::device::Device;

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(GpuBuildError::RequestAdapterFailed)?;

        let device = GpuBuilder::new(adapter).build().await?;
        let buffer = device.alloc::<f32>(1024, wgpu::BufferUsages::STORAGE).await;

        println!("{:?}", buffer);

        Ok(())
    }
}
