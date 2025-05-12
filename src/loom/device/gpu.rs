use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;
use thiserror::Error;
use wgpu::util::DeviceExt;

use super::{Cpu, Device, DeviceId};
use crate::num::Scalar;

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
    async fn create<T: Scalar>(&self, data: &[T], params: Self::Params) -> Arc<Self::Data> {
        #[cfg(feature = "trace")]
        let _span = tracing::trace_span!("create").entered();
        let data = bytemuck::cast_slice(data);
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: params,
            })
            .into()
    }

    #[inline]
    fn dealloc(&self, data: Arc<Self::Data>) {
        let _ = self.event.send(GpuEvent::Dealloc(data));
    }

    /// Reads back a buffer with [`STORAGE`](wgpu::BufferUsages::STORAGE) and [`COPY_SRC`](wgpu::BufferUsages::COPY_SRC) usages.
    async fn read<T: Scalar>(&self, source: &<Self as Device>::Data) -> Box<[T]> {
        let len = source.size() as usize / size_of::<T>();
        let size = (len * size_of::<T>()) as u64;
        let params = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let buffer = self.alloc::<T>(len, params).await;

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = flume::bounded(1);
        let _ = self.event.send(GpuEvent::Read { buffer, sender });

        let data = receiver
            .recv_async()
            .await
            .expect("failed to receive read back buffer");
        let data = bytemuck::cast_slice_mut(Box::leak(data));
        unsafe { Box::from_raw(data) }
    }
}

pub enum GpuEvent {
    Read {
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
}

#[derive(Debug, Error)]
pub enum GpuBuildError {
    #[error("failed to request adaptor")]
    RequestAdapterError(#[from] wgpu::RequestAdapterError),
    #[error("failed to request device")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
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
        }
    }

    pub async fn build(&mut self) -> Result<Gpu, GpuBuildError> {
        let Self {
            adapter,
            features,
            limits,
        } = self.clone();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;

        let id = uid::Id::new();
        let (event, receiver) = flume::unbounded();

        let device = Gpu {
            id,
            device,
            queue,
            event,
        };

        {
            let id = device.id;
            let device = device.device.clone();
            let receiver = receiver.clone();

            #[cfg(not(target_arch = "wasm32"))]
            tokio::spawn(handle_buffer_events(id, device, receiver));

            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(handle_buffer_events(id, device, receiver));
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
}

async fn read_back(device: wgpu::Device, buffer: Arc<wgpu::Buffer>) -> Box<[u8]> {
    let (sender, receiver) = flume::bounded(1);
    buffer.clone().map_async(wgpu::MapMode::Read, .., move |v| {
        if let Err(err) = v {
            let _ = sender.send(Err(err));
            return;
        }
        let data = buffer.get_mapped_range(..).to_vec().into_boxed_slice();
        buffer.unmap();
        let _ = sender.send(Ok(data));
    });

    #[cfg(not(target_arch = "wasm32"))]
    tokio::task::spawn_blocking(move || device.poll(wgpu::MaintainBase::Wait));
    #[cfg(target_arch = "wasm32")]
    let _ = device.poll(wgpu::MaintainBase::Wait);

    receiver
        .recv_async()
        .await
        .expect("failed to receive read back buffer")
        .expect("failed to map buffer")
}

async fn handle_buffer_events(
    id: uid::Id<DeviceId>,
    device: wgpu::Device,
    receiver: flume::Receiver<GpuEvent>,
) {
    let mut cache = HashMap::default();
    while let Ok(event) = receiver.recv_async().await {
        match event {
            GpuEvent::Read { buffer, sender } => {
                #[cfg(feature = "trace")]
                let _span = tracing::trace_span!("read").entered();
                let device = device.clone();
                let data = read_back(device, buffer).await;
                let _ = sender.send(data);
            }
            GpuEvent::Alloc {
                size,
                params,
                sender,
            } => {
                #[cfg(feature = "trace")]
                let _span = tracing::trace_span!("alloc").entered();
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
    use std::sync::Arc;

    use super::{Device, GpuBuilder};

    #[tokio::test]
    async fn test_alloc() -> anyhow::Result<()> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let device = GpuBuilder::new(adapter).build().await?;
        let usages = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

        const LEN: usize = 1024;
        let mut set = tokio::task::JoinSet::new();
        for index in 0..4 {
            let device = device.clone();
            set.spawn(async move {
                let len = LEN as u16;
                let y = (index + 1) as f32;
                let data: Vec<_> = (0..len).map(f32::from).map(|x| x * y).collect();
                let buffer = device.create(&data, usages).await;
                let read = device.read::<f32>(&buffer).await;
                assert_eq!(read.to_vec(), data);
                println!("addr: {:?}, size: {}", Arc::as_ptr(&buffer), buffer.size());
                device.dealloc(buffer);
            });
        }
        set.join_all().await;
        println!();

        for _ in 0..12 {
            let buffer = device.alloc::<f32>(LEN, usages).await;
            let read = device.read::<f32>(&buffer).await;
            println!("addr: {:?}, size: {}", Arc::as_ptr(&buffer), buffer.size());
            assert_eq!(read.len(), LEN);
            device.dealloc(buffer);
        }
        println!();

        let mut set = tokio::task::JoinSet::new();
        for _ in 0..6 {
            let device = device.clone();
            set.spawn(async move {
                let buffer = device.alloc::<f32>(LEN, usages).await;
                let read = device.read::<f32>(&buffer).await;
                println!("addr: {:?}, size: {}", Arc::as_ptr(&buffer), buffer.size());
                assert_eq!(read.len(), LEN);
                device.dealloc(buffer);
            });
        }
        set.join_all().await;
        println!();

        let mut set = tokio::task::JoinSet::new();
        for _ in 0..6 {
            let device = device.clone();
            set.spawn(async move {
                let buffer = device.alloc::<f32>(LEN, usages).await;
                let read = device.read::<f32>(&buffer).await;
                println!("addr: {:?}, size: {}", Arc::as_ptr(&buffer), buffer.size());
                assert_eq!(read.len(), LEN);
                device.dealloc(buffer);
            });
        }
        set.join_all().await;
        println!();

        Ok(())
    }
}
