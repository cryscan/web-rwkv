use thiserror::Error;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::num::Scalar;

pub trait Device: Send + Sync {
    /// Type of buffer on the device.
    type Data: ?Sized;
    /// Extra parameters for buffer allocation.
    type Params;

    /// Allocate buffer on the device.
    fn alloc<T: Scalar>(&self, len: usize, params: Self::Params) -> Box<Self::Data>;
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
    fn alloc<T: Scalar>(&self, len: usize, _params: Self::Params) -> Box<Self::Data> {
        let data = vec![T::zero(); len].into_boxed_slice();
        let data = Box::leak(data);
        unsafe {
            let data = bytemuck::cast_slice_mut(data);
            Box::from_raw(data)
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
    fn alloc<T: Scalar>(&self, len: usize, params: Self::Params) -> Box<Self::Data> {
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
}

#[cfg(not(target_arch = "wasm32"))]
pub struct GpuEvent {
    pub buffer: Box<<Gpu as Device>::Data>,
    pub sender: flume::Sender<Box<<Cpu as Device>::Data>>,
}

#[derive(Debug, Clone)]
pub struct GpuBuilder {
    pub adapter: wgpu::Adapter,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
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
        }
    }

    pub async fn build(self) -> Result<Gpu, GpuBuildError> {
        let Self {
            adapter,
            features,
            limits,
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

        // start a thread for reading back buffers
        #[cfg(not(target_arch = "wasm32"))]
        {
            let id = device.id;
            let device = device.device.clone();
            std::thread::spawn(move || {
                while let Ok(GpuEvent { buffer, sender }) = receiver.recv() {
                    #[cfg(feature = "trace")]
                    let _span = tracing::trace_span!("device").entered();
                    let data = read_back_buffer(&device, &buffer);
                    let _ = sender.send(data);
                }
                log::info!("device dropped: {id}");
            });
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
