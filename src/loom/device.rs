use std::sync::Arc;

use crate::num::Scalar;

pub trait Device {
    /// Type of buffer on the device.
    type Data: ?Sized;
    /// Extra parameters for buffer allocation.
    type Params;

    /// Allocate buffer.
    fn alloc<T: Scalar>(&self, len: usize, params: Self::Params) -> Arc<Self::Data>;
}

#[derive(Debug, Clone)]
pub struct Cpu;

impl Device for Cpu {
    type Data = [u8];
    type Params = ();

    #[inline]
    fn alloc<T: Scalar>(&self, len: usize, _params: Self::Params) -> Arc<Self::Data> {
        let data = vec![T::zero(); len].into_boxed_slice();
        let data = Box::leak(data);
        unsafe {
            let data = bytemuck::cast_slice_mut(data);
            Arc::from_raw(data)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Device for Gpu {
    type Data = wgpu::Buffer;
    type Params = wgpu::BufferUsages;

    #[inline]
    fn alloc<T: Scalar>(&self, len: usize, params: Self::Params) -> Arc<Self::Data> {
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
