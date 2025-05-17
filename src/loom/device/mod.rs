use std::sync::Arc;

use crate::loom::num::Scalar;

pub mod cpu;
pub mod gpu;

pub use cpu::Cpu;
pub use gpu::Gpu;

#[cfg_attr(not(target_arch = "wasm32"), trait_variant::make(Send))]
pub trait Device: Sized {
    /// Type of buffer on the device.
    type Data: ?Sized;
    /// Extra parameters for buffer allocation.
    type Params;

    /// Allocate empty buffer on the device.
    async fn alloc<T: Scalar>(&self, len: usize, params: Self::Params) -> Arc<Self::Data>;

    /// Allocate buffer with data.
    async fn create<T: Scalar>(&self, data: &[T], params: Self::Params) -> Arc<Self::Data>;

    /// Free the allocated data. The device would potentially recycle it for future use.
    #[inline]
    fn dealloc(&self, _data: Arc<Self::Data>) {}

    /// Read back a buffer.
    async fn read<T: Scalar>(&self, source: &Self::Data) -> Box<[T]>;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId;
