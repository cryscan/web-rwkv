use std::sync::Arc;

use crate::{future::Future, num::Scalar};

pub mod cpu;
pub mod gpu;

pub use cpu::Cpu;
pub use gpu::Gpu;

pub trait Device {
    /// Type of buffer on the device.
    type Data: ?Sized;
    /// Extra parameters for buffer allocation.
    type Params;

    /// Allocate empty buffer on the device.
    fn alloc<T: Scalar>(&self, len: usize, params: Self::Params) -> impl Future<Arc<Self::Data>>;

    /// Allocate buffer with data.
    fn create<T: Scalar>(&self, data: &[T], params: Self::Params) -> impl Future<Arc<Self::Data>>;

    /// Free the allocated data. The device would potentially recycle it for future use.
    #[inline]
    fn dealloc(&self, _data: Arc<Self::Data>) {}

    /// Read back a buffer.
    fn read<T: Scalar>(&self, source: &Self::Data) -> impl Future<Box<[T]>>;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId;
