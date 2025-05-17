use std::sync::Arc;

use super::{Device, DeviceId};
use crate::loom::num::Scalar;

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
        let data = vec![T::zeroed(); len];
        let data = bytemuck::cast_slice_mut(Vec::leak(data));
        unsafe { Arc::from_raw(data) }
    }

    #[inline]
    async fn create<T: Scalar>(&self, data: &[T], _params: Self::Params) -> Arc<Self::Data> {
        let data = data.to_vec();
        let data = bytemuck::cast_slice_mut(Vec::leak(data));
        unsafe { Arc::from_raw(data) }
    }

    #[inline]
    async fn read<T: Scalar>(&self, source: &<Self as Device>::Data) -> Box<[T]> {
        let data = source.to_vec();
        let data = bytemuck::cast_slice_mut(Vec::leak(data));
        unsafe { Box::from_raw(data) }
    }
}

impl Cpu {
    pub fn new() -> Self {
        Self(uid::Id::new())
    }
}
