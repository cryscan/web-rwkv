use std::sync::Arc;

use super::{Device, DeviceId};
use crate::num::Scalar;

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
        bytemuck::cast_slice_box(data).into()
    }

    #[inline]
    async fn create<T: Scalar>(&self, data: &[T], _params: Self::Params) -> Arc<Self::Data> {
        bytemuck::cast_vec(data.to_vec()).into()
    }

    #[inline]
    async fn read<T: Scalar>(&self, source: &<Self as Device>::Data) -> Box<[T]> {
        bytemuck::cast_slice(source).to_vec().into_boxed_slice()
    }
}

impl Cpu {
    pub fn new() -> Self {
        Self(uid::Id::new())
    }
}
