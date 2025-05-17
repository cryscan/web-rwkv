use std::{
    fmt::{Debug, Formatter},
    marker::PhantomData,
    sync::Arc,
};

use casey::snake;
use derive_more::{Deref, DerefMut, Display, From, Into};
use itertools::Itertools;
use thiserror::Error;

use super::{
    device::{Cpu, Device, Gpu},
    layout::Layout,
};
use crate::loom::num::Scalar;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("tensor creation error: layout {0}'s size not match data len {1}")]
    Create(Layout, usize),
    #[error("tensor reshape error: layout {0}'s size not match layout {1}'s")]
    Reshape(Layout, Layout),
    #[error("tensor slice error: slice {1} is not compatible with layout {0}")]
    Slice(Layout, Slice),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId;

pub struct Tensor<D: Device, T: Scalar> {
    device: D,
    layout: Layout,
    data: Arc<D::Data>,
    id: uid::Id<TensorId>,
    phantom: PhantomData<T>,
}

impl<D: Device + Debug, T: Scalar> Debug for Tensor<D, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("device", &self.device)
            .field("layout", &self.layout)
            .field("id", &self.id)
            .finish()
    }
}

impl<D: Device + Clone, T: Scalar> Clone for Tensor<D, T> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            layout: self.layout.clone(),
            data: self.data.clone(),
            id: self.id,
            phantom: PhantomData,
        }
    }
}

impl<D: Device, T: Scalar> PartialEq for Tensor<D, T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<D: Device, T: Scalar> Eq for Tensor<D, T> {}

impl<D: Device, T: Scalar> Drop for Tensor<D, T> {
    fn drop(&mut self) {
        match Arc::strong_count(&self.data) {
            0 | 1 => self.device.dealloc(self.data.clone()),
            _ => (),
        }
    }
}

impl<T: Scalar> Tensor<Cpu, T> {
    #[inline]
    pub fn data(&self) -> &[T] {
        bytemuck::cast_slice(&self.data)
    }
}

impl<T: Scalar> Tensor<Gpu, T> {
    pub const PARAMS: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
        .union(wgpu::BufferUsages::COPY_DST)
        .union(wgpu::BufferUsages::COPY_SRC);
}

#[cfg_attr(not(target_arch = "wasm32"), trait_variant::make(Send))]
pub trait TensorInit<D: Device, T: Scalar>: Sized {
    /// Init a tensor of zeros.
    async fn zeros(device: D, layout: Layout) -> Self;
    /// Create a tensor from data.
    async fn create(device: D, layout: Layout, data: &[T]) -> Result<Self, TensorError>;
}

impl<T: Scalar> TensorInit<Cpu, T> for Tensor<Cpu, T> {
    async fn zeros(device: Cpu, layout: Layout) -> Self {
        let data = device.alloc::<T>(layout.size(), ()).await;
        let id = uid::Id::new();
        let phantom = PhantomData;
        Self {
            device,
            layout,
            data,
            id,
            phantom,
        }
    }

    async fn create(device: Cpu, layout: Layout, data: &[T]) -> Result<Self, TensorError> {
        if layout.size() != data.len() {
            return Err(TensorError::Create(layout, data.len()));
        }
        let data = device.create(data, ()).await;
        let id = uid::Id::new();
        let phantom = PhantomData;
        Ok(Self {
            device,
            layout,
            data,
            id,
            phantom,
        })
    }
}

impl<T: Scalar> TensorInit<Gpu, T> for Tensor<Gpu, T> {
    async fn zeros(device: Gpu, layout: Layout) -> Self {
        let data = device.alloc::<T>(layout.size(), Self::PARAMS).await;
        let id = uid::Id::new();
        let phantom = PhantomData;
        Self {
            device,
            layout,
            data,
            id,
            phantom,
        }
    }

    async fn create(device: Gpu, layout: Layout, data: &[T]) -> Result<Self, TensorError> {
        if layout.size() != data.len() {
            return Err(TensorError::Create(layout, data.len()));
        }
        let data = device.create(data, Self::PARAMS).await;
        let id = uid::Id::new();
        let phantom = PhantomData;
        Ok(Self {
            device,
            layout,
            data,
            id,
            phantom,
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), trait_variant::make(Send))]
pub trait TensorTo<D: Device, T: Scalar> {
    /// Send a tensor to another device.
    async fn to(self, device: D) -> Tensor<D, T>;
}

impl<T: Scalar> TensorTo<Cpu, T> for Tensor<Cpu, T> {
    #[inline]
    async fn to(self, _device: Cpu) -> Tensor<Cpu, T> {
        self
    }
}

impl<T: Scalar> TensorTo<Gpu, T> for Tensor<Cpu, T> {
    #[inline]
    async fn to(self, device: Gpu) -> Tensor<Gpu, T> {
        let data = device.create(self.data(), Tensor::<Gpu, T>::PARAMS).await;
        let layout = self.layout.clone();
        let id = uid::Id::new();
        let phantom = PhantomData;
        Tensor {
            device,
            layout,
            data,
            id,
            phantom,
        }
    }
}

impl<T: Scalar> TensorTo<Cpu, T> for Tensor<Gpu, T> {
    #[inline]
    async fn to(self, device: Cpu) -> Tensor<Cpu, T> {
        let data = self.device.read(&self.data).await.into();
        let layout = self.layout.clone();
        let id = uid::Id::new();
        let phantom = PhantomData;
        Tensor {
            device,
            layout,
            data,
            id,
            phantom,
        }
    }
}

impl<T: Scalar> TensorTo<Gpu, T> for Tensor<Gpu, T> {
    #[inline]
    async fn to(self, device: Gpu) -> Tensor<Gpu, T> {
        if self.device == device {
            return self;
        }
        let cpu: Tensor<Cpu, T> = self.to(Cpu::new()).await;
        cpu.to(device).await
    }
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout.clone()
    }

    /// Reshape the tensor, leaving the underlying data untouched.
    #[inline]
    pub fn reshape(mut self, layout: Layout) -> Result<Self, TensorError> {
        if self.layout.size() != layout.size() {
            return Err(TensorError::Reshape(self.layout(), layout));
        }
        self.layout = layout;
        self.id = uid::Id::new();
        Ok(self)
    }

    /// Create a [`TensorSlice`] from the tensor.
    #[inline]
    pub fn slice(self, slice: Slice) -> Result<TensorSlice<D, T>, TensorError> {
        if slice.len() != self.layout.len() {
            return Err(TensorError::Slice(self.layout(), slice));
        }
        if slice
            .iter()
            .zip_eq(self.layout.shape().iter())
            .filter_map(|(&axis, &shape)| match axis {
                Axis::Full => None,
                Axis::One(index) => Some((index, shape)),
            })
            .any(|(index, shape)| index >= shape)
        {
            return Err(TensorError::Slice(self.layout(), slice));
        }
        let tensor = self;
        Ok(TensorSlice { tensor, slice })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Axis {
    #[default]
    #[display("..")]
    Full,
    #[display("{_0}")]
    One(usize),
}

impl From<usize> for Axis {
    #[inline]
    fn from(value: usize) -> Self {
        Self::One(value)
    }
}

impl From<std::ops::RangeFull> for Axis {
    #[inline]
    fn from(_: std::ops::RangeFull) -> Self {
        Self::Full
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deref, DerefMut, From, Into, Display)]
#[display("[{}]", _0.iter().format(", "))]
pub struct Slice(Arc<[Axis]>);

impl From<Vec<Axis>> for Slice {
    #[inline]
    fn from(value: Vec<Axis>) -> Self {
        Self(value.into())
    }
}

macro_rules! impl_slice_from {
    ($t:ident) => {
        impl<$t: Into<Axis>> From<$t> for Slice {
            #[inline]
            fn from(snake!($t): $t) -> Self {
                Self([snake!($t).into()].into())
            }
        }
    };
    ($($t:ident),+) => {
        impl<$($t),+> From<($($t),+)> for Slice
        where
            $($t: Into<Axis>),+
        {
            #[inline]
            fn from(($(snake!($t)),+): ($($t),+)) -> Self {
                Self([$(snake!($t).into()),+].into())
            }
        }
    };
}

impl_slice_from!(T0);
impl_slice_from!(T0, T1);
impl_slice_from!(T0, T1, T2);
impl_slice_from!(T0, T1, T2, T3);
impl_slice_from!(T0, T1, T2, T3, T4);
impl_slice_from!(T0, T1, T2, T3, T4, T5);
impl_slice_from!(T0, T1, T2, T3, T4, T5, T6);
impl_slice_from!(T0, T1, T2, T3, T4, T5, T6, T7);

impl Slice {
    /// Creates a full slice of the same mode as a `Layout`.
    #[inline]
    pub fn from_layout(layout: Layout) -> Self {
        Self::from(vec![Axis::Full; layout.len()])
    }

    /// Returns `true` if the slice contains only full axes.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.iter().all(|&axis| matches!(axis, Axis::Full))
    }
}

#[derive(Debug, Clone, Deref, PartialEq, Eq)]
pub struct TensorSlice<D: Device, T: Scalar> {
    #[deref]
    tensor: Tensor<D, T>,
    slice: Slice,
}

impl<D: Device, T: Scalar> TensorSlice<D, T> {
    #[inline]
    pub fn slice(&self) -> Slice {
        self.slice.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::Slice;

    #[test]
    fn test_slice() {
        println!("{}", Slice::from(1));
        println!("{}", Slice::from(..));
        println!("{}", Slice::from((0, 1)));
        println!("{}", Slice::from((1, ..)));
        println!("{}", Slice::from((0, .., 1, 5)));
    }
}
