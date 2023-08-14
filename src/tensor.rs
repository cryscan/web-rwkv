use bytemuck::Pod;
use derive_getters::Getters;
use half::prelude::*;
use std::{borrow::Cow, marker::PhantomData, sync::Arc};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoder, MaintainBase, MapMode,
};

use crate::Context;

#[derive(Debug, Clone)]
pub struct BufferView {
    pub buffer: Arc<Buffer>,
    pub offset: BufferAddress,
}

pub trait DeviceKind: sealed::Sealed {
    type Data: Clone;
}

pub struct Cpu<'a, T>(&'a PhantomData<T>);
pub struct Gpu;

impl<'a, T: DataKind> DeviceKind for Cpu<'a, T> {
    type Data = Cow<'a, [T]>;
}

impl DeviceKind for Gpu {
    type Data = BufferView;
}

pub trait DataKind: Sized + Clone + Copy + Pod + sealed::Sealed {
    fn byte_size() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl DataKind for f32 {}
impl DataKind for f16 {}
impl DataKind for u8 {}

/// The shape of a [`Tensor`].
/// Note that the fastest-moving axis occupies the lowest shape index, which is opposite to that in `torch`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorShape([usize; 4]);

impl TensorShape {
    pub fn len(&self) -> usize {
        self.0.into_iter().product()
    }
}

impl std::fmt::Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {}, {})", self[0], self[1], self[2], self[3])
    }
}

impl std::ops::Index<usize> for TensorShape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for TensorShape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TensorError {
    Size(usize, usize),
    Shape(TensorShape, TensorShape),
    Overflow {
        buffer_size: BufferAddress,
        offset: BufferAddress,
        size: BufferAddress,
    },
    DeviceError,
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::Size(a, b) => write!(f, "Data size not match: {} vs. {}", a, b),
            TensorError::Shape(a, b) => write!(f, "Tensor shape not match: {} vs. {}", a, b),
            TensorError::Overflow {
                buffer_size,
                offset,
                size: length,
            } => write!(
                f,
                "Buffer overflow with buffer size: {}, slice offset: {} and size: {}",
                buffer_size, offset, length
            ),
            TensorError::DeviceError => write!(f, "Tensor not on the same device"),
        }
    }
}

impl std::error::Error for TensorError {}

#[derive(Debug, Clone, Getters)]
pub struct Tensor<'a, Device: DeviceKind, T> {
    context: Context,
    shape: TensorShape,
    name: Option<&'a str>,
    data: Device::Data,
    #[getter(skip)]
    phantom: std::marker::PhantomData<(Device, T)>,
}

impl<Device: DeviceKind, T: DataKind> std::ops::Deref for Tensor<'_, Device, T> {
    type Target = Device::Data;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<Device: DeviceKind, T: DataKind> Tensor<'_, Device, T> {
    pub fn byte_size(&self) -> usize {
        self.shape.len() * T::byte_size()
    }

    pub fn byte_offset(offset: usize) -> usize {
        offset * T::byte_size()
    }
}

impl<'a, T: DataKind> TensorCpu<'a, T> {
    pub fn new(
        context: Context,
        shape: TensorShape,
        name: Option<&'a str>,
        data: Vec<T>,
    ) -> Result<Self, TensorError> {
        if shape.len() != data.len() {
            return Err(TensorError::Size(shape.len(), data.len()));
        }
        Ok(Self {
            context,
            shape,
            name,
            data: Cow::from(data),
            phantom: Default::default(),
        })
    }
}

impl<'a, T: DataKind> TensorGpu<'a, T> {
    /// Create a GPU tensor from a [`BufferView`].
    /// Fails if the buffer overflows.
    pub fn new(
        context: Context,
        shape: TensorShape,
        name: Option<&'a str>,
        data: BufferView,
    ) -> Result<Self, TensorError> {
        let size = shape.len() as u64 * T::byte_size() as u64;
        if data.offset + size >= data.buffer.size() {
            return Err(TensorError::Overflow {
                buffer_size: data.buffer.size(),
                offset: data.offset,
                size,
            });
        }
        Ok(Self {
            context,
            shape,
            name,
            data,
            phantom: Default::default(),
        })
    }

    /// Initialize a GPU tensor with a given shape.
    pub fn init(
        context: Context,
        shape: TensorShape,
        name: Option<&'a str>,
        usage: BufferUsages,
    ) -> Self {
        let label = name;
        let size = shape.len() as u64 * T::byte_size() as u64;
        let buffer = context
            .device
            .create_buffer(&BufferDescriptor {
                label,
                size,
                usage,
                mapped_at_creation: false,
            })
            .into();
        Self {
            context,
            shape,
            name,
            data: BufferView { buffer, offset: 0 },
            phantom: Default::default(),
        }
    }
}

impl<'a, T: DataKind> From<TensorCpu<'a, T>> for TensorGpu<'a, T> {
    fn from(value: TensorCpu<'a, T>) -> Self {
        let Tensor {
            context,
            shape,
            name,
            data,
            ..
        } = value;
        let label = name;
        let contents = bytemuck::cast_slice(&data);
        let buffer = context
            .device
            .create_buffer_init(&BufferInitDescriptor {
                label,
                contents,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
            .into();
        Self {
            context,
            shape,
            name,
            data: BufferView { buffer, offset: 0 },
            phantom: Default::default(),
        }
    }
}

impl<'a, T: DataKind> From<TensorGpu<'a, T>> for TensorCpu<'a, T> {
    fn from(value: TensorGpu<'a, T>) -> Self {
        let size = value.byte_size() as u64;
        let Tensor {
            context,
            shape,
            name,
            data: BufferView { buffer, offset },
            ..
        } = value;

        let slice = buffer.slice(offset..offset + size);
        slice.map_async(MapMode::Read, |_| ());

        context.device.poll(MaintainBase::Wait);

        let map = slice.get_mapped_range();
        let data = Cow::from(bytemuck::cast_slice(&map).to_owned());
        buffer.unmap();

        Self {
            context,
            shape,
            name,
            data,
            phantom: Default::default(),
        }
    }
}

pub trait CopyTensor<T> {
    fn copy_tensor(&mut self, src: &T, dst: &T) -> Result<(), TensorError>;
}

impl<'a, T: DataKind> CopyTensor<TensorGpu<'a, T>> for CommandEncoder {
    fn copy_tensor(
        &mut self,
        src: &TensorGpu<'a, T>,
        dst: &TensorGpu<'a, T>,
    ) -> Result<(), TensorError> {
        if src.shape != dst.shape {
            return Err(TensorError::Shape(src.shape, dst.shape));
        }
        let size = src.byte_size() as BufferAddress;
        self.copy_buffer_to_buffer(&src.buffer, src.offset, &dst.buffer, dst.offset, size);
        Ok(())
    }
}

pub type TensorCpu<'a, T> = Tensor<'a, Cpu<'a, T>, T>;
pub type TensorGpu<'a, T> = Tensor<'a, Gpu, T>;

mod sealed {
    use super::{Cpu, Gpu};
    use half::prelude::f16;

    pub trait Sealed {}

    impl<T> Sealed for Cpu<'_, T> {}
    impl Sealed for Gpu {}

    impl Sealed for f32 {}
    impl Sealed for f16 {}
    impl Sealed for u8 {}
}
