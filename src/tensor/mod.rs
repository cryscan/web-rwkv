use std::{borrow::Cow, marker::PhantomData, num::NonZeroU64, sync::Arc};

use web_rwkv_derive::{Deref, Id, Kind};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindingResource, Buffer, BufferAddress, BufferBinding, BufferDescriptor, BufferUsages, MapMode,
};

use crate::{context::Context, num::Scalar};
pub use ops::{TensorCommand, TensorOp, TensorPass, TensorQueue};
pub use shape::{Shape, ShapeCache, Slice};

mod ops;
mod shape;

#[derive(Debug, Clone)]
pub struct TensorBuffer {
    pub shape_buffer: Arc<Buffer>,
    pub buffer: Arc<Buffer>,
    pub offset: BufferAddress,
}

pub trait Device: sealed::Sealed {
    type Data: Clone;
}

pub struct Cpu<'a, T>(&'a PhantomData<T>);
pub struct Gpu;

impl<'a, T: Scalar> Device for Cpu<'a, T> {
    type Data = Cow<'a, [T]>;
}

impl Device for Gpu {
    type Data = TensorBuffer;
}

pub trait Kind: sealed::Sealed {
    fn buffer_usages() -> BufferUsages;
}

/// Tensor is a uniform buffer.
#[derive(Kind)]
#[usage(UNIFORM)]
pub struct Uniform;

/// Tensor is a storage buffer with can be copied to other buffers.
#[derive(Kind)]
#[usage(STORAGE, COPY_DST, COPY_SRC)]
pub struct ReadWrite;

/// Tensor is served as a read-back buffer.
#[derive(Kind)]
#[usage(MAP_READ, COPY_DST)]
pub struct ReadBack;

#[derive(Debug, Clone, Copy)]
pub enum TensorError {
    Size(usize, usize),
    Shape(Shape, Shape),
    Slice {
        shape: Shape,
        dim: usize,
    },
    Overflow {
        buffer: BufferAddress,
        offset: BufferAddress,
        size: BufferAddress,
    },
    PipelineError,
    DeviceError,
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::Size(a, b) => write!(f, "Data size not match: {} vs. {}", a, b),
            TensorError::Shape(a, b) => write!(f, "Tensor shape not match: {} vs. {}", a, b),
            TensorError::Slice { shape, dim } => {
                write!(f, "Slice overflows dim {} of shape {}", dim, shape)
            }
            TensorError::Overflow {
                buffer,
                offset,
                size,
            } => write!(
                f,
                "Buffer overflows with buffer size: {}, slice offset: {} and size: {}",
                buffer, offset, size
            ),
            TensorError::PipelineError => write!(f, "Pipeline not found"),
            TensorError::DeviceError => write!(f, "Tensor not on the same device"),
        }
    }
}

impl std::error::Error for TensorError {}

#[derive(Debug, Clone, Copy, Deref, Id, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

#[derive(Debug, Clone)]
pub struct Tensor<'a, D: Device, T: Scalar, K: Kind> {
    id: TensorId,
    context: &'a Context,
    shape: Shape,
    name: Option<&'a str>,
    data: D::Data,
    phantom: std::marker::PhantomData<(D, T, K)>,
}

pub type TensorCpu<'a, T, K> = Tensor<'a, Cpu<'a, T>, T, K>;
pub type TensorGpu<'a, T, K> = Tensor<'a, Gpu, T, K>;

pub trait TensorExt<'a, T: Scalar>: Sized {
    fn from_data(
        context: &'a Context,
        name: Option<&'a str>,
        shape: Shape,
        data: Vec<T>,
    ) -> Result<Self, TensorError>;

    fn init(context: &'a Context, name: Option<&'a str>, shape: Shape) -> Self;
}

impl<D: Device, T: Scalar, K: Kind> std::ops::Deref for Tensor<'_, D, T, K> {
    type Target = D::Data;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<D: Device, T: Scalar, K: Kind> Tensor<'_, D, T, K> {
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    /// Size of the tensor in bytes.
    pub fn size(&self) -> usize {
        self.len() * T::size()
    }

    /// The offset in bytes for a linear index.
    pub fn offset(index: usize) -> usize {
        index * T::size()
    }

    /// Convert a shaped index into a linear index.
    pub fn shape_index(&self, indices: Shape) -> usize {
        self.shape.shape_index(indices)
    }

    pub fn context(&self) -> &Context {
        self.context
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn name(&self) -> Option<&str> {
        self.name
    }

    pub fn data(&self) -> &D::Data {
        &self.data
    }
}

impl<'a, T: Scalar, K: Kind> TensorExt<'a, T> for TensorCpu<'a, T, K> {
    fn from_data(
        context: &'a Context,
        name: Option<&'a str>,
        shape: Shape,
        data: Vec<T>,
    ) -> Result<Self, TensorError> {
        if shape.len() != data.len() {
            return Err(TensorError::Size(shape.len(), data.len()));
        }
        Ok(Self {
            id: TensorId::new(),
            context,
            shape,
            name,
            data: Cow::from(data),
            phantom: Default::default(),
        })
    }

    fn init(context: &'a Context, name: Option<&'a str>, shape: Shape) -> Self {
        context.zeros(name, shape)
    }
}

impl<T: Scalar, K: Kind> From<TensorCpu<'_, T, K>> for Vec<T> {
    fn from(value: TensorCpu<'_, T, K>) -> Self {
        Self::from(value.data)
    }
}

impl<'a, T: Scalar, K: Kind> TensorExt<'a, T> for TensorGpu<'a, T, K> {
    fn from_data(
        context: &'a Context,
        name: Option<&'a str>,
        shape: Shape,
        data: Vec<T>,
    ) -> Result<Self, TensorError> {
        TensorCpu::from_data(context, name, shape, data).map(Into::into)
    }

    /// Initialize a GPU tensor with a given shape.
    fn init(context: &'a Context, name: Option<&'a str>, shape: Shape) -> Self {
        let label = name;
        let size = shape.len() as u64 * T::size() as u64;
        let buffer = context
            .device
            .create_buffer(&BufferDescriptor {
                label,
                size,
                usage: K::buffer_usages(),
                mapped_at_creation: false,
            })
            .into();
        let shape_buffer = context.shape_cache.buffer(shape, || {
            context.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&shape.to_u32_slice()),
                usage: BufferUsages::UNIFORM,
            })
        });
        Self {
            id: TensorId::new(),
            context,
            shape,
            name,
            data: TensorBuffer {
                shape_buffer,
                buffer,
                offset: 0,
            },
            phantom: Default::default(),
        }
    }
}

impl<'a, T: Scalar, K: Kind> TensorGpu<'a, T, K> {
    /// Create a GPU tensor from another one with new name, shape and offset.
    /// Fails if the buffer overflows.
    pub fn from_other(
        other: Self,
        name: Option<&'a str>,
        shape: Shape,
        offset: BufferAddress,
    ) -> Result<Self, TensorError> {
        let Self { context, data, .. } = other;
        let size = shape.len() as u64 * T::size() as u64;
        if data.offset + size >= data.buffer.size() {
            return Err(TensorError::Overflow {
                buffer: data.buffer.size(),
                offset: data.offset,
                size,
            });
        }
        Ok(Self {
            id: TensorId::new(),
            context,
            shape,
            name,
            data: TensorBuffer {
                shape_buffer: data.shape_buffer,
                buffer: data.buffer,
                offset,
            },
            phantom: Default::default(),
        })
    }

    pub fn shape_binding(&self) -> BindingResource {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.shape_buffer,
            offset: self.offset,
            size: NonZeroU64::new(16),
        })
    }

    pub fn binding(&self) -> BindingResource {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: NonZeroU64::new(self.size() as BufferAddress),
        })
    }
}

impl<'a, T: Scalar, K: Kind> From<TensorCpu<'a, T, K>> for TensorGpu<'a, T, K> {
    fn from(value: TensorCpu<'a, T, K>) -> Self {
        let Tensor {
            id,
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
        let shape_buffer = context.shape_cache.buffer(shape, || {
            context.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&shape.to_u32_slice()),
                usage: BufferUsages::UNIFORM,
            })
        });
        Self {
            id,
            context,
            shape,
            name,
            data: TensorBuffer {
                shape_buffer,
                buffer,
                offset: 0,
            },
            phantom: Default::default(),
        }
    }
}

impl<'a, T: Scalar> From<TensorGpu<'a, T, ReadBack>> for TensorCpu<'a, T, ReadBack> {
    fn from(value: TensorGpu<'a, T, ReadBack>) -> Self {
        let size = value.size() as u64;
        let Tensor {
            id,
            context,
            shape,
            name,
            data: TensorBuffer { buffer, offset, .. },
            ..
        } = value;

        let slice = buffer.slice(offset..offset + size);
        slice.map_async(MapMode::Read, |_| ());

        context.device.poll(wgpu::MaintainBase::Wait);

        let data = {
            let map = slice.get_mapped_range();
            Vec::from(bytemuck::cast_slice(&map))
        };
        buffer.unmap();

        Self {
            id,
            context,
            shape,
            name,
            data: Cow::from(data),
            phantom: Default::default(),
        }
    }
}

impl<'a> Context {
    pub fn zeros<T: Scalar, Tensor: TensorExt<'a, T>>(
        &'a self,
        name: Option<&'a str>,
        shape: Shape,
    ) -> Tensor {
        let data = vec![T::zero(); shape.len()];
        Tensor::from_data(self, name, shape, data).unwrap()
    }

    pub fn ones<T: Scalar, Tensor: TensorExt<'a, T>>(
        &'a self,
        name: Option<&'a str>,
        shape: Shape,
    ) -> Tensor {
        let data = vec![T::one(); shape.len()];
        Tensor::from_data(self, name, shape, data).unwrap()
    }

    pub fn tensor_from_data<T: Scalar, Tensor: TensorExt<'a, T>>(
        &'a self,
        name: Option<&'a str>,
        shape: Shape,
        data: Vec<T>,
    ) -> Result<Tensor, TensorError> {
        Tensor::from_data(self, name, shape, data)
    }

    pub fn tensor_init<T: Scalar, Tensor: TensorExt<'a, T>>(
        &'a self,
        name: Option<&'a str>,
        shape: Shape,
    ) -> Tensor {
        Tensor::init(self, name, shape)
    }
}

mod sealed {
    use super::{Cpu, Gpu, ReadBack, ReadWrite, Uniform};

    pub trait Sealed {}

    impl<T> Sealed for Cpu<'_, T> {}
    impl Sealed for Gpu {}

    impl Sealed for Uniform {}
    impl Sealed for ReadWrite {}
    impl Sealed for ReadBack {}
}
