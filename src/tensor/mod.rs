use std::{borrow::Cow, marker::PhantomData, sync::Arc};

use web_rwkv_derive::Kind;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindingResource, Buffer, BufferBinding, BufferDescriptor, BufferUsages, MapMode,
};

use crate::{context::Context, num::Scalar};
pub use ops::{TensorCommand, TensorOp, TensorPass, TensorQueue};
pub use shape::{Shape, TensorSlice};
pub use uniform::{IntoBytes, UniformCache};

mod ops;
mod shape;
mod uniform;

#[derive(Debug, Clone)]
pub struct TensorBuffer {
    pub meta: Arc<Buffer>,
    pub buffer: Arc<Buffer>,
}

impl TensorBuffer {
    #[inline]
    pub fn meta_binding(&self) -> BindingResource {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.meta,
            offset: 0,
            size: None,
        })
    }

    #[inline]
    pub fn binding(&self) -> BindingResource {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: None,
        })
    }
}

pub trait Device: sealed::Sealed {
    type Data: Clone;
}

#[derive(Debug)]
pub struct Cpu<'a, T>(&'a PhantomData<T>);

#[derive(Debug)]
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
#[derive(Debug, Kind)]
#[usage(UNIFORM)]
pub struct Uniform;

/// Tensor is a storage buffer with can be copied to other buffers.
#[derive(Debug, Kind)]
#[usage(STORAGE, COPY_DST, COPY_SRC)]
pub struct ReadWrite;

/// Tensor is served as a read-back buffer.
#[derive(Debug, Kind)]
#[usage(MAP_READ, COPY_DST)]
pub struct ReadBack;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorError {
    Size(usize, usize),
    Shape(Shape, Shape),
    SliceOutOfRange {
        dim: usize,
        start: usize,
        end: usize,
    },
    PipelineError,
    DeviceError,
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::Size(a, b) => write!(f, "Data size not match: {} vs. {}", a, b),
            TensorError::Shape(a, b) => write!(f, "Tensor shape not match: {} vs. {}", a, b),
            TensorError::SliceOutOfRange { dim, start, end } => write!(
                f,
                "Slice {}..{} out of range for dimension size {}",
                start, end, dim
            ),
            TensorError::PipelineError => write!(f, "Pipeline not found"),
            TensorError::DeviceError => write!(f, "Tensor not on the same device"),
        }
    }
}

impl std::error::Error for TensorError {}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct View {
    pub stride: Shape,
    pub offset: Shape,
    pub shape: Shape,
}

impl IntoBytes for View {
    fn into_bytes(self) -> Vec<u8> {
        [
            self.stride.into_bytes(),
            self.offset.into_bytes(),
            self.shape.into_bytes(),
        ]
        .concat()
    }
}

#[derive(Debug)]
pub struct Tensor<'a, D: Device, T: Scalar, K: Kind> {
    context: &'a Context,
    shape: Shape,
    data: D::Data,
    phantom: PhantomData<(D, T, K)>,
}

pub type TensorCpu<'a, 'b, T, K> = Tensor<'a, Cpu<'b, T>, T, K>;
pub type TensorGpu<'a, T, K> = Tensor<'a, Gpu, T, K>;

pub trait TensorExt<'a, 'b, T: Scalar>: Sized + Clone {
    fn from_data(
        context: &'a Context,
        shape: Shape,
        data: impl Into<Cow<'b, [T]>>,
    ) -> Result<Self, TensorError>;

    fn init(context: &'a Context, shape: Shape) -> Self;

    fn shape(&self) -> Shape;

    fn check_shape(&self, shape: Shape) -> Result<(), TensorError> {
        if self.shape() == shape {
            Ok(())
        } else {
            Err(TensorError::Shape(self.shape(), shape))
        }
    }
}

impl<D: Device, T: Scalar, K: Kind> std::ops::Deref for Tensor<'_, D, T, K> {
    type Target = D::Data;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<D: Device, T: Scalar, K: Kind> Clone for Tensor<'_, D, T, K> {
    fn clone(&self) -> Self {
        Self {
            context: self.context,
            shape: self.shape,
            data: self.data.clone(),
            phantom: PhantomData,
        }
    }
}

impl<D: Device, T: Scalar, K: Kind> Tensor<'_, D, T, K> {
    #[inline]
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    /// Size of the tensor in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.len() * T::size()
    }

    /// The offset in bytes for a linear index.
    #[inline]
    pub fn offset(index: usize) -> usize {
        index * T::size()
    }

    #[inline]
    pub fn context(&self) -> &Context {
        self.context
    }

    #[inline]
    pub fn data(&self) -> &D::Data {
        &self.data
    }
}

impl<'a, 'b, T: Scalar, K: Kind> TensorExt<'a, 'b, T> for TensorCpu<'a, 'b, T, K> {
    fn from_data(
        context: &'a Context,
        shape: Shape,
        data: impl Into<Cow<'b, [T]>>,
    ) -> Result<Self, TensorError> {
        let data = data.into();
        if shape.len() != data.len() {
            return Err(TensorError::Size(shape.len(), data.len()));
        }
        Ok(Self {
            context,
            shape,
            data,
            phantom: PhantomData,
        })
    }

    #[inline]
    fn init(context: &'a Context, shape: Shape) -> Self {
        context.zeros(shape)
    }

    #[inline]
    fn shape(&self) -> Shape {
        self.shape
    }
}

impl<T: Scalar, K: Kind> From<TensorCpu<'_, '_, T, K>> for Vec<T> {
    #[inline]
    fn from(value: TensorCpu<'_, '_, T, K>) -> Self {
        Self::from(value.data)
    }
}

impl<'a, 'b, T: Scalar, K: Kind> TensorExt<'a, 'b, T> for TensorGpu<'a, T, K> {
    #[inline]
    fn from_data(
        context: &'a Context,
        shape: Shape,
        data: impl Into<Cow<'b, [T]>>,
    ) -> Result<Self, TensorError> {
        TensorCpu::from_data(context, shape, data).map(Into::into)
    }

    /// Initialize a GPU tensor with a given shape.
    fn init(context: &'a Context, shape: Shape) -> Self {
        let size = shape.len() as u64 * T::size() as u64;
        let buffer = context
            .device
            .create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage: K::buffer_usages(),
                mapped_at_creation: false,
            })
            .into();

        Self {
            context,
            shape,
            data: TensorBuffer {
                meta: context.request_shape_uniform(shape),
                buffer,
            },
            phantom: PhantomData,
        }
    }

    #[inline]
    fn shape(&self) -> Shape {
        self.shape
    }
}

impl<'a, 'b, T: Scalar, K: Kind> From<TensorCpu<'a, 'b, T, K>> for TensorGpu<'a, T, K> {
    fn from(value: TensorCpu<'a, 'b, T, K>) -> Self {
        let Tensor {
            context,
            shape,
            data,
            ..
        } = value;
        let contents = bytemuck::cast_slice(&data);
        let buffer = context
            .device
            .create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents,
                usage: K::buffer_usages(),
            })
            .into();

        Self {
            context,
            shape,
            data: TensorBuffer {
                meta: context.request_shape_uniform(shape),
                buffer,
            },
            phantom: PhantomData,
        }
    }
}

impl<'a, 'b, T: Scalar> From<TensorGpu<'a, T, ReadBack>> for TensorCpu<'a, 'b, T, ReadBack> {
    fn from(value: TensorGpu<'a, T, ReadBack>) -> Self {
        let Tensor {
            context,
            shape,
            data: TensorBuffer { buffer, .. },
            ..
        } = value;

        let slice = buffer.slice(..);
        slice.map_async(MapMode::Read, |_| ());

        context.device.poll(wgpu::MaintainBase::Wait);

        let data = {
            let map = slice.get_mapped_range();
            Vec::from(bytemuck::cast_slice(&map))
        };
        buffer.unmap();

        Self {
            context,
            shape,
            data: Cow::from(data),
            phantom: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorView<'a, T: Scalar> {
    context: &'a Context,
    view: View,
    data: TensorBuffer,
    phantom: PhantomData<T>,
}

impl<'a, T: Scalar> std::ops::Deref for TensorView<'a, T> {
    type Target = TensorBuffer;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, 'b, T: Scalar> TensorExt<'a, 'b, T> for TensorView<'a, T> {
    #[inline]
    fn from_data(
        context: &'a Context,
        shape: Shape,
        data: impl Into<Cow<'b, [T]>>,
    ) -> Result<Self, TensorError> {
        TensorGpu::from_data(context, shape, data).map(|tensor| tensor.into_view((.., .., ..)))
    }

    #[inline]
    fn init(context: &'a Context, shape: Shape) -> Self {
        TensorGpu::init(context, shape).into_view((.., .., ..))
    }

    #[inline]
    fn shape(&self) -> Shape {
        self.view.shape
    }
}

impl<'a, T: Scalar> TensorView<'a, T> {
    #[inline]
    pub fn context(&self) -> &Context {
        self.context
    }

    #[inline]
    pub fn data(&self) -> &TensorBuffer {
        &self.data
    }
}

impl<'a, T: Scalar> TensorGpu<'a, T, ReadWrite> {
    pub fn as_view(&'a self, slice: impl TensorSlice) -> TensorView<'a, T> {
        let (start, end) = slice.shape_bounds(self.shape);
        let view = View {
            stride: self.shape,
            offset: start,
            shape: end - start,
        };
        TensorView {
            context: self.context,
            view,
            data: TensorBuffer {
                meta: self.context.request_view_uniform(view),
                buffer: self.buffer.clone(),
            },
            phantom: PhantomData,
        }
    }

    pub fn into_view(self, slice: impl TensorSlice) -> TensorView<'a, T> {
        let (start, end) = slice.shape_bounds(self.shape);
        let view = View {
            stride: self.shape,
            offset: start,
            shape: end - start,
        };
        TensorView {
            context: self.context,
            view,
            data: TensorBuffer {
                meta: self.context.request_view_uniform(view),
                buffer: self.buffer.clone(),
            },
            phantom: PhantomData,
        }
    }
}

impl<'a, 'b> Context {
    #[inline]
    pub fn zeros<T: Scalar, Tensor: TensorExt<'a, 'b, T>>(&'a self, shape: Shape) -> Tensor {
        let data = vec![T::zero(); shape.len()];
        Tensor::from_data(self, shape, data).unwrap()
    }

    #[inline]
    pub fn ones<T: Scalar, Tensor: TensorExt<'a, 'b, T>>(&'a self, shape: Shape) -> Tensor {
        let data = vec![T::one(); shape.len()];
        Tensor::from_data(self, shape, data).unwrap()
    }

    #[inline]
    pub fn tensor_from_data<T: Scalar, Tensor: TensorExt<'a, 'b, T>>(
        &'a self,
        shape: Shape,
        data: impl Into<Cow<'b, [T]>>,
    ) -> Result<Tensor, TensorError> {
        Tensor::from_data(self, shape, data)
    }

    #[inline]
    pub fn init_tensor<T: Scalar, Tensor: TensorExt<'a, 'b, T>>(&'a self, shape: Shape) -> Tensor {
        Tensor::init(self, shape)
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
