use std::{borrow::Cow, marker::PhantomData, sync::Arc};

use web_rwkv_derive::Kind;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindingResource, Buffer, BufferBinding, BufferDescriptor, BufferUsages, MapMode,
};

use crate::{context::Context, num::Scalar};
use shape::{IntoBytes, Shape, TensorSlice};

pub mod cache;
pub mod ops;
pub mod shape;

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
pub struct Cpu<'a, T: Scalar>(&'a PhantomData<T>);

#[derive(Debug)]
pub struct Gpu<K: Kind>(PhantomData<K>);

impl<'a, T: Scalar> Device for Cpu<'a, T> {
    type Data = Cow<'a, [T]>;
}

impl<K: Kind> Device for Gpu<K> {
    type Data = TensorBuffer;
}

pub trait Kind: sealed::Sealed {
    fn buffer_usages() -> BufferUsages;
}

/// Tensor is a uniform buffer.
#[derive(Debug, Kind)]
#[usage(UNIFORM, COPY_DST)]
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
    Empty,
    Size(usize, usize),
    Shape(Shape, Shape),
    OutOfRange {
        dim: usize,
        start: usize,
        end: usize,
    },
    Contiguous,
    Pipeline(&'static str),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::Empty => write!(f, "given list is empty"),
            TensorError::Size(a, b) => write!(f, "data size not match: {} vs. {}", a, b),
            TensorError::Shape(a, b) => write!(f, "tensor shape {} doesn't match {}", a, b),
            TensorError::OutOfRange { dim, start, end } => write!(
                f,
                "slice {}..{} out of range for dimension size {}",
                start, end, dim
            ),
            TensorError::Contiguous => write!(f, "slice not contiguous"),
            TensorError::Pipeline(name) => write!(f, "pipeline {} not found", name),
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
pub struct Tensor<'a, D: Device, T: Scalar> {
    context: &'a Context,
    shape: Shape,
    data: D::Data,
    phantom: PhantomData<(D, T)>,
}

pub type TensorCpu<'a, 'b, T> = Tensor<'a, Cpu<'b, T>, T>;
pub type TensorGpu<'a, T, K> = Tensor<'a, Gpu<K>, T>;

pub trait TensorExt<'a, 'b, T: Scalar>: Sized + Clone {
    fn from_data(
        context: &'a Context,
        shape: Shape,
        data: impl Into<Cow<'b, [T]>>,
    ) -> Result<Self, TensorError>;
    fn init(context: &'a Context, shape: Shape) -> Self;

    fn shape(&self) -> Shape;
    fn check_shape(&self, shape: Shape) -> Result<(), TensorError> {
        (self.shape() == shape)
            .then_some(())
            .ok_or(TensorError::Shape(self.shape(), shape))
    }
}

impl<D: Device, T: Scalar> std::ops::Deref for Tensor<'_, D, T> {
    type Target = D::Data;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<D: Device, T: Scalar> Clone for Tensor<'_, D, T> {
    fn clone(&self) -> Self {
        Self {
            context: self.context,
            shape: self.shape,
            data: self.data.clone(),
            phantom: PhantomData,
        }
    }
}

impl<D: Device, T: Scalar> Tensor<'_, D, T> {
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

impl<'a, 'b, T: Scalar> TensorExt<'a, 'b, T> for TensorCpu<'a, 'b, T> {
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

impl<'a, 'b, T: Scalar, K: Kind> From<TensorCpu<'a, 'b, T>> for TensorGpu<'a, T, K> {
    fn from(value: TensorCpu<'a, 'b, T>) -> Self {
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

impl<'a, 'b, T: Scalar> From<TensorGpu<'a, T, ReadBack>> for TensorCpu<'a, 'b, T> {
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

impl<'a, T: Scalar, K: Kind> TensorGpu<'a, T, K> {
    pub fn load(&self, host: &TensorCpu<'a, '_, T>) -> Result<(), TensorError> {
        self.check_shape(host.shape)?;
        self.context
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&host.data[..]));
        Ok(())
    }

    pub fn reshape(self, shape: Shape) -> Result<Self, TensorError> {
        if self.shape.len() != shape.len() {
            return Err(TensorError::Size(self.shape.len(), shape.len()));
        }
        Ok(Self { shape, ..self })
    }
}

impl<T: Scalar> From<TensorCpu<'_, '_, T>> for Vec<T> {
    #[inline]
    fn from(value: TensorCpu<'_, '_, T>) -> Self {
        Self::from(value.data)
    }
}

impl<'a, 'b, T: Scalar> std::ops::Index<(usize, usize, usize)> for TensorCpu<'a, 'b, T> {
    type Output = T;

    fn index(&self, (x, y, z): (usize, usize, usize)) -> &Self::Output {
        &self.data[self.shape.shape_index(Shape::new(x, y, z))]
    }
}

impl<'a, 'b, T: Scalar> TensorCpu<'a, 'b, T> {
    /// Repeat the tensor along a given axis.
    pub fn repeat(self, axis: usize, repeat: usize) -> Self {
        let Self {
            context,
            mut shape,
            data,
            ..
        } = self;
        let data = data.to_vec();
        let num_chunk: usize = shape.iter().skip(axis + 1).product();
        let chunk_size = data.len() / num_chunk;

        let data = (0..num_chunk)
            .map(|chunk| {
                let start = chunk * chunk_size;
                let end = start + chunk_size;
                let chunk = data[start..end].to_vec();
                chunk.repeat(repeat)
            })
            .collect::<Vec<_>>()
            .concat()
            .into();
        shape[axis] *= repeat;
        Self {
            context,
            shape,
            data,
            phantom: PhantomData,
        }
    }

    /// Split the tensor along the highest plural axis.
    pub fn split(self) -> Vec<Self> {
        match self.shape {
            Shape([0, _, _]) | Shape([_, 0, _]) | Shape([_, _, 0]) => vec![],
            Shape([1, 1, 1]) => vec![self],
            Shape([x, 1, 1]) => (0..x)
                .map(|batch| self.as_slice((batch, .., ..)).unwrap())
                .collect(),
            Shape([_, x, 1]) => (0..x)
                .map(|batch| self.as_slice((.., batch, ..)).unwrap())
                .collect(),
            Shape([_, _, x]) => (0..x)
                .map(|batch| self.as_slice((.., .., batch)).unwrap())
                .collect(),
        }
    }

    /// Concat a batch of tensors.
    pub fn concat(batches: Vec<Self>) -> Result<Self, TensorError> {
        if batches.is_empty() {
            return Err(TensorError::Empty);
        }

        let mut shape = batches[0].shape;
        let context = batches[0].context;

        batches.iter().try_for_each(|batch| {
            batch.check_shape(Shape::new(shape[0], shape[1], batch.shape[2]))
        })?;

        let num_batch: usize = batches.iter().map(|batch| batch.shape[2]).sum();
        shape[2] = num_batch;

        let data = batches
            .into_iter()
            .map(|batch| batch.data.to_vec())
            .collect::<Vec<_>>()
            .concat()
            .into();
        Ok(Self {
            context,
            shape,
            data,
            phantom: PhantomData,
        })
    }

    pub fn reshape(self, shape: Shape) -> Result<Self, TensorError> {
        if self.shape.len() != shape.len() {
            return Err(TensorError::Size(self.shape.len(), shape.len()));
        }
        Ok(Self { shape, ..self })
    }

    pub fn as_slice(&self, slice: impl TensorSlice) -> Result<TensorCpu<'a, 'b, T>, TensorError> {
        let (start, end) = slice.shape_bounds(self.shape)?;
        let shape = end - start;

        let (start, end) = slice.contiguous_bounds(self.shape)?;
        let data = match &self.data {
            Cow::Borrowed(data) => Cow::Borrowed(&data[start..end]),
            Cow::Owned(data) => Cow::Owned(data[start..end].to_owned()),
        };

        Ok(Self {
            context: self.context,
            shape,
            data,
            phantom: PhantomData,
        })
    }

    pub fn into_slice(self, slice: impl TensorSlice) -> Result<Self, TensorError> {
        let (start, end) = slice.shape_bounds(self.shape)?;
        let shape = end - start;

        let (start, end) = slice.contiguous_bounds(self.shape)?;
        let data = match self.data {
            Cow::Borrowed(data) => Cow::Borrowed(&data[start..end]),
            Cow::Owned(data) => Cow::Owned(data[start..end].to_owned()),
        };

        Ok(Self {
            context: self.context,
            shape,
            data,
            phantom: PhantomData,
        })
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
        TensorGpu::from_data(context, shape, data).and_then(|tensor| tensor.into_view((.., .., ..)))
    }

    #[inline]
    fn init(context: &'a Context, shape: Shape) -> Self {
        TensorGpu::init(context, shape)
            .into_view((.., .., ..))
            .unwrap()
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

impl<'a, T: Scalar> From<TensorGpu<'a, T, ReadWrite>> for TensorView<'a, T> {
    fn from(value: TensorGpu<'a, T, ReadWrite>) -> Self {
        value.into_view((.., .., ..)).unwrap()
    }
}

impl<'a, T: Scalar> TensorGpu<'a, T, ReadWrite> {
    pub fn as_view(&self, slice: impl TensorSlice) -> Result<TensorView<'a, T>, TensorError> {
        let (start, end) = slice.shape_bounds(self.shape)?;
        let view = View {
            stride: self.shape,
            offset: start,
            shape: end - start,
        };
        Ok(TensorView {
            context: self.context,
            view,
            data: TensorBuffer {
                meta: self.context.request_view_uniform(view),
                buffer: self.buffer.clone(),
            },
            phantom: PhantomData,
        })
    }

    pub fn into_view(self, slice: impl TensorSlice) -> Result<TensorView<'a, T>, TensorError> {
        let (start, end) = slice.shape_bounds(self.shape)?;
        let view = View {
            stride: self.shape,
            offset: start,
            shape: end - start,
        };
        Ok(TensorView {
            context: self.context,
            view,
            data: TensorBuffer {
                meta: self.context.request_view_uniform(view),
                buffer: self.buffer.clone(),
            },
            phantom: PhantomData,
        })
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
    use super::{Cpu, Gpu, Kind, ReadBack, ReadWrite, Uniform};
    use crate::num::Scalar;

    pub trait Sealed {}

    impl<T: Scalar> Sealed for Cpu<'_, T> {}
    impl<K: Kind> Sealed for Gpu<K> {}

    impl Sealed for Uniform {}
    impl Sealed for ReadWrite {}
    impl Sealed for ReadBack {}
}

#[cfg(test)]
mod tests {
    use wgpu::PowerPreference;

    use super::Shape;
    use crate::{
        context::{Context, ContextBuilder, Instance},
        tensor::{TensorCpu, TensorExt},
    };

    fn create_context() -> Result<Context, anyhow::Error> {
        let adapter = pollster::block_on(async {
            let instance = Instance::new();
            instance.adapter(PowerPreference::HighPerformance).await
        })?;
        let context = pollster::block_on(async {
            ContextBuilder::new(adapter)
                .with_default_pipelines()
                .build()
                .await
        })?;
        Ok(context)
    }

    #[test]
    fn test_repeat() -> Result<(), anyhow::Error> {
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };

        let shape = Shape::new(5, 1, 2);
        let x: Vec<_> = (0..10).map(|x| x as f32).collect();
        let x = TensorCpu::from_data(&context, shape, x)?;

        let y = x.clone().repeat(1, 3);
        let ans = [
            vec![0.0, 1.0, 2.0, 3.0, 4.0].repeat(3),
            vec![5.0, 6.0, 7.0, 8.0, 9.0].repeat(3),
        ]
        .concat();
        y.check_shape(Shape::new(5, 3, 2))?;
        assert_eq!(y.to_vec(), ans);

        let y = x.clone().repeat(0, 3);
        y.check_shape(Shape::new(15, 1, 2))?;
        assert_eq!(y.to_vec(), ans);

        let y = x.repeat(2, 3);
        let ans = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].repeat(3);
        y.check_shape(Shape::new(5, 1, 6))?;
        assert_eq!(y.to_vec(), ans);

        Ok(())
    }
}
