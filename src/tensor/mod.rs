use std::{borrow::Cow, marker::PhantomData, sync::Arc};

use itertools::Itertools;
use web_rwkv_derive::JsError;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindingResource, Buffer, BufferBinding, BufferDescriptor, MapMode,
};

use self::{
    kind::{Kind, ReadBack, ReadWrite, Uniform},
    ops::TensorCommand,
    shape::{IntoBytes, Shape, TensorAxis, TensorDimension, TensorSlice},
};
use crate::{
    context::Context,
    model::loader::ReaderTensor,
    num::{Float, Scalar},
};

pub mod cache;
pub mod matrix;
pub mod ops;
pub mod shape;

/// Buffer of the tensor on GPU.
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

pub mod kind {
    use web_rwkv_derive::Kind;
    use wgpu::BufferUsages;

    use super::sealed;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, JsError)]
pub enum TensorError {
    Empty,
    Type,
    Size(usize, usize),
    Batch(usize, usize),
    Shape(Shape, Shape),
    Deduce,
    BatchOutOfRange {
        batch: usize,
        max: usize,
    },
    SliceOutOfRange {
        dim: usize,
        start: usize,
        end: usize,
    },
    Contiguous,
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::Empty => write!(f, "list must not be empty"),
            TensorError::Type => write!(f, "data type mismatch"),
            TensorError::Size(a, b) => write!(f, "data size not match: {a} vs. {b}"),
            TensorError::Batch(a, b) => write!(f, "batch size not match: {a} vs. {b}"),
            TensorError::Shape(a, b) => write!(f, "tensor shape not match: {a} vs. {b}"),
            TensorError::Deduce => write!(f, "cannot deduce dimension"),
            TensorError::BatchOutOfRange { batch, max } => {
                write!(f, "batch {batch} out of range of max {max}")
            }
            TensorError::SliceOutOfRange { dim, start, end } => write!(
                f,
                "slice {start}..{end} out of range for dimension size {dim}",
            ),
            TensorError::Contiguous => write!(f, "slice not contiguous"),
        }
    }
}

impl std::error::Error for TensorError {}

/// Data defining a tensor view in shader.
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

/// A record in order to separate different batches of input of various lengths.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cursor {
    pub batch: usize,
    pub token: usize,
    pub len: usize,
}

impl Cursor {
    pub fn pack(self) -> u32 {
        let batch = self.batch as u8;
        let token = (self.token as u16).to_ne_bytes();
        let len = self.len as u8;
        bytemuck::cast([batch, token[0], token[1], len])
    }
}

pub trait IntoPackedCursors {
    fn into_stack(self) -> Vec<u32>;
    fn into_cursors(self) -> Vec<u32>;
}

impl IntoPackedCursors for Vec<Cursor> {
    fn into_stack(self) -> Vec<u32> {
        self.into_iter()
            .filter(|cursor| cursor.len > 0)
            .map(Cursor::pack)
            .collect()
    }

    fn into_cursors(self) -> Vec<u32> {
        self.into_iter()
            .filter(|cursor| cursor.len > 0)
            .map(|cursor| {
                let repeat = cursor.len;
                vec![cursor.pack(); repeat]
            })
            .collect_vec()
            .concat()
    }
}

pub trait DeepClone: Sized {
    fn deep_clone(&self) -> Self;
}

pub trait TensorScalar {
    type T: Scalar;
}

pub trait TensorInit<'a, T: Scalar>: Sized {
    /// Init the tensor with given shape and contents.
    fn from_data(
        context: &Context,
        shape: Shape,
        data: impl Into<Cow<'a, [T]>>,
    ) -> Result<Self, TensorError>;
    /// Init the tensor with given shape.
    fn init(context: &Context, shape: Shape) -> Self;

    /// Create a tensor from safetensors reader.
    fn from_reader(
        context: &Context,
        (dt, shape, data): ReaderTensor<'a>,
    ) -> Result<Self, TensorError> {
        if T::DATA_TYPE != dt {
            return Err(TensorError::Type);
        }

        let shape = Shape::from_slice_rev(&shape)?;
        match data {
            Cow::Borrowed(data) => Self::from_data(context, shape, bytemuck::cast_slice(data)),
            Cow::Owned(data) => {
                let data = bytemuck::cast_slice(&data);
                let data = Cow::Owned(data.to_vec());
                Self::from_data(context, shape, data)
            }
        }
    }
}

pub trait TensorShape: Sized {
    fn shape(&self) -> Shape;

    fn check_shape(&self, shape: Shape) -> Result<(), TensorError> {
        (self.shape() == shape)
            .then_some(())
            .ok_or(TensorError::Shape(self.shape(), shape))
    }
}

pub trait TensorReshape: Sized {
    fn reshape(
        &self,
        x: TensorDimension,
        y: TensorDimension,
        z: TensorDimension,
        w: TensorDimension,
    ) -> Result<Self, TensorError>;
}

pub trait TensorTransfer: Sized {
    /// Transfer the tensor to another (may be the same) context.
    fn transfer(self, context: &Context) -> Result<Self, TensorError>;
}

/// A tensor on either CPU or GPU.
#[derive(Debug)]
pub struct Tensor<D: Device, T: Scalar> {
    context: Context,
    shape: Shape,
    data: D::Data,
    phantom: PhantomData<T>,
}

pub type TensorCpu<'a, T> = Tensor<Cpu<'a, T>, T>;
pub type TensorGpu<T, K> = Tensor<Gpu<K>, T>;

impl<D: Device, T: Scalar> Clone for Tensor<D, T> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            shape: self.shape,
            data: self.data.clone(),
            phantom: PhantomData,
        }
    }
}

impl<D: Device, T: Scalar> std::ops::Deref for Tensor<D, T> {
    type Target = D::Data;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<D: Device, T: Scalar> TensorScalar for Tensor<D, T> {
    type T = T;
}

impl<D: Device, T: Scalar> Tensor<D, T> {
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
    pub fn data(&self) -> &D::Data {
        &self.data
    }
}

impl<D: Device, F: Float> Tensor<D, F> {
    #[inline]
    pub const fn def(&self) -> &'static str {
        F::DEF
    }
}

impl<'a, T: Scalar> TensorInit<'a, T> for TensorCpu<'a, T> {
    fn from_data(
        context: &Context,
        shape: Shape,
        data: impl Into<Cow<'a, [T]>>,
    ) -> Result<Self, TensorError> {
        let data = data.into();
        if shape.len() != data.len() {
            return Err(TensorError::Size(shape.len(), data.len()));
        }
        Ok(Self {
            context: context.clone(),
            shape,
            data,
            phantom: PhantomData,
        })
    }

    #[inline]
    fn init(context: &Context, shape: Shape) -> Self {
        context.zeros(shape)
    }
}

impl<'a, T: Scalar> DeepClone for TensorCpu<'a, T> {
    fn deep_clone(&self) -> Self {
        self.clone()
    }
}

impl<T: Scalar> TensorShape for TensorCpu<'_, T> {
    #[inline]
    fn shape(&self) -> Shape {
        self.shape
    }
}

impl<T: Scalar> TensorReshape for TensorCpu<'_, T> {
    #[inline]
    fn reshape(
        &self,
        x: TensorDimension,
        y: TensorDimension,
        z: TensorDimension,
        w: TensorDimension,
    ) -> Result<Self, TensorError> {
        let shape = TensorDimension::deduce(self.shape, x, y, z, w)?;
        Ok(Self {
            shape,
            ..self.clone()
        })
    }
}

impl<T: Scalar> TensorTransfer for TensorCpu<'_, T> {
    fn transfer(self, context: &Context) -> Result<Self, TensorError> {
        Ok(Self {
            context: context.clone(),
            shape: self.shape,
            data: self.data,
            phantom: PhantomData,
        })
    }
}

impl<'a, T: Scalar, K: Kind> TensorInit<'a, T> for TensorGpu<T, K> {
    #[inline]
    fn from_data(
        context: &Context,
        shape: Shape,
        data: impl Into<Cow<'a, [T]>>,
    ) -> Result<Self, TensorError> {
        TensorCpu::from_data(context, shape, data).map(Into::into)
    }

    /// Initialize a GPU tensor with a given shape.
    fn init(context: &Context, shape: Shape) -> Self {
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
            context: context.clone(),
            shape,
            data: TensorBuffer {
                meta: context.checkout_shape_uniform(shape),
                buffer,
            },
            phantom: PhantomData,
        }
    }
}

impl<T: Scalar, K: Kind> TensorShape for TensorGpu<T, K> {
    #[inline]
    fn shape(&self) -> Shape {
        self.shape
    }
}

impl<T: Scalar, K: Kind> TensorReshape for TensorGpu<T, K> {
    #[inline]
    fn reshape(
        &self,
        x: TensorDimension,
        y: TensorDimension,
        z: TensorDimension,
        w: TensorDimension,
    ) -> Result<Self, TensorError> {
        let shape = TensorDimension::deduce(self.shape, x, y, z, w)?;
        let meta = self.context.checkout_shape_uniform(shape);
        Ok(Self {
            shape,
            data: TensorBuffer {
                meta,
                buffer: self.data.buffer.clone(),
            },
            ..self.clone()
        })
    }
}

impl<T: Scalar, K: Kind> From<TensorCpu<'_, T>> for TensorGpu<T, K> {
    fn from(value: TensorCpu<T>) -> Self {
        let Tensor {
            context,
            shape,
            data,
            ..
        } = value;
        let meta = context.checkout_shape_uniform(shape);
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
            data: TensorBuffer { meta, buffer },
            phantom: PhantomData,
        }
    }
}

impl<T: Scalar> TensorGpu<T, ReadBack> {
    pub fn back<'a>(self) -> TensorCpu<'a, T> {
        let Tensor {
            context,
            shape,
            data: TensorBuffer { buffer, .. },
            ..
        } = self;

        let (sender, receiver) = flume::unbounded();

        let slice = buffer.slice(..);
        slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

        context.device.poll(wgpu::MaintainBase::Wait);
        receiver.recv().unwrap().unwrap();

        let data = {
            let map = slice.get_mapped_range();
            Vec::from(bytemuck::cast_slice(&map))
        };
        buffer.unmap();

        TensorCpu {
            context,
            shape,
            data: Cow::from(data),
            phantom: PhantomData,
        }
    }

    pub async fn back_async<'a>(self) -> TensorCpu<'a, T> {
        let Tensor {
            context,
            shape,
            data: TensorBuffer { buffer, .. },
            ..
        } = self;

        let (sender, receiver) = flume::unbounded();

        let slice = buffer.slice(..);
        slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

        context.device.poll(wgpu::MaintainBase::Wait);
        receiver.recv_async().await.unwrap().unwrap();

        let data = {
            let map = slice.get_mapped_range();
            Vec::from(bytemuck::cast_slice(&map))
        };
        buffer.unmap();

        TensorCpu {
            context,
            shape,
            data: Cow::from(data),
            phantom: PhantomData,
        }
    }
}

impl<T: Scalar, K: Kind> TensorGpu<T, K> {
    #[inline]
    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn load(&self, host: &TensorCpu<T>) -> Result<(), TensorError> {
        host.check_shape(self.shape)?;
        self.context
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&host.data[..]));
        Ok(())
    }

    pub fn load_batch(&self, host: &TensorCpu<'_, T>, batch: usize) -> Result<(), TensorError> {
        host.check_shape(Shape::new(self.shape[0], self.shape[1], 1, 1))?;
        if batch >= self.shape[2] {
            return Err(TensorError::BatchOutOfRange {
                batch,
                max: self.shape[2],
            });
        }
        let offset = (T::size() * self.shape[0] * self.shape[1] * batch) as u64;
        self.context
            .queue
            .write_buffer(&self.buffer, offset, bytemuck::cast_slice(&host.data[..]));
        Ok(())
    }

    pub fn destroy(self) {
        self.buffer.destroy();
    }
}

impl<T: Scalar> From<TensorCpu<'_, T>> for Vec<T> {
    #[inline]
    fn from(value: TensorCpu<T>) -> Self {
        Self::from(value.data)
    }
}

impl<T: Scalar> std::ops::Index<(usize, usize, usize, usize)> for TensorCpu<'_, T> {
    type Output = T;

    fn index(&self, (x, y, z, w): (usize, usize, usize, usize)) -> &Self::Output {
        &self.data[self.shape.shape_index(Shape::new(x, y, z, w))]
    }
}

impl<'a, T: Scalar> TensorCpu<'a, T> {
    /// Apply a map `f` to every element in the tensor.
    pub fn map<U: Scalar>(self, f: impl FnMut(&T) -> U) -> TensorCpu<'a, U> {
        let Self {
            context,
            shape,
            data,
            ..
        } = self;
        let data = data.iter().map(f).collect_vec();
        TensorCpu::from_data(&context, shape, data).expect("this never happens")
    }

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
            .collect_vec()
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
    pub fn split(self, axis: usize) -> Result<Vec<Self>, TensorError> {
        match axis {
            0 => (0..self.shape[0])
                .map(|index| self.slice(index, .., .., ..))
                .try_collect(),
            1 => (0..self.shape[1])
                .map(|index| self.slice(.., index, .., ..))
                .try_collect(),
            2 => (0..self.shape[2])
                .map(|index| self.slice(.., .., index, ..))
                .try_collect(),
            3 => (0..self.shape[3])
                .map(|index| self.slice(.., .., .., index))
                .try_collect(),
            _ => Ok(vec![self]),
        }
    }

    /// Concat a batch of tensors.
    pub fn stack(batches: Vec<Self>) -> Result<Self, TensorError> {
        let (context, mut shape) = match batches.first() {
            Some(batch) => (batch.context.clone(), batch.shape),
            None => return Err(TensorError::Empty),
        };

        batches.iter().try_for_each(|batch| {
            batch.check_shape(Shape::new(shape[0], shape[1], batch.shape[2], 1))
        })?;

        let num_batch: usize = batches.iter().map(|batch| batch.shape[2]).sum();
        shape[2] = num_batch;

        let data = batches
            .into_iter()
            .map(|batch| batch.data.to_vec())
            .collect_vec()
            .concat()
            .into();
        Ok(Self {
            context,
            shape,
            data,
            phantom: PhantomData,
        })
    }

    pub fn slice(
        &self,
        x: impl TensorAxis,
        y: impl TensorAxis,
        z: impl TensorAxis,
        w: impl TensorAxis,
    ) -> Result<TensorCpu<'a, T>, TensorError> {
        let slice = (x, y, z, w);
        let (start, end) = slice.shape_bounds(self.shape)?;
        let shape = end - start;

        let (start, end) = slice.contiguous_bounds(self.shape)?;
        let data = match &self.data {
            Cow::Borrowed(data) => Cow::Borrowed(&data[start..end]),
            Cow::Owned(data) => Cow::Owned(data[start..end].to_owned()),
        };

        Ok(Self {
            context: self.context.clone(),
            shape,
            data,
            phantom: PhantomData,
        })
    }

    pub fn into_slice(
        self,
        x: impl TensorAxis,
        y: impl TensorAxis,
        z: impl TensorAxis,
        w: impl TensorAxis,
    ) -> Result<Self, TensorError> {
        let slice = (x, y, z, w);
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

/// Like a reference to a tensor, but refer to a sub-chunk of it.
#[derive(Debug, Clone)]
pub struct TensorView<'a, T: Scalar> {
    tensor: &'a TensorGpu<T, ReadWrite>,
    meta: Arc<Buffer>,
    view: View,
}

impl<T: Scalar> TensorShape for TensorView<'_, T> {
    #[inline]
    fn shape(&self) -> Shape {
        self.view.shape
    }
}

impl<T: Scalar> TensorView<'_, T> {
    #[inline]
    pub fn tensor(&self) -> &TensorGpu<T, ReadWrite> {
        self.tensor
    }

    #[inline]
    pub fn context(&self) -> &Context {
        self.tensor.context()
    }

    #[inline]
    pub fn data(&self) -> &TensorBuffer {
        &self.tensor.data
    }

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
        self.data().binding()
    }
}

impl<T: Scalar> TensorScalar for TensorView<'_, T> {
    type T = T;
}

impl<F: Float> TensorView<'_, F> {
    #[inline]
    pub const fn def(&self) -> &'static str {
        F::DEF
    }
}

impl<T: Scalar> TensorGpu<T, ReadWrite> {
    /// Create a view for the tensor.
    pub fn view(
        &self,
        x: impl TensorAxis,
        y: impl TensorAxis,
        z: impl TensorAxis,
        w: impl TensorAxis,
    ) -> Result<TensorView<'_, T>, TensorError> {
        let slice = (x, y, z, w);
        let (start, end) = slice.shape_bounds(self.shape)?;
        let view = View {
            stride: self.shape,
            offset: start,
            shape: end - start,
        };
        let meta = self.context.checkout_view_uniform(view);
        Ok(TensorView {
            tensor: self,
            meta,
            view,
        })
    }
}

impl<T: Scalar> DeepClone for TensorGpu<T, ReadWrite> {
    fn deep_clone(&self) -> Self {
        let context = &self.context;
        let shape = self.shape;
        let cloned = context.tensor_init(shape);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder
            .copy_tensor(self, &cloned)
            .expect("tensor deep clone");
        context.queue.submit(Some(encoder.finish()));

        cloned
    }
}

/// Stack a batch of tensors of shape `[C, T, 1]` to one with shape `[C, A, 1]`, with cursors information.
pub struct TensorStack<'a, T: Scalar> {
    pub tensor: TensorCpu<'a, T>,
    pub cursors: Vec<Cursor>,
}

impl<'a, T: Scalar> TensorStack<'a, T> {
    /// Number of input batches (including empty batches).
    #[inline]
    pub fn num_batch(&self) -> usize {
        self.cursors.len()
    }

    /// Number of non-empty input batches.
    #[inline]
    pub fn num_active_batch(&self) -> usize {
        self.cursors.iter().filter(|cursor| cursor.len > 0).count()
    }

    #[inline]
    pub fn num_token(&self) -> usize {
        self.tensor.shape[1]
    }
}

impl<T: Scalar> TryFrom<Vec<TensorCpu<'_, T>>> for TensorStack<'_, T> {
    type Error = TensorError;

    fn try_from(value: Vec<TensorCpu<T>>) -> Result<Self, Self::Error> {
        let (context, shape) = match value.first() {
            Some(batch) => (batch.context.clone(), batch.shape),
            None => return Err(TensorError::Empty),
        };

        value
            .iter()
            .try_for_each(|batch| batch.check_shape(Shape::new(shape[0], batch.shape[1], 1, 1)))?;

        // erase empty batches and pack them tightly
        // let mut redirect = vec![None; value.len()];
        // value
        //     .iter()
        //     .enumerate()
        //     .filter_map(|(index, tensor)| (!tensor.is_empty()).then_some(index))
        //     .enumerate()
        //     .for_each(|(packed, index)| redirect[index] = Some(packed));

        let cursors = value
            .iter()
            .enumerate()
            .scan(0, |token, (batch, tensor)| {
                let len = tensor.shape[1];
                let cursor = Cursor {
                    batch,
                    token: *token,
                    len,
                };
                *token += len;
                Some(cursor)
            })
            .collect_vec();

        let (shape, data) = value.into_iter().fold(
            (Shape::new(shape[0], 0, 1, 1), vec![]),
            |(mut shape, mut data), tensor| {
                shape[1] += tensor.shape[1];
                data.append(&mut tensor.data.to_vec());
                (shape, data)
            },
        );

        Ok(Self {
            tensor: Tensor {
                context,
                shape,
                data: data.into(),
                phantom: PhantomData,
            },
            cursors,
            // redirect,
        })
    }
}

impl<'a> Context {
    #[inline]
    pub fn zeros<T: Scalar, Tensor: TensorInit<'a, T>>(&self, shape: Shape) -> Tensor {
        let data = vec![T::zero(); shape.len()];
        Tensor::from_data(self, shape, data).unwrap()
    }

    #[inline]
    pub fn ones<T: Scalar, Tensor: TensorInit<'a, T>>(&self, shape: Shape) -> Tensor {
        let data = vec![T::one(); shape.len()];
        Tensor::from_data(self, shape, data).unwrap()
    }

    #[inline]
    pub fn tensor_from_data<T: Scalar, Tensor: TensorInit<'a, T>>(
        &self,
        shape: Shape,
        data: impl Into<Cow<'a, [T]>>,
    ) -> Result<Tensor, TensorError> {
        Tensor::from_data(self, shape, data)
    }

    #[inline]
    pub fn tensor_init<T: Scalar, Tensor: TensorInit<'a, T>>(&self, shape: Shape) -> Tensor {
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
    use anyhow::Result;
    use wgpu::PowerPreference;

    use super::Shape;
    use crate::{
        context::{Context, ContextBuilder, Instance},
        tensor::{TensorCpu, TensorInit, TensorShape},
    };

    #[tokio::main]
    async fn create_context() -> Result<Context> {
        let instance = Instance::new();
        let adapter = instance.adapter(PowerPreference::HighPerformance).await?;
        let context = ContextBuilder::new(adapter)
            // .with_features(Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES)
            .build()
            .await?;
        Ok(context)
    }

    #[test]
    fn test_repeat() -> Result<()> {
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };

        let shape = Shape::new(5, 1, 2, 1);
        let x: Vec<_> = (0..10).map(|x| x as f32).collect();
        let x = TensorCpu::from_data(&context, shape, x)?;

        let y = x.clone().repeat(1, 3);
        let ans = [
            [0.0, 1.0, 2.0, 3.0, 4.0].repeat(3),
            [5.0, 6.0, 7.0, 8.0, 9.0].repeat(3),
        ]
        .concat();
        y.check_shape(Shape::new(5, 3, 2, 1))?;
        assert_eq!(y.to_vec(), ans);

        let y = x.clone().repeat(0, 3);
        y.check_shape(Shape::new(15, 1, 2, 1))?;
        assert_eq!(y.to_vec(), ans);

        let y = x.repeat(2, 3);
        let ans = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].repeat(3);
        y.check_shape(Shape::new(5, 1, 6, 1))?;
        assert_eq!(y.to_vec(), ans);

        Ok(())
    }
}
