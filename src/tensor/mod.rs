use std::{borrow::Cow, marker::PhantomData, sync::Arc};

use itertools::Itertools;
use shape::ShapedIndex;
use thiserror::Error;
use wgpu::{
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferUsages, ShaderStages,
};

use self::{
    kind::{Kind, ReadWrite, Uniform},
    shape::{IntoBytes, Shape, TensorAxis, TensorDimension, TensorSlice},
};
use crate::{
    context::Context,
    num::{Float, Scalar},
};

pub mod cache;
pub mod matrix;
pub mod ops;
pub mod serialization;
pub mod shape;

/// Data defining a tensor view in shader.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct View {
    pub shape: Shape,
    pub stride: Shape,
    pub offset: Shape,
}

impl IntoBytes for View {
    fn into_bytes(self) -> Vec<u8> {
        [
            self.shape.into_bytes(),
            self.stride.into_bytes(),
            self.offset.into_bytes(),
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

#[derive(Debug)]
pub struct TensorError {
    pub error: TensorErrorKind,
    #[cfg(feature = "backtrace")]
    pub backtrace: std::backtrace::Backtrace,
}

impl TensorError {
    #[cfg(feature = "backtrace")]
    pub fn new(error: TensorErrorKind) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        Self { error, backtrace }
    }

    #[cfg(not(feature = "backtrace"))]
    pub fn new(error: TensorErrorKind) -> Self {
        Self { error }
    }
}

impl From<TensorErrorKind> for TensorError {
    fn from(value: TensorErrorKind) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for TensorError {
    #[cfg(feature = "backtrace")]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}\n\nBacktrace:\n{}", self.error, self.backtrace)
    }

    #[cfg(not(feature = "backtrace"))]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.error)
    }
}

impl std::error::Error for TensorError {}

#[derive(Debug, Error)]
pub enum TensorErrorKind {
    #[error("list must not be empty")]
    Empty,
    #[error("data type mismatch")]
    Type,
    #[error("data size not match: {0} vs. {1}")]
    Size(usize, usize),
    #[error("batch size not match: {0} vs. {1}")]
    Batch(usize, usize),
    #[error("tensor shape not match: {0} vs. {1}")]
    Shape(Shape, Shape),
    #[error("cannot deduce dimension")]
    Deduce,
    #[error("batch {batch} out of range of max {max}")]
    BatchOutOfRange { batch: usize, max: usize },
    #[error("slice {start}..{end} out of range for dimension size {dim}")]
    SliceOutOfRange {
        dim: usize,
        start: usize,
        end: usize,
    },
    #[error("slice not contiguous")]
    SliceInvalid,
    #[error("cannot split along the axis {0}")]
    SplitInvalid(usize),
    #[error("possible tensor error(s):\n{0}")]
    Any(#[from] AnyTensorError),
}

#[derive(Debug, Error)]
pub struct AnyTensorError(pub Vec<Box<dyn std::error::Error + Send + Sync>>);

impl std::fmt::Display for AnyTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0
            .iter()
            .enumerate()
            .try_for_each(|(index, error)| writeln!(f, "{index}. {error}"))
    }
}

pub trait DeepClone: Sized {
    fn deep_clone(&self) -> Self;
}

pub trait TensorScalar {
    type T: Scalar;
}

pub trait TensorInitContext<T: Scalar>: Sized {
    /// Init the tensor with given shape and contents.
    fn from_data<'a, S, D>(context: &Context, shape: S, data: D) -> Result<Self, TensorError>
    where
        S: Into<Shape>,
        D: Into<Cow<'a, [T]>>;
    /// Init the tensor with given shape.
    fn init(context: &Context, shape: impl Into<Shape>) -> Self;
}

pub trait TensorInit<T: Scalar>: Sized {
    /// Init the tensor with given shape and contents.
    fn from_data<'a, S, D>(shape: S, data: D) -> Result<Self, TensorError>
    where
        S: Into<Shape>,
        D: Into<Cow<'a, [T]>>;
    /// Init the tensor with given shape.
    fn init(shape: impl Into<Shape>) -> Self;

    /// Init an 1-D tensor from data.
    fn from_data_1d<'a>(data: impl Into<Cow<'a, [T]>>) -> Self {
        let data: Cow<'_, [T]> = data.into();
        let shape = [data.len(), 1, 1, 1];
        Self::from_data(shape, data).expect("tensor 1d from data")
    }
}

pub trait TensorInto<Into> {
    fn to(self, context: &Context) -> Into;
}

pub trait TensorShape: Sized {
    /// Get the shape of the tensor.
    fn shape(&self) -> Shape;

    /// Check if the tensor's shape is the same as what expected.
    fn check_shape(&self, shape: impl Into<Shape>) -> Result<(), TensorError> {
        let shape = shape.into();
        (self.shape() == shape)
            .then_some(())
            .ok_or(TensorErrorKind::Shape(self.shape(), shape))
            .map_err(Into::into)
    }

    /// Check if the tensor's shape matches any of the expected ones.
    fn check_shape_any<S>(&self, shapes: &[S]) -> Result<(), TensorError>
    where
        S: Into<Shape> + ToOwned<Owned = S>,
    {
        let (oks, errors): (Vec<_>, Vec<_>) = shapes
            .iter()
            .map(|shape| self.check_shape(shape.to_owned()).map_err(Into::into))
            .partition_result();
        match oks.is_empty() {
            true => Err(TensorErrorKind::Any(AnyTensorError(errors)))?,
            false => Ok(()),
        }
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

pub trait TensorResource {
    /// Retrieve the key identifying a resource.
    fn resource_key(&self) -> ResourceKey;
    /// Binding for metadata of the tensor (shape, stride, etc.).
    fn meta_binding(&self) -> BindingResource<'_>;
    /// Binding for actual data of the tensor.
    fn binding(&self) -> BindingResource<'_>;
}

/// A tensor on either CPU or GPU.
#[derive(Debug)]
pub struct Tensor<D: Device, T: Scalar> {
    shape: Shape,
    data: D::Data,
    id: uid::Id<TensorId>,
    phantom: PhantomData<T>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId;

/// A unique identifier of tensor views. Useful in caches.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceKey {
    pub id: uid::Id<TensorId>,
    pub view: View,
}

pub trait Device: sealed::Sealed {
    type Data: Clone;
}

#[derive(Debug)]
pub struct Cpu<T: Scalar>(PhantomData<T>);

#[derive(Debug)]
pub struct Gpu<K: Kind>(PhantomData<K>);

impl<T: Scalar> Device for Cpu<T> {
    type Data = Arc<[T]>;
}

impl<K: Kind> Device for Gpu<K> {
    type Data = TensorGpuData;
}

/// Buffer of the tensor on GPU.
#[derive(Debug, Clone)]
pub struct TensorGpuData {
    pub context: Context,
    pub meta: Arc<Buffer>,
    pub buffer: Arc<Buffer>,
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
    #[usage(UNIFORM, COPY_DST, COPY_SRC)]
    pub struct Uniform;

    /// Tensor is a storage buffer with can be copied to other buffers.
    #[derive(Debug, Kind)]
    #[usage(STORAGE, COPY_DST, COPY_SRC)]
    pub struct ReadWrite;
}

pub type TensorCpu<T> = Tensor<Cpu<T>, T>;
pub type TensorGpu<T, K> = Tensor<Gpu<K>, T>;

impl<D: Device, T: Scalar> Clone for Tensor<D, T> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape,
            data: self.data.clone(),
            id: self.id,
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

    #[inline]
    pub fn id(&self) -> uid::Id<TensorId> {
        self.id
    }
}

impl<D: Device, F: Float> Tensor<D, F> {
    #[inline]
    pub const fn def(&self) -> &'static str {
        F::DEF
    }
}

impl<T: Scalar> TensorInit<T> for TensorCpu<T> {
    fn from_data<'a, S, D>(shape: S, data: D) -> Result<Self, TensorError>
    where
        S: Into<Shape>,
        D: Into<Cow<'a, [T]>>,
    {
        let shape = shape.into();
        let data: Cow<'_, _> = data.into();
        if shape.len() != data.len() {
            Err(TensorErrorKind::Size(shape.len(), data.len()))?;
        }
        let data = data.into_owned().into();
        Ok(Self {
            shape,
            data,
            id: uid::Id::new(),
            phantom: PhantomData,
        })
    }

    #[inline]
    fn init(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let data = vec![T::zero(); shape.len()].into();
        Self {
            shape,
            data,
            id: uid::Id::new(),
            phantom: PhantomData,
        }
    }
}

impl<T: Scalar> TensorInitContext<T> for TensorCpu<T> {
    fn from_data<'a, S, D>(_context: &Context, shape: S, data: D) -> Result<Self, TensorError>
    where
        S: Into<Shape>,
        D: Into<Cow<'a, [T]>>,
    {
        TensorInit::from_data(shape, data)
    }

    fn init(_context: &Context, shape: impl Into<Shape>) -> Self {
        TensorInit::init(shape)
    }
}

impl<T: Scalar> TensorInto<TensorCpu<T>> for TensorCpu<T> {
    fn to(self, _: &Context) -> Self {
        self
    }
}

impl<T: Scalar> DeepClone for TensorCpu<T> {
    fn deep_clone(&self) -> Self {
        self.clone()
    }
}

impl<T: Scalar> TensorShape for TensorCpu<T> {
    #[inline]
    fn shape(&self) -> Shape {
        self.shape
    }
}

impl<T: Scalar> TensorReshape for TensorCpu<T> {
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

impl<T: Scalar, K: Kind> TensorInitContext<T> for TensorGpu<T, K> {
    fn from_data<'a, S, D>(context: &Context, shape: S, data: D) -> Result<Self, TensorError>
    where
        S: Into<Shape>,
        D: Into<Cow<'a, [T]>>,
    {
        let tensor: TensorCpu<T> = TensorInit::from_data(shape, data)?;
        Ok(tensor.to(context))
    }

    fn init(context: &Context, shape: impl Into<Shape>) -> Self {
        let context = context.clone();
        let shape = shape.into();
        let meta = context.checkout_shape_uniform(shape);
        let size = shape.len() * std::mem::size_of::<T>();
        let buffer = context.checkout_buffer(size, K::buffer_usages());
        Self {
            shape,
            data: TensorGpuData {
                context,
                meta,
                buffer,
            },
            id: uid::Id::new(),
            phantom: PhantomData,
        }
    }
}

impl<T: Scalar, K: Kind> TensorInto<TensorGpu<T, K>> for TensorCpu<T> {
    fn to(self, context: &Context) -> TensorGpu<T, K> {
        let Tensor { shape, data, .. } = self;
        let context = context.clone();
        let meta = context.checkout_shape_uniform(shape);
        let contents = bytemuck::cast_slice(&data);
        let buffer = context.checkout_buffer_init(contents, K::buffer_usages());
        TensorGpu {
            shape,
            data: TensorGpuData {
                context,
                meta,
                buffer,
            },
            id: uid::Id::new(),
            phantom: PhantomData,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Scalar> TensorInto<TensorGpu<T, ReadWrite>> for TensorGpu<T, ReadWrite> {
    fn to(self, context: &Context) -> Self {
        match context {
            context if context == &self.context => self,
            _ => self.back_in_place().to(context),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl<T: Scalar> TensorInto<TensorGpu<T, ReadWrite>> for TensorGpu<T, ReadWrite> {
    fn to(self, _: &Context) -> Self {
        self
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
        let context = self.context.clone();
        let meta = context.checkout_shape_uniform(shape);
        let buffer = self.buffer.clone();
        Ok(Self {
            shape,
            data: TensorGpuData {
                context,
                meta,
                buffer,
            },
            ..self.clone()
        })
    }
}

impl<T: Scalar, K: Kind> TensorResource for TensorGpu<T, K> {
    #[inline]
    fn resource_key(&self) -> ResourceKey {
        let id = self.id;
        let view = View {
            shape: self.shape,
            stride: self.shape,
            offset: [0, 0, 0, 0].into(),
        };
        ResourceKey { id, view }
    }

    #[inline]
    fn meta_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.meta,
            offset: 0,
            size: None,
        })
    }

    #[inline]
    fn binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: None,
        })
    }
}

impl<T: Scalar, K: Kind> TensorGpu<T, K> {
    pub fn from_data_u8(
        context: &Context,
        shape: impl Into<Shape>,
        contents: &[u8],
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let size = shape.len() * size_of::<T>();
        if contents.len() != size {
            Err(TensorErrorKind::Size(size, contents.len()))?;
        }
        let buffer = context.checkout_buffer_init(contents, K::buffer_usages());
        let meta = context.checkout_shape_uniform(shape);
        Ok(Self {
            shape,
            data: TensorGpuData {
                context: context.clone(),
                meta,
                buffer,
            },
            id: uid::Id::new(),
            phantom: PhantomData,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn back_in_place(&self) -> TensorCpu<T> {
        use crate::context::ContextEvent;

        if self.is_empty() {
            return TensorCpu {
                shape: self.shape,
                data: Arc::new([]),
                id: uid::Id::new(),
                phantom: PhantomData,
            };
        }

        let context = &self.context;
        let size = self.buffer.size();
        let buffer = context.checkout_buffer(
            size as usize,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, size);
        context.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = flume::bounded(1);
        let _ = context.event().send(ContextEvent { buffer, sender });
        let data = receiver.recv().expect("failed to receive read back buffer");
        let data = unsafe {
            let data = Box::leak(data);
            let slice = bytemuck::cast_slice_mut::<_, T>(data);
            Box::from_raw(slice)
        };
        let data = data.into_vec().into();
        let shape = self.shape;

        TensorCpu {
            shape,
            data,
            id: uid::Id::new(),
            phantom: PhantomData,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn back(&self) -> TensorCpu<T> {
        if self.is_empty() {
            return TensorCpu {
                shape: self.shape,
                data: Arc::new([]),
                id: uid::Id::new(),
                phantom: PhantomData,
            };
        }

        let context = &self.context;
        let size = self.buffer.size();
        let buffer = context.checkout_buffer(
            size as usize,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, size);
        context.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = flume::bounded(1);

        let _ = context
            .event()
            .send(crate::context::ContextEvent { buffer, sender });
        let data = receiver
            .recv_async()
            .await
            .expect("failed to receive read back buffer");
        let data = unsafe {
            let data = Box::leak(data);
            let slice = bytemuck::cast_slice_mut::<_, T>(data);
            Box::from_raw(slice)
        };
        let data = data.into_vec().into();

        TensorCpu {
            shape: self.shape,
            data,
            id: uid::Id::new(),
            phantom: PhantomData,
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn back(self) -> TensorCpu<T> {
        if self.is_empty() {
            return TensorCpu {
                shape: self.shape,
                data: Arc::new([]),
                id: uid::Id::new(),
                phantom: PhantomData,
            };
        }

        let context = &self.context;
        let size = self.buffer.size();
        let buffer = context.checkout_buffer(
            size as usize,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, size);
        context.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = flume::unbounded();

        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        context.device.poll(wgpu::MaintainBase::Wait);
        receiver
            .recv_async()
            .await
            .expect("failed to receive read back buffer")
            .expect("failed to map buffer");

        let data = {
            let map = slice.get_mapped_range();
            Vec::from(bytemuck::cast_slice(&map)).into()
        };
        buffer.unmap();

        TensorCpu {
            shape: self.shape,
            data,
            id: uid::Id::new(),
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

    pub fn load_batch(&self, host: &TensorCpu<T>, batch: usize) -> Result<(), TensorError> {
        host.check_shape([self.shape[0], self.shape[1], 1, 1])?;
        if batch >= self.shape[2] {
            Err(TensorErrorKind::BatchOutOfRange {
                batch,
                max: self.shape[2],
            })?;
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

impl<T: Scalar> TensorGpu<T, Uniform> {
    #[inline]
    pub fn layout(&self, binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}

impl<T: Scalar> TensorGpu<T, ReadWrite> {
    #[inline]
    pub fn meta_layout(&self, binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    #[inline]
    pub fn layout(&self, binding: u32, read_only: bool) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}

impl<T: Scalar> From<TensorCpu<T>> for Vec<T> {
    #[inline]
    fn from(value: TensorCpu<T>) -> Self {
        // match Arc::get_mut(&mut value.data) {
        //     Some(data) => {
        //         // SAFETY: if `data` is unique, it stays unique in the scope of this function since we own the `Arc`.
        //         unsafe {
        //             let len = data.len();
        //             let data = Arc::into_raw(value.data) as *mut T;
        //             let slice = core::slice::from_raw_parts_mut(data, len);
        //             let boxed = Box::from_raw(slice);
        //             boxed.into_vec()
        //         }
        //     }
        //     None => value.data.to_vec(),
        // }
        value.to_vec()
    }
}

impl<T: Scalar, S: Into<ShapedIndex>> std::ops::Index<S> for TensorCpu<T> {
    type Output = T;

    fn index(&self, index: S) -> &Self::Output {
        &self.data[self.shape.linear_index(index)]
    }
}

impl<T: Scalar> TensorCpu<T> {
    /// Apply a map `f` to every element in the tensor.
    pub fn map<U: Scalar>(self, f: impl FnMut(&T) -> U) -> TensorCpu<U> {
        let Self { shape, data, .. } = self;
        let data = data.iter().map(f).collect_vec();
        TensorInit::from_data(shape, &data).unwrap()
    }

    /// Pad each dimension to multiples with zeros.
    pub fn pad(self, multiples: impl Into<Shape>) -> Self {
        // let shape = Shape::new(
        //     self.shape[0].next_multiple_of(64),
        //     self.shape[1].next_multiple_of(64),
        //     self.shape[2].next_multiple_of(64),
        //     self.shape[3].next_multiple_of(64),
        // );

        let multiples: Shape = multiples.into();

        let mut shape = self.shape;
        for (axis, multiple) in multiples.iter().enumerate() {
            shape[axis] = shape[axis].next_multiple_of(multiple);
        }

        let mut data = vec![T::zero(); shape.len()];
        for index in self.shape.cartesian_product() {
            let value = self[index];
            data[shape.linear_index(index)] = value;
        }
        TensorInit::from_data(shape, data).unwrap()
    }

    /// Repeat the tensor along a given axis.
    pub fn repeat(self, axis: usize, repeat: usize) -> Self {
        let mut shape = self.shape;
        let data = self.data;

        let num_chunk: usize = shape.iter().skip(axis + 1).product();
        let chunk_size = data.len() / num_chunk;

        shape[axis] *= repeat;

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

        Self {
            shape,
            data,
            id: uid::Id::new(),
            phantom: PhantomData,
        }
    }

    /// Concat a batch of tensors.
    pub fn stack(batches: Vec<Self>, axis: usize) -> Result<Self, TensorError> {
        let mut shape = match batches.first() {
            Some(batch) => batch.shape,
            None => Err(TensorErrorKind::Empty)?,
        };

        // batches
        //     .iter()
        //     .try_for_each(|batch| batch.check_shape([shape[0], shape[1], batch.shape[2], 1]))?;

        batches.iter().try_for_each(|batch| match axis {
            0 => batch.check_shape([batch.shape[0], 1, 1, 1]),
            1 => batch.check_shape([shape[0], batch.shape[1], 1, 1]),
            2 => batch.check_shape([shape[0], shape[1], batch.shape[2], 1]),
            3 => batch.check_shape([shape[0], shape[1], shape[2], batch.shape[3]]),
            _ => unreachable!(),
        })?;

        let num_batch: usize = batches.iter().map(|batch| batch.shape[axis]).sum();
        shape[axis] = num_batch;

        let data = batches
            .into_iter()
            .map(|batch| batch.data.to_vec())
            .collect_vec()
            .concat()
            .into();

        Ok(Self {
            shape,
            data,
            id: uid::Id::new(),
            phantom: PhantomData,
        })
    }

    /// Split the tensor along the batch axis.
    pub fn split(self, axis: usize) -> Result<Vec<Self>, TensorError> {
        if self.shape.iter().skip(axis + 1).any(|dim| dim > 1) {
            Err(TensorErrorKind::SplitInvalid(axis))?;
        }

        (0..self.shape[axis])
            .map(|index| match axis {
                0 => self.slice(index, .., .., ..),
                1 => self.slice(.., index, .., ..),
                2 => self.slice(.., .., index, ..),
                3 => self.slice(.., .., .., ..),
                _ => Err(TensorErrorKind::SplitInvalid(axis))?,
            })
            .try_collect()
    }

    pub fn slice(
        &self,
        x: impl TensorAxis,
        y: impl TensorAxis,
        z: impl TensorAxis,
        w: impl TensorAxis,
    ) -> Result<Self, TensorError> {
        let slice = (x, y, z, w);
        let (start, end) = slice.shaped_bounds(self.shape)?;
        let shape = (end - start).into();

        let (start, end) = slice.linear_bounds(self.shape)?;
        let data = self.data[start..end].into();

        Ok(Self {
            shape,
            data,
            id: uid::Id::new(),
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
        let (start, end) = slice.shaped_bounds(self.shape)?;
        let shape = (end - start).into();

        let (start, end) = slice.linear_bounds(self.shape)?;
        let data = self.data[start..end].into();

        Ok(Self {
            shape,
            data,
            id: uid::Id::new(),
            phantom: PhantomData,
        })
    }
}

/// Like a reference to a tensor, but refer to a sub-chunk of it.
#[derive(Debug, Clone)]
pub struct TensorGpuView<'a, T: Scalar> {
    tensor: &'a TensorGpu<T, ReadWrite>,
    meta: Arc<Buffer>,
    view: View,
}

impl<T: Scalar> TensorShape for TensorGpuView<'_, T> {
    #[inline]
    fn shape(&self) -> Shape {
        self.view.shape
    }
}

impl<T: Scalar> TensorGpuView<'_, T> {
    #[inline]
    pub fn tensor(&self) -> &TensorGpu<T, ReadWrite> {
        self.tensor
    }

    #[inline]
    pub fn context(&self) -> &Context {
        self.tensor.context()
    }

    #[inline]
    pub fn data(&self) -> &TensorGpuData {
        &self.tensor.data
    }

    #[inline]
    pub fn meta_layout(&self, binding: u32) -> BindGroupLayoutEntry {
        self.tensor.meta_layout(binding)
    }

    #[inline]
    pub fn layout(&self, binding: u32, read_only: bool) -> BindGroupLayoutEntry {
        self.tensor.layout(binding, read_only)
    }
}

impl<T: Scalar> TensorResource for TensorGpuView<'_, T> {
    #[inline]
    fn resource_key(&self) -> ResourceKey {
        ResourceKey {
            id: self.tensor.id,
            view: self.view,
        }
    }

    #[inline]
    fn meta_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.meta,
            offset: 0,
            size: None,
        })
    }

    #[inline]
    fn binding(&self) -> BindingResource<'_> {
        self.tensor.binding()
    }
}

impl<T: Scalar> TensorScalar for TensorGpuView<'_, T> {
    type T = T;
}

impl<F: Float> TensorGpuView<'_, F> {
    #[inline]
    pub const fn def(&self) -> &'static str {
        F::DEF
    }
}

impl<'a, T: Scalar> From<&'a TensorGpu<T, ReadWrite>> for TensorGpuView<'a, T> {
    fn from(value: &'a TensorGpu<T, ReadWrite>) -> Self {
        value.view(.., .., .., ..).unwrap()
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
    ) -> Result<TensorGpuView<'_, T>, TensorError> {
        let slice = (x, y, z, w);
        let (start, end) = slice.shaped_bounds(self.shape)?;
        let view = View {
            stride: self.shape,
            offset: start.into(),
            shape: (end - start).into(),
        };
        let meta = self.context.checkout_view_uniform(view);
        Ok(TensorGpuView {
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
        let size = shape.len() as u64;
        let cloned: TensorGpu<_, _> = context.tensor_init(shape);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &cloned.buffer, 0, size);
        context.queue.submit(Some(encoder.finish()));

        cloned
    }
}

/// Stack a batch of tensors of shape `[C, T, 1]` to one with shape `[C, A, 1]`, with cursors information.
#[derive(Debug, Clone)]
pub struct TensorStack<T: Scalar> {
    pub tensor: TensorCpu<T>,
    pub cursors: Vec<Cursor>,
}

impl<T: Scalar> TensorStack<T> {
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

impl<T: Scalar> TryFrom<Vec<TensorCpu<T>>> for TensorStack<T> {
    type Error = TensorError;

    fn try_from(value: Vec<TensorCpu<T>>) -> Result<Self, Self::Error> {
        let shape = match value.first() {
            Some(batch) => batch.shape,
            None => Err(TensorErrorKind::Empty)?,
        };

        value
            .iter()
            .try_for_each(|batch| batch.check_shape([shape[0], batch.shape[1], 1, 1]))?;

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
                data.extend(tensor.data.to_vec());
                (shape, data)
            },
        );
        let data = data.into();

        Ok(Self {
            tensor: Tensor {
                shape,
                data,
                id: uid::Id::new(),
                phantom: PhantomData,
            },
            cursors,
        })
    }
}

impl Context {
    #[inline]
    pub fn zeros<T: Scalar, Tensor>(&self, shape: impl Into<Shape>) -> Tensor
    where
        TensorCpu<T>: TensorInto<Tensor>,
    {
        let tensor: TensorCpu<T> = TensorInit::init(shape);
        tensor.to(self)
    }

    #[inline]
    pub fn ones<T: Scalar, Tensor>(&self, shape: impl Into<Shape>) -> Tensor
    where
        TensorCpu<T>: TensorInto<Tensor>,
    {
        let shape = shape.into();
        let data = vec![T::one(); shape.len()];
        let tensor: TensorCpu<T> = TensorInit::from_data(shape, data).unwrap();
        tensor.to(self)
    }

    #[inline]
    pub fn tensor_from_data<'a, T: Scalar, Tensor: TensorInitContext<T>>(
        &self,
        shape: impl Into<Shape>,
        data: impl Into<Cow<'a, [T]>>,
    ) -> Result<Tensor, TensorError> {
        TensorInitContext::from_data(self, shape, data)
    }

    #[inline]
    pub fn tensor_init<T: Scalar, Tensor: TensorInitContext<T>>(
        &self,
        shape: impl Into<Shape>,
    ) -> Tensor {
        TensorInitContext::init(self, shape)
    }
}

mod sealed {
    use super::{Cpu, Gpu, Kind, ReadWrite, Uniform};
    use crate::num::Scalar;

    pub trait Sealed {}

    impl<T: Scalar> Sealed for Cpu<T> {}
    impl<K: Kind> Sealed for Gpu<K> {}

    impl Sealed for Uniform {}
    impl Sealed for ReadWrite {}
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::Shape;
    use crate::tensor::{TensorCpu, TensorInit, TensorShape};

    #[test]
    fn test_pad_64() -> Result<()> {
        let shape = Shape::new(133, 256, 1, 1);
        let x: Vec<_> = (0..shape.len()).map(|x| x as f32).collect();
        let x = TensorCpu::from_data(shape, x)?.pad([64, 64, 1, 1]);

        assert_eq!(x.shape(), Shape::new(192, 256, 1, 1));
        assert_eq!(x[(132, 255, 0, 0)], (shape.len() - 1) as f32);
        assert_eq!(x[(133, 255, 0, 0)], 0.0);

        Ok(())
    }

    #[test]
    fn test_repeat() -> Result<()> {
        let shape = Shape::new(5, 1, 2, 1);
        let x: Vec<_> = (0..shape.len()).map(|x| x as f32).collect();
        let x = TensorCpu::from_data(shape, x)?;

        let y = x.clone().repeat(1, 3);
        let ans = [
            [0.0, 1.0, 2.0, 3.0, 4.0].repeat(3),
            [5.0, 6.0, 7.0, 8.0, 9.0].repeat(3),
        ]
        .concat();
        y.check_shape([5, 3, 2, 1])?;
        assert_eq!(y.to_vec(), ans);

        let y = x.clone().repeat(0, 3);
        y.check_shape([15, 1, 2, 1])?;
        assert_eq!(y.to_vec(), ans);

        let y = x.repeat(2, 3);
        let ans = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].repeat(3);
        y.check_shape([5, 1, 6, 1])?;
        assert_eq!(y.to_vec(), ans);

        Ok(())
    }

    #[test]
    fn test_split() -> Result<()> {
        let shape = Shape::new(5, 1, 2, 1);
        let x: Vec<_> = (0..10).map(|x| x as f32).collect();
        let x = TensorCpu::from_data(shape, x)?;

        assert!(x.clone().split(0).is_err());
        assert!(x.clone().split(1).is_err());

        let x = x.split(2)?;
        x[0].check_shape([5, 1, 1, 1])?;
        x[1].check_shape([5, 1, 1, 1])?;
        assert_eq!(x[0].to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(x[1].to_vec(), vec![5.0, 6.0, 7.0, 8.0, 9.0]);

        Ok(())
    }
}
