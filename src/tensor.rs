use ahash::HashMap;
use half::f16;
use std::{
    borrow::Cow,
    cell::RefCell,
    marker::PhantomData,
    num::NonZeroU64,
    sync::{Arc, RwLock},
};
use web_rwkv_derive::{Deref, Id, Kind};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferAddress,
    BufferBinding, BufferDescriptor, BufferUsages, CommandEncoder, ComputePass, ComputePipeline,
    MaintainBase, MapMode, Queue,
};

use crate::{context::Context, num::Scalar};

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

/// The shape of a [`Tensor`].
/// Note that the fastest-moving axis occupies the lowest shape index, which is opposite to that in `torch`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorShape(pub [usize; 3]);

impl TensorShape {
    pub fn len(&self) -> usize {
        self.0.into_iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.0.into_iter().any(|x| x == 0)
    }

    /// Convert a shaped index into a linear index.
    pub fn shape_index(&self, indices: TensorShape) -> usize {
        Iterator::zip(self.0.into_iter().rev(), indices.0.into_iter().rev())
            .fold(0, |acc, (shape, index)| acc * shape + index)
    }

    pub fn to_u32_slice(self) -> [u32; 4] {
        [self.0[0] as u32, self.0[1] as u32, self.0[2] as u32, 1]
    }
}

impl std::fmt::Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self[0], self[1], self[2])
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

fn check_shape<D: Device, T: Scalar, K: Kind>(
    tensor: &Tensor<D, T, K>,
    shape: TensorShape,
) -> Result<(), TensorError> {
    if tensor.shape == shape {
        Ok(())
    } else {
        Err(TensorError::Shape(tensor.shape, shape))
    }
}

#[allow(dead_code)]
fn check_shape_dims<D: Device, T: Scalar, K: Kind>(
    tensor: &Tensor<D, T, K>,
    shape: TensorShape,
    dims: Vec<usize>,
) -> Result<(), TensorError> {
    dims.into_iter()
        .all(|dim| tensor.shape[dim] == shape[dim])
        .then_some(())
        .ok_or(TensorError::Shape(tensor.shape, shape))
}

#[derive(Debug, Default, Deref)]
pub struct TensorShapeCache(RefCell<RwLock<HashMap<TensorShape, Arc<Buffer>>>>);

impl TensorShapeCache {
    pub fn clear(&self) {
        let map = self.borrow_mut();
        let mut map = map.write().unwrap();
        map.clear();
    }

    pub fn query(&self, shape: TensorShape) -> Option<Arc<Buffer>> {
        let map = self.borrow();
        let map = map.read().unwrap();
        map.get(&shape).cloned()
    }

    pub fn buffer<F>(&self, shape: TensorShape, op: F) -> Arc<Buffer>
    where
        F: FnOnce() -> Buffer,
    {
        match self.query(shape) {
            Some(buffer) => buffer,
            None => {
                let buffer = Arc::new(op());
                let map = self.borrow_mut();
                let mut map = map.write().unwrap();
                map.insert(shape, buffer.clone());
                buffer
            }
        }
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
    PipelineError,
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
                size,
            } => write!(
                f,
                "Buffer overflow with buffer size: {}, slice offset: {} and size: {}",
                buffer_size, offset, size
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
    shape: TensorShape,
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
        shape: TensorShape,
        data: Vec<T>,
    ) -> Result<Self, TensorError>;

    fn init(context: &'a Context, name: Option<&'a str>, shape: TensorShape) -> Self;
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
    pub fn shape_index(&self, indices: TensorShape) -> usize {
        self.shape.shape_index(indices)
    }

    pub fn context(&self) -> &Context {
        self.context
    }

    pub fn shape(&self) -> TensorShape {
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
        shape: TensorShape,
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

    fn init(context: &'a Context, name: Option<&'a str>, shape: TensorShape) -> Self {
        context.zeros(name, shape)
    }
}

impl<T: Scalar, K: Kind> From<TensorCpu<'_, T, K>> for Vec<T> {
    fn from(value: TensorCpu<'_, T, K>) -> Self {
        Self::from(value.data)
    }
}

impl<'a, T: Scalar, K: Kind> std::ops::Index<usize> for TensorCpu<'a, T, K> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, T: Scalar, K: Kind> std::ops::Index<std::ops::Range<usize>> for TensorCpu<'a, T, K> {
    type Output = [T];

    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, T: Scalar, K: Kind> TensorExt<'a, T> for TensorGpu<'a, T, K> {
    fn from_data(
        context: &'a Context,
        name: Option<&'a str>,
        shape: TensorShape,
        data: Vec<T>,
    ) -> Result<Self, TensorError> {
        TensorCpu::from_data(context, name, shape, data).map(Into::into)
    }

    /// Initialize a GPU tensor with a given shape.
    fn init(context: &'a Context, name: Option<&'a str>, shape: TensorShape) -> Self {
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
        let shape_buffer = context.tensor_shape_cache.buffer(shape, || {
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
        shape: TensorShape,
        offset: BufferAddress,
    ) -> Result<Self, TensorError> {
        let Self { context, data, .. } = other;
        let size = shape.len() as u64 * T::size() as u64;
        if data.offset + size >= data.buffer.size() {
            return Err(TensorError::Overflow {
                buffer_size: data.buffer.size(),
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
        let shape_buffer = context.tensor_shape_cache.buffer(shape, || {
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

        context.device.poll(MaintainBase::Wait);

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

pub trait TensorCommand<T: Scalar, K: Kind> {
    fn copy_tensor(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
    ) -> Result<(), TensorError>;
}

impl<T: Scalar, K: Kind> TensorCommand<T, K> for CommandEncoder {
    fn copy_tensor(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
    ) -> Result<(), TensorError> {
        check_shape(source, destination.shape)?;
        let size = source.size() as BufferAddress;
        self.copy_buffer_to_buffer(
            &source.buffer,
            source.offset,
            &destination.buffer,
            destination.offset,
            size,
        );
        Ok(())
    }
}

pub trait TensorQueue<T: Scalar, K: Kind> {
    fn write_tensor(
        &mut self,
        host: &TensorCpu<T, K>,
        device: &TensorGpu<T, K>,
    ) -> Result<(), TensorError>;
}

impl<T: Scalar, K: Kind> TensorQueue<T, K> for Queue {
    fn write_tensor(
        &mut self,
        host: &TensorCpu<T, K>,
        device: &TensorGpu<T, K>,
    ) -> Result<(), TensorError> {
        check_shape(host, device.shape)?;
        self.write_buffer(
            &device.buffer,
            device.offset,
            bytemuck::cast_slice(&host.data),
        );
        Ok(())
    }
}

pub trait TensorPass<'a> {
    fn execute_tensor_op(&mut self, op: &'a TensorOp);
}

impl<'a, 'b> TensorPass<'a> for ComputePass<'b>
where
    'a: 'b,
{
    fn execute_tensor_op(&mut self, op: &'a TensorOp) {
        self.set_pipeline(op.pipeline);
        op.bindings
            .iter()
            .enumerate()
            .for_each(|(index, bind_group)| self.set_bind_group(index as u32, bind_group, &[]));
        self.dispatch_workgroups(op.dispatch[0], op.dispatch[1], op.dispatch[2]);
    }
}

pub struct TensorOp<'a> {
    pub pipeline: &'a ComputePipeline,
    pub bindings: Vec<BindGroup>,
    pub dispatch: [u32; 3],
}

impl<'a> TensorOp<'a> {
    const BLOCK_SIZE: u32 = 128;

    fn block_count(x: u32) -> u32 {
        (x + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE
    }

    pub fn softmax(
        x: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        check_shape(x, shape)?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("softmax")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    pub fn layer_norm(
        x: &'a TensorGpu<f32, ReadWrite>,
        w: &'a TensorGpu<f16, ReadWrite>,
        b: &'a TensorGpu<f16, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        check_shape(x, shape)?;
        check_shape(w, TensorShape([shape[0], 1, 1]))?;
        check_shape(b, TensorShape([shape[0], 1, 1]))?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("layer_norm")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: w.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: b.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    pub fn matmul(
        matrix: &'a TensorGpu<f16, ReadWrite>,
        input: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        check_shape(matrix, TensorShape([input.shape[0], shape[0], 1]))?;
        check_shape(input, TensorShape([matrix.shape[0], shape[1], shape[2]]))?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("matmul")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [matrix.shape[1] as u32 / 4, shape[1] as u32, shape[2] as u32],
        })
    }

    pub fn matmul_int8(
        matrix: &'a TensorGpu<u8, ReadWrite>,
        mx: &'a TensorGpu<f32, ReadWrite>,
        rx: &'a TensorGpu<f32, ReadWrite>,
        my: &'a TensorGpu<f32, ReadWrite>,
        ry: &'a TensorGpu<f32, ReadWrite>,
        input: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        check_shape(matrix, TensorShape([input.shape[0], shape[0], 1]))?;
        check_shape(input, TensorShape([matrix.shape[0], shape[1], shape[2]]))?;
        check_shape(mx, TensorShape([matrix.shape[0], 1, 1]))?;
        check_shape(rx, TensorShape([matrix.shape[0], 1, 1]))?;
        check_shape(my, TensorShape([matrix.shape[1], 1, 1]))?;
        check_shape(ry, TensorShape([matrix.shape[1], 1, 1]))?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("matmul_int8")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: mx.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: rx.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: my.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: ry.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [matrix.shape[1] as u32 / 4, shape[1] as u32, shape[2] as u32],
        })
    }

    pub fn add(
        x: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        check_shape(x, shape)?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("add")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn token_shift(
        time_mix: &'a TensorGpu<f16, ReadWrite>,
        x: &'a TensorGpu<f32, ReadWrite>,
        sx: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        check_shape(x, shape)?;
        check_shape(time_mix, TensorShape([shape[0], 1, 1]))?;
        check_shape(sx, TensorShape([shape[0], 1, shape[2]]))?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("token_shift")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: time_mix.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: sx.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn token_mix(
        mask: &'a TensorGpu<u32, Uniform>,
        time_decay: &'a TensorGpu<f32, ReadWrite>,
        time_first: &'a TensorGpu<f32, ReadWrite>,
        x: &'a TensorGpu<f32, ReadWrite>,
        k: &'a TensorGpu<f32, ReadWrite>,
        v: &'a TensorGpu<f32, ReadWrite>,
        r: &'a TensorGpu<f32, ReadWrite>,
        state: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        check_shape(mask, TensorShape([1, 1, 1]))?;
        check_shape(x, shape)?;
        check_shape(k, shape)?;
        check_shape(v, shape)?;
        check_shape(r, shape)?;
        check_shape(time_decay, TensorShape([shape[0], 1, 1]))?;
        check_shape(time_first, TensorShape([shape[0], 1, 1]))?;
        check_shape(state, TensorShape([shape[0], 4, shape[2]]))?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("token_mix")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: mask.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: time_decay.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: time_first.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: k.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: v.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: r.binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: state.binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [Self::block_count(shape[0] as u32 / 4), 1, shape[2] as u32],
        })
    }

    pub fn squared_relu(
        x: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        check_shape(x, shape)?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("squared_relu")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn channel_mix(
        x: &'a TensorGpu<f32, ReadWrite>,
        r: &'a TensorGpu<f32, ReadWrite>,
        v: &'a TensorGpu<f32, ReadWrite>,
        state: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        check_shape(x, shape)?;
        check_shape(v, shape)?;
        check_shape(r, shape)?;
        check_shape(state, TensorShape([shape[0], 1, shape[2]]))?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("channel_mix")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: r.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: v.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: state.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }
}

impl<'a> Context {
    pub fn zeros<T: Scalar, Tensor: TensorExt<'a, T>>(
        &'a self,
        name: Option<&'a str>,
        shape: TensorShape,
    ) -> Tensor {
        let data = vec![T::zero(); shape.len()];
        Tensor::from_data(self, name, shape, data).unwrap()
    }

    pub fn ones<T: Scalar, Tensor: TensorExt<'a, T>>(
        &'a self,
        name: Option<&'a str>,
        shape: TensorShape,
    ) -> Tensor {
        let data = vec![T::one(); shape.len()];
        Tensor::from_data(self, name, shape, data).unwrap()
    }

    pub fn tensor_from_data<T: Scalar, Tensor: TensorExt<'a, T>>(
        &'a self,
        name: Option<&'a str>,
        shape: TensorShape,
        data: Vec<T>,
    ) -> Result<Tensor, TensorError> {
        Tensor::from_data(self, name, shape, data)
    }

    pub fn tensor_init<T: Scalar, Tensor: TensorExt<'a, T>>(
        &'a self,
        name: Option<&'a str>,
        shape: TensorShape,
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

#[cfg(test)]
mod tests {
    use half::f16;
    use itertools::Itertools;
    use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor, PowerPreference};

    use super::{TensorExt, TensorOp, TensorPass};
    use crate::{
        context::{Context, ContextBuilder, Instance},
        tensor::{TensorCommand, TensorCpu, TensorGpu, TensorShape},
    };

    fn is_approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= f32::max(f32::EPSILON, f32::max(a.abs(), b.abs()) * f32::EPSILON)
    }

    fn is_approx_eps(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= f32::max(eps, f32::max(a.abs(), b.abs()) * eps)
    }

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
    fn test_shape_index() {
        let shape = TensorShape([1024, 768, 12]);
        let indices = TensorShape([35, 42, 9]);
        let index = shape.shape_index(indices);
        assert_eq!(index, 35 + 42 * 1024 + 9 * 1024 * 768);
    }

    #[test]
    fn test_copy() -> Result<(), anyhow::Error> {
        let context = create_context()?;

        let x = vec![0.0, 1.5, 2.0, -1.0];
        let shape = TensorShape([x.len(), 1, 1]);

        let x_device: TensorGpu<_, _> = context.tensor_from_data(None, shape, x.clone())?;
        let x_map = context.tensor_init(None, x_device.shape());

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_tensor(&x_device, &x_map)?;
        context.queue.submit(Some(encoder.finish()));

        let x_host = TensorCpu::from(x_map);
        let x_host = Vec::from(x_host);

        assert_eq!(x, x_host);
        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<(), anyhow::Error> {
        let context = create_context()?;

        let x = [(); 6000].map(|_| 10.0 * (fastrand::f32() - 0.5)).to_vec();
        let shape = TensorShape([x.len() / 6, 3, 2]);

        let x_dev: TensorGpu<_, _> = context.tensor_from_data(None, shape, x.clone())?;
        let x_out = context.tensor_init(None, x_dev.shape());
        let x_map = context.tensor_init(None, x_dev.shape());

        let softmax = TensorOp::softmax(&x_dev, &x_out)?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&softmax);
        drop(pass);

        encoder.copy_tensor(&x_out, &x_map)?;
        context.queue.submit(Some(encoder.finish()));

        let x_host = TensorCpu::from(x_map);
        let x_host = Vec::from(x_host);

        let mut ans = vec![];
        for x in &x.into_iter().chunks(1000) {
            let x: Vec<_> = x.collect();
            let x = x.into_iter();
            let max = x.clone().reduce(f32::max).unwrap_or_default();
            let x = x.map(|x| (x - max).exp());
            let sum: f32 = x.clone().sum();
            let mut x: Vec<_> = x.map(|x| x / sum).collect();
            ans.append(&mut x);
        }

        for (index, (a, b)) in Iterator::zip(x_host.into_iter(), ans.into_iter()).enumerate() {
            assert!(
                is_approx(a, b),
                "Failed at index {index}, computed: {a} vs. answer: {b}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_matmul() -> Result<(), anyhow::Error> {
        let context = create_context()?;

        const C: usize = 1024;
        const R: usize = 768;
        const T: usize = 7;
        const B: usize = 3;

        let matrix: Vec<_> = vec![(); C * R]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .map(f16::from_f32)
            .collect();
        let input: Vec<_> = vec![(); C * T * B]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .collect();

        let matrix_dev = TensorGpu::from_data(
            &context,
            Some("matrix"),
            TensorShape([C, R, 1]),
            matrix.clone(),
        )?;
        let input_dev = TensorGpu::from_data(
            &context,
            Some("input"),
            TensorShape([C, T, B]),
            input.clone(),
        )?;
        let output_dev = TensorGpu::init(&context, None, TensorShape([R, T, B]));
        let output_map = TensorGpu::init(&context, None, output_dev.shape());

        let matmul = TensorOp::matmul(&matrix_dev, &input_dev, &output_dev)?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&matmul);
        drop(pass);

        encoder.copy_tensor(&output_dev, &output_map)?;
        context.queue.submit(Some(encoder.finish()));

        let output_host = TensorCpu::from(output_map);
        let output_host = Vec::from(output_host);

        let mut ans = vec![0.0; output_host.len()];
        for batch in 0..B {
            for token in 0..T {
                for line in 0..R {
                    let matrix = &matrix[line * C..(line + 1) * C];
                    let input = &input[(batch * T + token) * C..(batch * T + token + 1) * C];
                    let product = matrix
                        .iter()
                        .map(|x| (*x).to_f32())
                        .zip(input.iter())
                        .fold(0.0f32, |acc, x| acc + x.0 * *x.1);
                    ans[(batch * T + token) * R + line] = product;
                }
            }
        }

        for (index, (a, b)) in Iterator::zip(output_host.into_iter(), ans.into_iter()).enumerate() {
            assert!(
                is_approx_eps(a, b, 1.0e-3),
                "Failed at index {index}, computed: {a} vs. answer: {b}"
            );
        }

        Ok(())
    }
}
