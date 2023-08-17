use std::{borrow::Cow, marker::PhantomData, num::NonZeroU64, sync::Arc};
use web_rwkv_derive::{Deref, Id, Kind};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferAddress,
    BufferBinding, BufferDescriptor, BufferUsages, CommandEncoder, ComputePass, ComputePipeline,
    MaintainBase, MapMode,
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
#[kind(UNIFORM)]
pub struct Uniform;

/// Tensor is a storage buffer with can be copied to other buffers.
#[derive(Kind)]
#[kind(STORAGE, COPY_DST, COPY_SRC)]
pub struct ReadWrite;

/// Tensor is served as a read-back buffer.
#[derive(Kind)]
#[kind(MAP_READ, COPY_DST)]
pub struct ReadBack;

/// The shape of a [`Tensor`].
/// Note that the fastest-moving axis occupies the lowest shape index, which is opposite to that in `torch`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorShape(pub [usize; 4]);

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
        let shape_buffer = context
            .device
            .create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&shape.0),
                usage: BufferUsages::UNIFORM,
            })
            .into();
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
        let shape_buffer = context
            .device
            .create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&shape.0),
                usage: BufferUsages::UNIFORM,
            })
            .into();
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
        source: &TensorGpu<'_, T, ReadWrite>,
        destination: &TensorGpu<'_, T, K>,
    ) -> Result<(), TensorError> {
        if source.shape != destination.shape {
            return Err(TensorError::Shape(source.shape, destination.shape));
        }
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

pub trait TensorPass<'a> {
    fn execute_tensor_op(&mut self, op: &'a TensorOp);
}

impl<'a, 'b> TensorPass<'a> for ComputePass<'b>
where
    'a: 'b,
{
    fn execute_tensor_op(&mut self, op: &'a TensorOp) {
        self.set_pipeline(op.pipeline);
        self.set_bind_group(0, &op.bind_group, &[]);
        self.dispatch_workgroups(op.dispatch.0, op.dispatch.1, op.dispatch.2);
    }
}

pub struct TensorOp<'a> {
    pub pipeline: &'a ComputePipeline,
    pub bind_group: BindGroup,
    pub dispatch: (u32, u32, u32),
}

impl<'a> TensorOp<'a> {
    pub fn softmax(
        x: &'a TensorGpu<'a, f32, ReadWrite>,
        out: &'a TensorGpu<'a, f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        if x.context != out.context {
            return Err(TensorError::DeviceError);
        }
        if x.shape != out.shape {
            return Err(TensorError::Shape(x.shape, out.shape));
        }

        let context = x.context;
        let pipeline = context
            .pipelines
            .get("softmax")
            .ok_or(TensorError::PipelineError)?;
        let bind_group = context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out.binding(),
                },
            ],
        });

        let shape = x.shape;
        let dispatch = (1, shape[1] as u32, shape[2] as u32);

        Ok(Self {
            pipeline,
            bind_group,
            dispatch,
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
    use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor, PowerPreference};

    use crate::{
        context::{ContextBuilder, Instance},
        tensor::{TensorCommand, TensorCpu, TensorGpu, TensorShape},
    };

    use super::{TensorOp, TensorPass};

    #[test]
    fn test_shape_index() {
        let shape = TensorShape([1024, 768, 12, 1]);
        let indices = TensorShape([35, 42, 9, 0]);
        let index = shape.shape_index(indices);
        assert_eq!(index, 35 + 42 * 1024 + 9 * 1024 * 768);
    }

    #[test]
    fn test_copy() -> Result<(), anyhow::Error> {
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

        let x = vec![0.0, 1.5, 2.0, -1.0];
        let shape = TensorShape([x.len(), 1, 1, 1]);

        let x_device: TensorGpu<_, _> = context.tensor_from_data(None, shape, x.clone())?;
        let x_map = context.tensor_init(None, x_device.shape());

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_tensor(&x_device, &x_map)?;
        context.queue.submit(Some(encoder.finish()));

        let x_host = TensorCpu::from(x_map);
        let x_vec = Vec::from(x_host);

        assert_eq!(x, x_vec);
        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<(), anyhow::Error> {
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

        let x = [(); 1000].map(|_| 10.0 * (fastrand::f32() - 0.5)).to_vec();
        let shape = TensorShape([x.len(), 1, 1, 1]);

        let x_device: TensorGpu<_, _> = context.tensor_from_data(None, shape, x.clone())?;
        let x_out = context.tensor_init(None, x_device.shape());
        let x_map = context.tensor_init(None, x_device.shape());

        let softmax = TensorOp::softmax(&x_device, &x_out)?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&softmax);
        drop(pass);

        encoder.copy_tensor(&x_out, &x_map)?;
        context.queue.submit(Some(encoder.finish()));

        let x_host = TensorCpu::from(x_map);
        let x_vec = Vec::from(x_host);

        let ans = x.into_iter();
        let max = ans.clone().reduce(f32::max).unwrap_or_default();
        let ans = ans.map(|x| (x - max).exp());
        let sum: f32 = ans.clone().sum();
        let ans: Vec<_> = ans.map(|x| x / sum).collect();

        for (a, b) in x_vec.into_iter().zip(ans.into_iter()) {
            let d = (a - b).abs();
            assert!(d <= f32::max(f32::EPSILON, f32::max(a.abs(), b.abs()) * f32::EPSILON));
        }

        Ok(())
    }
}
