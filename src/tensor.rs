use derive_getters::Getters;
use half::prelude::*;
use std::{borrow::Cow, sync::Arc};
use web_rwkv_derive::{Deref, TensorBuffer};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferDescriptor, BufferUsages,
};

use crate::Context;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    #[default]
    Fp32,
    Fp16,
    Int8,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::Fp32 => 4,
            DataType::Fp16 => 2,
            DataType::Int8 => 1,
        }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorInfo {
    shape: TensorShape,
    data_type: DataType,
}

macro_rules! impl_tensor_conversion {
    ($tensor_type:ty, $tensor_variant:ident, $dual_variant:ident, $data_type:ident, $dual_function:ident) => {
        impl<'a> TryFrom<Tensor<'a, TensorData<'a>>> for Tensor<'a, $tensor_type> {
            type Error = TensorError;

            fn try_from(value: Tensor<'a, TensorData<'a>>) -> Result<Self, Self::Error> {
                let data_type = value.data_type();
                match value.data {
                    TensorData::$dual_variant(_) => Self::try_from(value.$dual_function()),
                    TensorData::$tensor_variant(buffer) => {
                        let Tensor {
                            context,
                            label,
                            shape,
                            shape_buffer,
                            ..
                        } = value;
                        Ok(Self {
                            data: $tensor_variant(buffer),
                            context,
                            label,
                            shape,
                            shape_buffer,
                        })
                    }
                    _ => Err(TensorError::DataType(data_type, DataType::$data_type)),
                }
            }
        }

        impl<'a> From<Tensor<'a, $tensor_type>> for Tensor<'a, TensorData<'a>> {
            fn from(value: Tensor<'a, $tensor_type>) -> Self {
                let Tensor {
                    context,
                    label,
                    shape,
                    shape_buffer,
                    data: $tensor_variant(buffer),
                } = value;
                Self {
                    context,
                    label,
                    shape,
                    shape_buffer,
                    data: TensorData::$tensor_variant(buffer),
                }
            }
        }
    };
}

impl_tensor_conversion!(CpuFp32<'a>, CpuFp32, GpuFp32, Fp32, cpu);
impl_tensor_conversion!(CpuFp16<'a>, CpuFp16, GpuFp16, Fp16, cpu);
impl_tensor_conversion!(CpuInt8<'a>, CpuInt8, GpuInt8, Int8, cpu);
impl_tensor_conversion!(GpuFp32, GpuFp32, CpuFp32, Fp32, device);
impl_tensor_conversion!(GpuFp16, GpuFp16, CpuFp16, Fp16, device);
impl_tensor_conversion!(GpuInt8, GpuInt8, CpuInt8, Int8, device);

macro_rules! impl_tensor_index {
    ($tensor_type:ty, $output_type:ty) => {
        impl std::ops::Index<usize> for Tensor<'_, $tensor_type> {
            type Output = $output_type;

            fn index(&self, index: usize) -> &Self::Output {
                &self.data[index]
            }
        }

        impl std::ops::Index<std::ops::Range<usize>> for Tensor<'_, $tensor_type> {
            type Output = [$output_type];

            fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
                &self.data[index]
            }
        }
    };
}

impl_tensor_index!(CpuFp32<'_>, f32);
impl_tensor_index!(CpuFp16<'_>, f16);
impl_tensor_index!(CpuInt8<'_>, u8);

#[derive(Debug, Clone)]
pub enum TensorData<'a> {
    CpuFp32(Cow<'a, [f32]>),
    CpuFp16(Cow<'a, [f16]>),
    CpuInt8(Cow<'a, [u8]>),
    GpuFp32(Arc<Buffer>),
    GpuFp16(Arc<Buffer>),
    GpuInt8(Arc<Buffer>),
}

impl TensorBuffer<'_> for TensorData<'_> {
    fn buffer(&self) -> Result<&Buffer, TensorError> {
        match self {
            TensorData::GpuFp32(buffer)
            | TensorData::GpuFp16(buffer)
            | TensorData::GpuInt8(buffer) => Ok(buffer),
            _ => Err(TensorError::DeviceError),
        }
    }

    fn data_type(&self) -> DataType {
        match self {
            Self::CpuFp32(_) | Self::GpuFp32(_) => DataType::Fp32,
            Self::CpuFp16(_) | Self::GpuFp16(_) => DataType::Fp16,
            Self::CpuInt8(_) | Self::GpuInt8(_) => DataType::Int8,
        }
    }
}

#[derive(Debug, Clone, Deref, TensorBuffer)]
#[data_type(Fp32)]
pub struct CpuFp32<'a>(Cow<'a, [f32]>);

#[derive(Debug, Clone, Deref, TensorBuffer)]
#[data_type(Fp16)]
pub struct CpuFp16<'a>(Cow<'a, [f16]>);

#[derive(Debug, Clone, Deref, TensorBuffer)]
#[data_type(Int8)]
pub struct CpuInt8<'a>(Cow<'a, [u8]>);

#[derive(Debug, Clone, Deref, TensorBuffer)]
#[data_type(Fp32)]
pub struct GpuFp32(Arc<Buffer>);

#[derive(Debug, Clone, Deref, TensorBuffer)]
#[data_type(Fp16)]
pub struct GpuFp16(Arc<Buffer>);

#[derive(Debug, Clone, Deref, TensorBuffer)]
#[data_type(Int8)]
pub struct GpuInt8(Arc<Buffer>);

#[derive(Debug, Clone, Copy)]
pub enum TensorError {
    Size(usize, usize),
    DataType(DataType, DataType),
    DeviceError,
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::Size(a, b) => write!(f, "data size not match: {} vs. {}", a, b),
            TensorError::DataType(a, b) => write!(f, "data type not match: {:?} vs. {:?}", a, b),
            TensorError::DeviceError => write!(f, "tensor not on device"),
        }
    }
}

impl std::error::Error for TensorError {}

pub trait TensorBuffer<'a> {
    fn data_type(&'a self) -> DataType;
    fn buffer(&'a self) -> Result<&'a Buffer, TensorError>;
}

#[derive(Debug, Clone, Getters)]
pub struct Tensor<'a, T> {
    context: Context,
    label: Option<&'a str>,
    shape: TensorShape,
    shape_buffer: Arc<Buffer>,
    data: T,
}

impl<'a, T: TensorBuffer<'a>> Tensor<'a, T> {
    pub fn data_type(&'a self) -> DataType {
        self.data.data_type()
    }

    pub fn info(&'a self) -> TensorInfo {
        TensorInfo {
            shape: self.shape,
            data_type: self.data_type(),
        }
    }

    pub fn buffer(&'a self) -> Result<&'a Buffer, TensorError> {
        self.data.buffer()
    }

    pub fn byte_offset(&'a self, offset: usize) -> usize {
        offset * self.data_type().size()
    }
}

impl<'a> Tensor<'a, TensorData<'a>> {
    /// Upload the [`Tensor`] to device.
    pub fn device(self) -> Tensor<'a, TensorData<'a>> {
        let Tensor {
            context,
            label,
            shape,
            shape_buffer,
            data,
        } = self;
        let contents = match &data {
            TensorData::CpuFp32(data) => Some(bytemuck::cast_slice(data)),
            TensorData::CpuFp16(data) => Some(bytemuck::cast_slice(data)),
            TensorData::CpuInt8(data) => Some(bytemuck::cast_slice(data)),
            TensorData::GpuFp32(_) | TensorData::GpuFp16(_) | TensorData::GpuInt8(_) => None,
        };
        let data = match contents {
            Some(contents) => {
                let buffer = context.device.create_buffer_init(&BufferInitDescriptor {
                    label,
                    contents,
                    usage: BufferUsages::STORAGE
                        | BufferUsages::COPY_DST
                        | BufferUsages::COPY_SRC
                        | BufferUsages::MAP_READ,
                });
                match data.data_type() {
                    DataType::Fp32 => TensorData::GpuFp32(Arc::new(buffer)),
                    DataType::Fp16 => TensorData::GpuFp16(Arc::new(buffer)),
                    DataType::Int8 => TensorData::GpuInt8(Arc::new(buffer)),
                }
            }
            None => data,
        };
        Tensor {
            context,
            label,
            shape,
            shape_buffer,
            data,
        }
    }

    /// Read back the [`Tensor`] to CPU.
    pub fn cpu(self) -> Tensor<'a, TensorData<'a>> {
        let Tensor {
            context,
            label,
            shape,
            shape_buffer,
            data,
        } = self;
        let buffer = match &data {
            TensorData::GpuFp32(buffer)
            | TensorData::GpuFp16(buffer)
            | TensorData::GpuInt8(buffer) => Some(buffer.clone()),
            _ => None,
        };
        let data = match buffer {
            Some(buffer) => {
                let slice = buffer.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});

                context.device.poll(wgpu::MaintainBase::Wait);

                let map = slice.get_mapped_range();
                let data = match data {
                    TensorData::GpuFp32(_) => {
                        TensorData::CpuFp32(Cow::from(bytemuck::cast_slice(&map).to_owned()))
                    }
                    TensorData::GpuFp16(_) => {
                        TensorData::CpuFp16(Cow::from(bytemuck::cast_slice(&map).to_owned()))
                    }
                    TensorData::GpuInt8(_) => {
                        TensorData::CpuInt8(Cow::from(bytemuck::cast_slice(&map).to_owned()))
                    }
                    _ => unreachable!(),
                };
                buffer.unmap();
                data
            }
            None => data,
        };
        Tensor {
            context,
            label,
            shape,
            shape_buffer,
            data,
        }
    }
}

impl Context {
    pub fn create_shape_uniform(&self, shape: TensorShape) -> Arc<Buffer> {
        let buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&shape.0),
            usage: BufferUsages::UNIFORM,
        });
        Arc::new(buffer)
    }

    pub fn create_tensor_cpu_f32<'a, 'b>(
        &'a self,
        label: Option<&'b str>,
        shape: TensorShape,
        data: Vec<f32>,
    ) -> Result<Tensor<'b, TensorData<'b>>, TensorError> {
        if shape.len() != data.len() {
            return Err(TensorError::Size(shape.len(), data.len()));
        }
        Ok(Tensor {
            context: self.clone(),
            label,
            shape,
            shape_buffer: self.create_shape_uniform(shape),
            data: TensorData::CpuFp32(Cow::from(data)),
        })
    }

    pub fn create_tensor_cpu_f16<'a, 'b>(
        &'a self,
        label: Option<&'b str>,
        shape: TensorShape,
        data: Vec<f16>,
    ) -> Result<Tensor<'b, TensorData<'b>>, TensorError> {
        if shape.len() != data.len() {
            return Err(TensorError::Size(shape.len(), data.len()));
        }
        Ok(Tensor {
            context: self.clone(),
            label,
            shape,
            shape_buffer: self.create_shape_uniform(shape),
            data: TensorData::CpuFp16(Cow::from(data)),
        })
    }

    pub fn create_tensor_device<'a, 'b>(
        &'a self,
        label: Option<&'b str>,
        shape: TensorShape,
        data_type: DataType,
    ) -> Tensor<'b, TensorData<'b>> {
        let size = data_type.size() as u64 * shape.len() as u64;
        let buffer = Arc::new(self.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC
                | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        Tensor {
            context: self.clone(),
            label,
            shape,
            shape_buffer: self.create_shape_uniform(shape),
            data: match data_type {
                DataType::Fp32 => TensorData::GpuFp32(buffer),
                DataType::Fp16 => TensorData::GpuFp16(buffer),
                DataType::Int8 => TensorData::GpuInt8(buffer),
            },
        }
    }

    pub fn transfer_tensor<'a, 'b>(
        &'a self,
        tensor: Tensor<'b, TensorData<'b>>,
    ) -> Tensor<'b, TensorData<'b>> {
        if &tensor.context == self {
            return tensor;
        }
        let mut tensor = tensor.cpu();
        tensor.context = self.clone();
        tensor.device()
    }
}
