use derive_getters::Getters;
use half::prelude::*;
use std::{borrow::Cow, sync::Arc};
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
    date_type: DataType,
}

#[derive(Debug, Clone)]
pub enum TensorData<'a> {
    CpuFp32(Cow<'a, [f32]>),
    CpuFp16(Cow<'a, [f16]>),
    CpuInt8(Cow<'a, [u8]>),
    GpuFp32(Arc<Buffer>),
    GpuFp16(Arc<Buffer>),
    GpuInt8(Arc<Buffer>),
}

#[derive(Debug, Clone, Copy)]
pub enum TensorError {
    CreateShapeNotMatch(usize, usize),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::CreateShapeNotMatch(shape_len, data_len) => write!(
                f,
                "shape and data length not match: {} vs. {}",
                shape_len, data_len
            ),
        }
    }
}

impl std::error::Error for TensorError {}

#[derive(Debug, Clone, Getters)]
pub struct Tensor<'a> {
    context: Context,
    label: Option<&'a str>,
    shape: TensorShape,
    data: TensorData<'a>,
}

impl<'a> Tensor<'a> {
    pub fn data_type(&self) -> DataType {
        match &self.data {
            TensorData::CpuFp32(_) | TensorData::GpuFp32(_) => DataType::Fp32,
            TensorData::CpuFp16(_) | TensorData::GpuFp16(_) => DataType::Fp16,
            TensorData::CpuInt8(_) | TensorData::GpuInt8(_) => DataType::Int8,
        }
    }

    pub fn info(&self) -> TensorInfo {
        TensorInfo {
            shape: self.shape,
            date_type: self.data_type(),
        }
    }

    /// Upload the [`Tensor`] to device.
    pub fn device(self) -> Tensor<'a> {
        let Tensor {
            context,
            label,
            shape,
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
                    label: label.clone(),
                    contents,
                    usage: BufferUsages::STORAGE
                        | BufferUsages::COPY_DST
                        | BufferUsages::COPY_SRC
                        | BufferUsages::MAP_READ,
                });
                match data {
                    TensorData::CpuFp32(_) => TensorData::GpuFp32(Arc::new(buffer)),
                    TensorData::CpuFp16(_) => TensorData::GpuFp16(Arc::new(buffer)),
                    TensorData::CpuInt8(_) => TensorData::GpuInt8(Arc::new(buffer)),
                    _ => unreachable!(),
                }
            }
            None => data,
        };
        Tensor {
            context,
            label,
            shape,
            data,
        }
    }

    /// Read back the [`Tensor`] to CPU.
    pub fn cpu(self) -> Tensor<'a> {
        let Tensor {
            context,
            label,
            shape,
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
            data,
        }
    }
}

impl Context {
    pub fn create_tensor_cpu_f32<'a>(
        &'a self,
        label: Option<&'a str>,
        shape: TensorShape,
        data: Vec<f32>,
    ) -> Result<Tensor<'a>, TensorError> {
        if shape.len() != data.len() {
            return Err(TensorError::CreateShapeNotMatch(shape.len(), data.len()));
        }
        Ok(Tensor {
            context: self.clone(),
            label,
            shape,
            data: TensorData::CpuFp32(Cow::from(data)),
        })
    }

    pub fn create_tensor_cpu_f16<'a>(
        &'a self,
        label: Option<&'a str>,
        shape: TensorShape,
        data: Vec<f16>,
    ) -> Result<Tensor, TensorError> {
        if shape.len() != data.len() {
            return Err(TensorError::CreateShapeNotMatch(shape.len(), data.len()));
        }
        Ok(Tensor {
            context: self.clone(),
            label,
            shape,
            data: TensorData::CpuFp16(Cow::from(data)),
        })
    }

    pub fn create_tensor_device<'a>(
        &'a self,
        label: Option<&'a str>,
        shape: TensorShape,
        data_type: DataType,
    ) -> Tensor<'a> {
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
            data: match data_type {
                DataType::Fp32 => TensorData::GpuFp32(buffer),
                DataType::Fp16 => TensorData::GpuFp16(buffer),
                DataType::Int8 => TensorData::GpuInt8(buffer),
            },
        }
    }
}
