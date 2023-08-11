use derive_getters::Getters;
use half::prelude::*;
use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use crate::Environment;

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
pub enum TensorData {
    CpuFp32(Vec<f32>),
    CpuFp16(Vec<f16>),
    CpuInt8(Vec<u8>),
    GpuFp32(Arc<Buffer>),
    GpuFp16(Arc<Buffer>),
    GpuInt8(Arc<Buffer>),
}

#[derive(Debug, Clone, Getters)]
pub struct Tensor {
    shape: TensorShape,
    data: TensorData,
}

impl Tensor {
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

impl Environment {
    pub fn create_tensor_cpu_f32(
        &self,
        shape: TensorShape,
        data: Vec<f32>,
    ) -> Result<Tensor, TensorError> {
        if shape.len() != data.len() {
            return Err(TensorError::CreateShapeNotMatch(shape.len(), data.len()));
        }
        Ok(Tensor {
            shape,
            data: TensorData::CpuFp32(data),
        })
    }

    pub fn create_tensor_cpu_f16(
        &self,
        shape: TensorShape,
        data: Vec<f16>,
    ) -> Result<Tensor, TensorError> {
        if shape.len() != data.len() {
            return Err(TensorError::CreateShapeNotMatch(shape.len(), data.len()));
        }
        Ok(Tensor {
            shape,
            data: TensorData::CpuFp16(data),
        })
    }

    pub fn create_tensor_device(&self, shape: TensorShape, data_type: DataType) -> Tensor {
        let size = data_type.size() as u64 * shape.len() as u64;
        let buffer = Arc::new(self.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        Tensor {
            shape,
            data: match data_type {
                DataType::Fp32 => TensorData::GpuFp32(buffer),
                DataType::Fp16 => TensorData::GpuFp16(buffer),
                DataType::Int8 => TensorData::GpuInt8(buffer),
            },
        }
    }
}
