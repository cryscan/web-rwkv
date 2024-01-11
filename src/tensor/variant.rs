use std::borrow::Cow;

use half::f16;
use wgpu::BindingResource;

use super::{
    shape::{Shape, TensorAxis, TensorDimension},
    DeepClone, ReadWrite, TensorBuffer, TensorError, TensorGpu, TensorInit, TensorReshape,
    TensorShape, TensorView,
};
use crate::context::Context;

/// A GPU tensor that is either fp32 or fp16.
#[derive(Debug, Clone)]
pub enum TensorGpuFloat {
    Fp32(TensorGpu<f32, ReadWrite>),
    Fp16(TensorGpu<f16, ReadWrite>),
}

impl From<TensorGpu<f32, ReadWrite>> for TensorGpuFloat {
    fn from(value: TensorGpu<f32, ReadWrite>) -> Self {
        Self::Fp32(value)
    }
}

impl From<TensorGpu<f16, ReadWrite>> for TensorGpuFloat {
    fn from(value: TensorGpu<f16, ReadWrite>) -> Self {
        Self::Fp16(value)
    }
}

impl std::ops::Deref for TensorGpuFloat {
    type Target = TensorBuffer;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Fp32(x) => x.data(),
            Self::Fp16(x) => x.data(),
        }
    }
}

impl DeepClone for TensorGpuFloat {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Fp32(x) => Self::Fp32(x.deep_clone()),
            Self::Fp16(x) => Self::Fp16(x.deep_clone()),
        }
    }
}

impl<'a> TensorInit<'a, f32> for TensorGpuFloat {
    fn from_data(
        context: &Context,
        shape: Shape,
        data: impl Into<Cow<'a, [f32]>>,
    ) -> Result<Self, TensorError> {
        Ok(Self::Fp32(TensorGpu::from_data(context, shape, data)?))
    }

    fn init(context: &Context, shape: Shape) -> Self {
        Self::Fp32(TensorGpu::init(context, shape))
    }
}

impl<'a> TensorInit<'a, f16> for TensorGpuFloat {
    fn from_data(
        context: &Context,
        shape: Shape,
        data: impl Into<Cow<'a, [f16]>>,
    ) -> Result<Self, TensorError> {
        Ok(Self::Fp16(TensorGpu::from_data(context, shape, data)?))
    }

    fn init(context: &Context, shape: Shape) -> Self {
        Self::Fp16(TensorGpu::init(context, shape))
    }
}

impl TensorShape for TensorGpuFloat {
    fn shape(&self) -> Shape {
        match self {
            Self::Fp32(x) => x.shape(),
            Self::Fp16(x) => x.shape(),
        }
    }
}

impl TensorReshape for TensorGpuFloat {
    fn reshape(
        &self,
        x: TensorDimension,
        y: TensorDimension,
        z: TensorDimension,
        w: TensorDimension,
    ) -> Result<Self, TensorError> {
        match self {
            Self::Fp32(t) => Ok(Self::Fp32(t.reshape(x, y, z, w)?)),
            Self::Fp16(t) => Ok(Self::Fp16(t.reshape(x, y, z, w)?)),
        }
    }
}

impl TensorGpuFloat {
    #[inline]
    pub fn context(&self) -> &Context {
        match self {
            Self::Fp32(x) => x.context(),
            Self::Fp16(x) => x.context(),
        }
    }

    #[inline]
    pub fn data(&self) -> &TensorBuffer {
        match self {
            Self::Fp32(x) => x.data(),
            Self::Fp16(x) => x.data(),
        }
    }

    #[inline]
    pub fn view(
        &self,
        x: impl TensorAxis,
        y: impl TensorAxis,
        z: impl TensorAxis,
        w: impl TensorAxis,
    ) -> Result<TensorViewFloat<'_>, TensorError> {
        match self {
            Self::Fp32(t) => Ok(TensorViewFloat::Fp32(t.view(x, y, z, w)?)),
            Self::Fp16(t) => Ok(TensorViewFloat::Fp16(t.view(x, y, z, w)?)),
        }
    }

    #[inline]
    pub fn literal(&self) -> Cow<'_, [u8]> {
        match self {
            Self::Fp32(_) => Cow::Borrowed("FP32"),
            Self::Fp16(_) => Cow::Borrowed("FP16"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TensorViewFloat<'a> {
    Fp32(TensorView<'a, f32>),
    Fp16(TensorView<'a, f16>),
}

impl TensorShape for TensorViewFloat<'_> {
    fn shape(&self) -> Shape {
        match self {
            Self::Fp32(x) => x.shape(),
            Self::Fp16(x) => x.shape(),
        }
    }
}

impl TensorViewFloat<'_> {
    #[inline]
    fn tensor(&self) -> TensorGpuFloat {
        match self {
            Self::Fp32(x) => TensorGpuFloat::Fp32(x.tensor().clone()),
            Self::Fp16(x) => TensorGpuFloat::Fp16(x.tensor().clone()),
        }
    }

    #[inline]
    pub fn context(&self) -> &Context {
        match self {
            Self::Fp32(x) => x.context(),
            Self::Fp16(x) => x.context(),
        }
    }

    #[inline]
    pub fn data(&self) -> &TensorBuffer {
        match self {
            Self::Fp32(x) => x.data(),
            Self::Fp16(x) => x.data(),
        }
    }

    #[inline]
    pub fn meta_binding(&self) -> BindingResource {
        match self {
            Self::Fp32(x) => x.meta_binding(),
            Self::Fp16(x) => x.meta_binding(),
        }
    }

    #[inline]
    pub fn binding(&self) -> BindingResource {
        match self {
            Self::Fp32(x) => x.binding(),
            Self::Fp16(x) => x.binding(),
        }
    }

    #[inline]
    pub fn literal(&self) -> Cow<'_, [u8]> {
        match self {
            Self::Fp32(_) => Cow::Borrowed("FP32"),
            Self::Fp16(_) => Cow::Borrowed("FP16"),
        }
    }
}
