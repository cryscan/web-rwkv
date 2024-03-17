use std::marker::PhantomData;

use half::f16;
use serde::{de::DeserializeSeed, Deserialize, Serialize};

use super::{kind::Kind, matrix::Matrix, shape::Shape, Cpu, Device, TensorCpu, TensorGpu};
use crate::{
    context::Context,
    num::Scalar,
    tensor::{TensorFrom, TensorInto},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
struct TensorBlob<'a, T: Scalar> {
    shape: Shape,
    data: <Cpu<'a, T> as Device>::Data,
}

impl<'a, T: Scalar> From<TensorCpu<'a, T>> for TensorBlob<'a, T> {
    fn from(value: TensorCpu<'a, T>) -> Self {
        let TensorCpu { shape, data, .. } = value;
        Self { shape, data }
    }
}

impl<'a, T: Scalar> From<TensorBlob<'a, T>> for TensorCpu<'a, T> {
    fn from(value: TensorBlob<'a, T>) -> Self {
        let TensorBlob { shape, data } = value;
        Self {
            shape,
            data,
            phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Serialize> Serialize for TensorCpu<'_, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        TensorBlob::from(self.clone()).serialize(serializer)
    }
}

impl<'de, T: Scalar + Deserialize<'de>> Deserialize<'de> for TensorCpu<'_, T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        TensorBlob::deserialize(deserializer).map(Into::into)
    }
}

impl<T: Scalar + Serialize, K: Kind> Serialize for TensorGpu<T, K> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        TensorBlob::from(self.back()).serialize(serializer)
    }
}

pub struct Seed<T> {
    context: Context,
    _phantom: PhantomData<T>,
}

impl<T> Seed<T> {
    pub fn new(context: &Context) -> Self {
        Self {
            context: context.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<'de, T: Scalar + Deserialize<'de>, K: Kind> DeserializeSeed<'de> for Seed<TensorGpu<T, K>> {
    type Value = TensorGpu<T, K>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let context = &self.context;
        let tensor: TensorCpu<T> = Deserialize::deserialize(deserializer)?;
        Ok(tensor.transfer_into(context))
    }
}

impl<'de> DeserializeSeed<'de> for Seed<Matrix> {
    type Value = Matrix;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // #[derive(Deserialize)]
        // enum _Matrix<'a> {
        //     Fp16(TensorCpu<'a, f16>),
        //     Int8 {
        //         w: TensorCpu<'a, u8>,
        //         m: TensorCpu<'a, f16>,
        //     },
        //     NF4 {
        //         q: TensorCpu<'a, f32>,
        //         w: TensorCpu<'a, u8>,
        //         m: TensorCpu<'a, f16>,
        //     },
        // }

        // impl<'a> TensorFrom<_Matrix<'a>> for Matrix {
        //     fn transfer_from(context: &Context, value: _Matrix) -> Self {
        //         match value {
        //             _Matrix::Fp16(x) => Matrix::Fp16(x.transfer_into(context)),
        //             _Matrix::Int8 { w, m } => Matrix::Int8 {
        //                 w: w.transfer_into(context),
        //                 m: m.transfer_into(context),
        //             },
        //             _Matrix::NF4 { q, w, m } => Matrix::NF4 {
        //                 q: q.transfer_into(context),
        //                 w: w.transfer_into(context),
        //                 m: m.transfer_into(context),
        //             },
        //         }
        //     }
        // }

        let context = &self.context;

        todo!()
    }
}
