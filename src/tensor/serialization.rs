use std::{marker::PhantomData, sync::Mutex};

use serde::{Deserialize, Serialize};

use super::{kind::ReadWrite, shape::Shape, Cpu, Device, TensorCpu, TensorGpu};
use crate::{context::Context, num::Scalar};

static _CONTEXT: Mutex<Option<Context>> = Mutex::new(None);

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

impl<T: Scalar + Serialize> Serialize for TensorGpu<T, ReadWrite> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        TensorBlob::from(self.back()).serialize(serializer)
    }
}

impl<'de, T: Scalar + Deserialize<'de>> Deserialize<'de> for TensorGpu<T, ReadWrite> {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!()
    }
}
