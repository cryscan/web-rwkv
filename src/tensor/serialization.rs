use std::{marker::PhantomData, sync::RwLock};

use serde::{Deserialize, Serialize};

use super::{kind::Kind, shape::Shape, Cpu, Device, TensorCpu, TensorGpu};
use crate::{context::Context, num::Scalar, tensor::TensorInto};

static CONTEXT: RwLock<Option<Context>> = RwLock::new(None);

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

impl<'de, T: Scalar + Deserialize<'de>, K: Kind> Deserialize<'de> for TensorGpu<T, K> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let context = CONTEXT.read().unwrap();
        let context = context.as_ref().unwrap();

        let tensor: TensorBlob<T> = TensorBlob::deserialize(deserializer)?;
        let tensor = TensorCpu::from(tensor);
        Ok(tensor.transfer_into(context))
    }
}

/// Set the global deserialization context. This *MUST* be called before deserializing the model.
pub fn set_de_context(context: &Context) {
    let mut ctx = CONTEXT.write().unwrap();
    ctx.replace(context.clone());
}

pub fn de_context() -> Context {
    let ctx = CONTEXT.read().unwrap();
    ctx.clone().unwrap()
}
