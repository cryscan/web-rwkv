use std::{fmt, marker::PhantomData};

use serde::{
    de::{DeserializeSeed, Error, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};

use super::{kind::Kind, shape::Shape, Cpu, Device, TensorCpu, TensorGpu};
use crate::{context::Context, num::Scalar, tensor::TensorInto};

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
    pub context: Context,
    _phantom: PhantomData<T>,
}

impl<T> Seed<T> {
    pub fn new(context: Context) -> Self {
        Self {
            context,
            _phantom: PhantomData,
        }
    }
}

impl<'de, 'a, T: Scalar + Deserialize<'de>> DeserializeSeed<'de> for Seed<TensorCpu<'a, T>> {
    type Value = TensorCpu<'a, T>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Deserialize::deserialize(deserializer)
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

impl<'de, T> DeserializeSeed<'de> for Seed<Vec<T>>
where
    Seed<T>: DeserializeSeed<'de, Value = T>,
{
    type Value = Vec<T>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct VecVisitor<T> {
            context: Context,
            marker: PhantomData<T>,
        }

        impl<'de, T> Visitor<'de> for VecVisitor<T>
        where
            Seed<T>: DeserializeSeed<'de, Value = T>,
        {
            type Value = Vec<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::<T>::new();

                while let Some(value) =
                    seq.next_element_seed(Seed::<T>::new(self.context.clone()))?
                {
                    values.push(value);
                }

                Ok(values)
            }
        }

        let visitor: VecVisitor<T> = VecVisitor {
            context: self.context.clone(),
            marker: PhantomData,
        };
        deserializer.deserialize_seq(visitor)
    }
}

impl<'de, T> DeserializeSeed<'de> for Seed<Option<T>>
where
    Seed<T>: DeserializeSeed<'de, Value = T>,
{
    type Value = Option<T>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct OptionVisitor<T> {
            context: Context,
            marker: PhantomData<T>,
        }

        impl<'de, T> Visitor<'de> for OptionVisitor<T>
        where
            Seed<T>: DeserializeSeed<'de, Value = T>,
        {
            type Value = Option<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("option")
            }

            #[inline]
            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(None)
            }

            #[inline]
            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(None)
            }

            #[inline]
            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                let seed = Seed::<T>::new(self.context.clone());
                DeserializeSeed::deserialize(seed, deserializer).map(Some)
            }

            fn __private_visit_untagged_option<D>(self, deserializer: D) -> Result<Self::Value, ()>
            where
                D: Deserializer<'de>,
            {
                let seed = Seed::<T>::new(self.context.clone());
                Ok(DeserializeSeed::deserialize(seed, deserializer).ok())
            }
        }

        let visitor: OptionVisitor<T> = OptionVisitor {
            context: self.context.clone(),
            marker: PhantomData,
        };
        deserializer.deserialize_option(visitor)
    }
}
