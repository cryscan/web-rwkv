use std::{borrow::Cow, fmt, marker::PhantomData, sync::Arc};

use serde::{
    de::{DeserializeSeed, Error, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};

use super::{kind::Kind, shape::Shape, TensorCpu, TensorGpu};
use crate::{context::Context, num::Scalar};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
struct TensorBlob<'a> {
    shape: Shape,
    #[serde(with = "serde_bytes")]
    data: Cow<'a, [u8]>,
}

impl<T: Scalar> From<TensorCpu<T>> for TensorBlob<'_> {
    fn from(value: TensorCpu<T>) -> Self {
        let TensorCpu { shape, data, .. } = value;
        let data = bytemuck::cast_slice(&data).to_vec().into();
        Self { shape, data }
    }
}

impl<T: Scalar> From<TensorBlob<'_>> for TensorCpu<T> {
    fn from(value: TensorBlob) -> Self {
        let TensorBlob { shape, data } = value;
        let data = data.to_vec().into_boxed_slice();
        // let data: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        let data = Box::leak(data);
        let data: Box<[T]> = unsafe {
            let ptr = data.as_ptr() as *const T;
            let len = data.len() / size_of::<T>();
            let slice = core::slice::from_raw_parts(ptr, len);
            Box::from(slice)
        };
        let data = data.into();
        Self {
            shape,
            data,
            id: uid::Id::new(),
            phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Serialize> Serialize for TensorCpu<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        TensorBlob::from(self.clone()).serialize(serializer)
    }
}

impl<'de, T: Scalar + Deserialize<'de>> Deserialize<'de> for TensorCpu<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        TensorBlob::deserialize(deserializer).map(Into::into)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Scalar + Serialize, K: Kind> Serialize for TensorGpu<T, K> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        TensorBlob::from(self.back_in_place()).serialize(serializer)
    }
}

#[cfg(target_arch = "wasm32")]
impl<T: Scalar + Serialize, K: Kind> Serialize for TensorGpu<T, K> {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unimplemented!()
    }
}

impl Serialize for Context {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        PhantomData::<Context>::serialize(&PhantomData, serializer)
    }
}

pub struct Seed<'a, Context, Product> {
    pub context: &'a Context,
    _phantom: PhantomData<Product>,
}

impl<'a, Context, Product> Seed<'a, Context, Product> {
    pub fn new(context: &'a Context) -> Self {
        Self {
            context,
            _phantom: PhantomData,
        }
    }
}

impl<'de> DeserializeSeed<'de> for Seed<'de, Context, Context> {
    type Value = Context;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        <PhantomData<Context> as Deserialize<'de>>::deserialize(deserializer)?;
        Ok(self.context.clone())
    }
}

impl<'de, T: Scalar + Deserialize<'de>> DeserializeSeed<'de> for Seed<'de, Context, TensorCpu<T>> {
    type Value = TensorCpu<T>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Deserialize::deserialize(deserializer)
    }
}

impl<'de, T: Deserialize<'de>> DeserializeSeed<'de> for Seed<'de, Context, Arc<T>> {
    type Value = Arc<T>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Deserialize::deserialize(deserializer)
    }
}

impl<'de, T: Scalar + Deserialize<'de>, K: Kind> DeserializeSeed<'de>
    for Seed<'de, Context, TensorGpu<T, K>>
{
    type Value = TensorGpu<T, K>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let context = &self.context;
        let tensor: TensorBlob<'de> = Deserialize::deserialize(deserializer)?;
        let tensor = TensorGpu::from_data_u8(context, tensor.shape, &tensor.data);
        context.queue.submit(None);
        tensor.map_err(D::Error::custom)
    }
}

#[macro_export]
macro_rules! impl_deserialize_seed {
    ($tt:tt) => {
        impl<'de, C> serde::de::DeserializeSeed<'de>
            for $crate::tensor::serialization::Seed<'de, C, $tt>
        {
            type Value = $tt;

            fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: serde::de::Deserializer<'de>,
            {
                $tt::deserialize(deserializer)
            }
        }
    };
    ($tt:tt, $gt:tt) => {
        impl<'de, C, $gt> serde::de::DeserializeSeed<'de>
            for $crate::tensor::serialization::Seed<'de, C, $tt<$gt>>
        {
            type Value = $tt<$gt>;

            fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: serde::de::Deserializer<'de>,
            {
                $tt::<$gt>::deserialize(deserializer)
            }
        }
    };
}

impl_deserialize_seed!(bool);
impl_deserialize_seed!(usize);
impl_deserialize_seed!(PhantomData, T);

impl<'de, C, T> DeserializeSeed<'de> for Seed<'de, C, Vec<T>>
where
    Seed<'de, C, T>: DeserializeSeed<'de, Value = T>,
{
    type Value = Vec<T>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct VecVisitor<'de, C, T> {
            context: &'de C,
            marker: PhantomData<T>,
        }

        impl<'de, C, T> Visitor<'de> for VecVisitor<'de, C, T>
        where
            Seed<'de, C, T>: DeserializeSeed<'de, Value = T>,
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

                while let Some(value) = seq.next_element_seed(Seed::<C, T>::new(self.context))? {
                    values.push(value);
                }

                Ok(values)
            }
        }

        let visitor: VecVisitor<C, T> = VecVisitor {
            context: self.context,
            marker: PhantomData,
        };
        deserializer.deserialize_seq(visitor)
    }
}

impl<'de, C, T> DeserializeSeed<'de> for Seed<'de, C, Option<T>>
where
    Seed<'de, C, T>: DeserializeSeed<'de, Value = T>,
{
    type Value = Option<T>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct OptionVisitor<'de, C, T> {
            context: &'de C,
            marker: PhantomData<T>,
        }

        impl<'de, C, T> Visitor<'de> for OptionVisitor<'de, C, T>
        where
            Seed<'de, C, T>: DeserializeSeed<'de, Value = T>,
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
                let seed = Seed::<C, T>::new(self.context);
                DeserializeSeed::deserialize(seed, deserializer).map(Some)
            }

            fn __private_visit_untagged_option<D>(self, deserializer: D) -> Result<Self::Value, ()>
            where
                D: Deserializer<'de>,
            {
                let seed = Seed::<C, T>::new(self.context);
                Ok(DeserializeSeed::deserialize(seed, deserializer).ok())
            }
        }

        let visitor: OptionVisitor<C, T> = OptionVisitor {
            context: self.context,
            marker: PhantomData,
        };
        deserializer.deserialize_option(visitor)
    }
}
