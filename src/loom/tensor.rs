use std::{marker::PhantomData, sync::Arc};

use super::layout::Layout;
use crate::num::Scalar;

pub trait Device {
    type Data;
}

#[derive(Debug)]
pub struct Cpu<T: Scalar>(PhantomData<T>);

impl<T: Scalar> Device for Cpu<T> {
    type Data = Arc<[T]>;
}

#[derive(Debug)]
pub struct Gpu;

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Tensor<D: Device, T: Scalar> {
    layout: Layout,
    slice: Slice,
    data: D::Data,
    id: uid::Id<TensorId>,
    phantom: PhantomData<T>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    #[default]
    Full,
    One(usize),
}

impl From<usize> for Axis {
    fn from(value: usize) -> Self {
        Self::One(value)
    }
}

impl From<std::ops::RangeFull> for Axis {
    fn from(_: std::ops::RangeFull) -> Self {
        Self::Full
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Slice(Vec<Axis>);

impl<T0> From<T0> for Slice
where
    T0: Into<Axis>,
{
    fn from(t0: T0) -> Self {
        Self(vec![t0.into()])
    }
}
