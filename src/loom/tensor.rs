use std::{marker::PhantomData, sync::Arc};

use derive_more::{Deref, DerefMut, Display, From, Into};

use super::layout::Layout;
use crate::num::Scalar;

pub trait Device {
    type Data: ?Sized;
}

#[derive(Debug)]
pub struct Cpu<T: Scalar>(PhantomData<T>);

impl<T: Scalar> Device for Cpu<T> {
    #[cfg(feature = "tokio")]
    type Data = tokio::sync::RwLock<[T]>;
    #[cfg(not(feature = "tokio"))]
    type Data = std::sync::RwLock<[T]>;
}

#[derive(Debug)]
pub struct Gpu;

impl Device for Gpu {
    type Data = wgpu::Buffer;
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Tensor<D: Device, T: Scalar> {
    layout: Layout,
    slice: Slice,
    data: Arc<D::Data>,
    id: uid::Id<TensorId>,
    phantom: PhantomData<T>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Axis {
    #[default]
    #[display("..")]
    Full,
    #[display("{_0}")]
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

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deref, DerefMut, From, Into)]
pub struct Slice(pub Vec<Axis>);

impl std::fmt::Display for Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (index, axis) in self.0.iter().enumerate() {
            match index {
                x if x + 1 == self.len() => write!(f, "{axis}")?,
                _ => write!(f, "{axis}, ")?,
            }
        }
        write!(f, "]")
    }
}

macro_rules! impl_slice_from {
    ($t:ident, $v:ident) => {
        impl<$t: Into<Axis>> From<$t> for Slice {
            fn from($v: $t) -> Self {
                Self(vec![$v.into()])
            }
        }
    };
    (($($t:ident),+), ($($v:ident),+)) => {
        impl<$($t),+> From<($($t),+)> for Slice
        where
            $($t: Into<Axis>),+
        {
            fn from(($($v),+): ($($t),+)) -> Self {
                Self(vec![$($v.into()),+])
            }
        }
    };
}

impl_slice_from!(T0, t0);
impl_slice_from!((T0, T1), (t0, t1));
impl_slice_from!((T0, T1, T2), (t0, t1, t2));
impl_slice_from!((T0, T1, T2, T3), (t0, t1, t2, t3));
impl_slice_from!((T0, T1, T2, T3, T4), (t0, t1, t2, t3, t4));
impl_slice_from!((T0, T1, T2, T3, T4, T5), (t0, t1, t2, t3, t4, t5));
impl_slice_from!((T0, T1, T2, T3, T4, T5, T6), (t0, t1, t2, t3, t4, t5, t6));

#[cfg(test)]
mod tests {
    use super::Slice;

    #[test]
    fn test_slice() {
        println!("{}", Slice::from(1));
        println!("{}", Slice::from(..));
        println!("{}", Slice::from((0, 1)));
        println!("{}", Slice::from((1, ..)));
        println!("{}", Slice::from((0, .., 1, 5)));
    }
}
