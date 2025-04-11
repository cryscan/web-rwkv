use std::{marker::PhantomData, sync::Arc};

use derive_more::{Deref, DerefMut, Display, From, Into};
use itertools::Itertools;

use super::{device::Device, layout::Layout};
use crate::num::Scalar;

#[derive(Debug, Clone)]
pub struct Tensor<D: Device, T: Scalar> {
    pub device: D,
    pub layout: Layout,
    pub slice: Slice,
    pub data: Arc<D::Data>,
    pub id: uid::Id<TensorId>,
    pub phantom: PhantomData<T>,
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

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deref, DerefMut, From, Into, Display)]
#[display("[{}]", _0.iter().format(", "))]
pub struct Slice(pub Vec<Axis>);

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
