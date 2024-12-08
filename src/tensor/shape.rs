use std::{cmp::Ordering, hash::Hash};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::TensorError;

pub trait IntoBytes {
    fn into_bytes(self) -> Vec<u8>;
}

/// Indices along each dimension.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapedIndex([usize; 4]);

impl ShapedIndex {
    pub fn new(x: usize, y: usize, z: usize, w: usize) -> Self {
        Self([x, y, z, w])
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> {
        self.0.into_iter()
    }
}

impl From<[usize; 4]> for ShapedIndex {
    fn from(value: [usize; 4]) -> Self {
        Self(value)
    }
}

impl From<(usize, usize, usize, usize)> for ShapedIndex {
    fn from((x, y, z, w): (usize, usize, usize, usize)) -> Self {
        Self([x, y, z, w])
    }
}

impl std::fmt::Display for ShapedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {}, {})", self[0], self[1], self[2], self[3])
    }
}

impl std::ops::Index<usize> for ShapedIndex {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for ShapedIndex {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl std::ops::Add<ShapedIndex> for ShapedIndex {
    type Output = Self;

    fn add(self, rhs: ShapedIndex) -> Self::Output {
        Self::new(
            self[0] + rhs[0],
            self[1] + rhs[1],
            self[2] + rhs[2],
            self[3] + rhs[3],
        )
    }
}

impl std::ops::Sub<ShapedIndex> for ShapedIndex {
    type Output = Self;

    fn sub(self, rhs: ShapedIndex) -> Self::Output {
        Self::new(
            self[0] - rhs[0],
            self[1] - rhs[1],
            self[2] - rhs[2],
            self[3] - rhs[3],
        )
    }
}

impl std::ops::AddAssign<ShapedIndex> for ShapedIndex {
    fn add_assign(&mut self, rhs: ShapedIndex) {
        *self = *self + rhs;
    }
}

impl std::ops::SubAssign<ShapedIndex> for ShapedIndex {
    fn sub_assign(&mut self, rhs: ShapedIndex) {
        *self = *self - rhs;
    }
}

/// The shape of a [`Tensor`](super::Tensor).
/// Note that the fastest-moving axis occupies the lowest shape index, which is opposite to that in `torch`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape([usize; 4]);

impl Shape {
    pub fn new(x: usize, y: usize, z: usize, w: usize) -> Self {
        Self([x, y, z, w])
    }

    pub fn from_slice(slice: &[usize]) -> Self {
        let mut shape = Self::new(1, 1, 1, 1);
        for (index, &dim) in slice.iter().take(4).enumerate() {
            shape[index] = dim;
        }
        shape
    }

    pub fn from_slice_rev(shape: &[usize]) -> Result<Self, TensorError> {
        let shape = match shape[..] {
            [] => Shape::new(0, 0, 0, 0),
            [x] => Shape::new(x, 1, 1, 1),
            [y, x] => Shape::new(x, y, 1, 1),
            [z, y, x] => Shape::new(x, y, z, 1),
            [w, z, y, x] => Shape::new(x, y, z, w),
            _ => return Err(TensorError::Deduce),
        };
        Ok(shape)
    }

    pub fn len(&self) -> usize {
        self.0.into_iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.0.into_iter().any(|x| x == 0)
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> {
        self.0.into_iter()
    }

    /// Convert a shaped index into a linear index.
    pub fn linear_index(&self, index: impl Into<ShapedIndex>) -> usize {
        let index: ShapedIndex = index.into();
        Iterator::zip(self.0.into_iter().rev(), index.0.into_iter().rev())
            .fold(0, |acc, (shape, index)| acc * shape + index)
    }

    /// Iterate through all indices within the shape's bound.
    pub fn cartesian_product(&self) -> impl Iterator<Item = ShapedIndex> {
        (0..self[3])
            .cartesian_product(0..self[2])
            .cartesian_product(0..self[1])
            .cartesian_product(0..self[0])
            .map(|(((w, z), y), x)| ShapedIndex::new(x, y, z, w))
    }
}

impl From<ShapedIndex> for Shape {
    fn from(value: ShapedIndex) -> Self {
        Self(value.0)
    }
}

impl From<[usize; 4]> for Shape {
    fn from(value: [usize; 4]) -> Self {
        Self(value)
    }
}

impl From<Shape> for [usize; 4] {
    fn from(value: Shape) -> Self {
        value.0
    }
}

impl IntoBytes for Shape {
    fn into_bytes(self) -> Vec<u8> {
        let data = self.0.map(|x| x as u32);
        bytemuck::pod_collect_to_vec(&data)
    }
}

impl std::cmp::PartialOrd for Shape {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Ordering::Equal;
        match (
            self[0].cmp(&other[0]),
            self[1].cmp(&other[1]),
            self[2].cmp(&other[2]),
            self[3].cmp(&other[3]),
        ) {
            (x, y, z, w) if x == y && y == z && z == w => Some(x),
            (x, y, z, Equal) if x == y && y == z => Some(x),
            (x, y, Equal, w) if x == y && y == w => Some(y),
            (x, Equal, z, w) if x == z && z == w => Some(z),
            (Equal, y, z, w) if y == z && z == w => Some(w),
            (x, y, Equal, Equal) if x == y => Some(x),
            (x, Equal, z, Equal) if x == z => Some(x),
            (x, Equal, Equal, w) if x == w => Some(x),
            (Equal, y, z, Equal) if y == z => Some(y),
            (Equal, y, Equal, w) if y == w => Some(y),
            (Equal, Equal, z, w) if z == w => Some(z),
            (x, Equal, Equal, Equal) => Some(x),
            (Equal, y, Equal, Equal) => Some(y),
            (Equal, Equal, z, Equal) => Some(z),
            (Equal, Equal, Equal, w) => Some(w),
            _ => None,
        }
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {}, {})", self[0], self[1], self[2], self[3])
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

pub trait TensorSlice {
    fn shaped_bounds(&self, shape: Shape) -> Result<(ShapedIndex, ShapedIndex), TensorError>;
    fn linear_bounds(&self, shape: Shape) -> Result<(usize, usize), TensorError>;
}

pub trait TensorAxis: Clone + PartialEq + Eq + Hash {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError>;
}

#[inline]
fn checked_bounds(dim: usize, start: usize, end: usize) -> Result<(usize, usize), TensorError> {
    if start > end || end - start > dim || end > dim {
        Err(TensorError::SliceOutOfRange { dim, start, end })
    } else {
        Ok((start, end))
    }
}

impl TensorAxis for usize {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        let start = *self;
        let end = start + 1;
        checked_bounds(dim, start, end)
    }
}

impl TensorAxis for std::ops::RangeFull {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        Ok((0, dim))
    }
}

impl TensorAxis for std::ops::Range<usize> {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        checked_bounds(dim, self.start, self.end)
    }
}

impl TensorAxis for std::ops::RangeInclusive<usize> {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        let start = *self.start();
        let end = self.end() + 1;
        checked_bounds(dim, start, end)
    }
}

impl TensorAxis for std::ops::RangeFrom<usize> {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        checked_bounds(dim, self.start, dim)
    }
}

impl TensorAxis for std::ops::RangeTo<usize> {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        checked_bounds(dim, 0, self.end)
    }
}

impl TensorAxis for std::ops::RangeToInclusive<usize> {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        checked_bounds(dim, 0, self.end + 1)
    }
}

// impl<T: std::ops::RangeBounds<usize>> TensorAxis for T {
//     fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
//         let start = match self.start_bound() {
//             Bound::Included(&bound) => bound,
//             Bound::Excluded(&bound) => bound + 1,
//             Bound::Unbounded => 0,
//         };
//         let end = match self.end_bound() {
//             Bound::Included(&bound) => bound + 1,
//             Bound::Excluded(&bound) => bound,
//             Bound::Unbounded => dim,
//         };
//         if start > end || start >= dim || end > dim {
//             Err(TensorError::SliceOutOfRange { dim, start, end })
//         } else {
//             Ok((start, end))
//         }
//     }
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SliceQuantState {
    Zero,
    One,
    Plural,
}

enum SliceFillState {
    NotFull,
    Full,
}

impl<X, Y, Z, W> TensorSlice for (X, Y, Z, W)
where
    X: TensorAxis,
    Y: TensorAxis,
    Z: TensorAxis,
    W: TensorAxis,
{
    fn shaped_bounds(&self, shape: Shape) -> Result<(ShapedIndex, ShapedIndex), TensorError> {
        let mut start = ShapedIndex::default();
        let mut end = ShapedIndex::default();
        (start[0], end[0]) = self.0.bounds(shape[0])?;
        (start[1], end[1]) = self.1.bounds(shape[1])?;
        (start[2], end[2]) = self.2.bounds(shape[2])?;
        (start[3], end[3]) = self.3.bounds(shape[3])?;
        Ok((start, end))
    }

    fn linear_bounds(&self, shape: Shape) -> Result<(usize, usize), TensorError> {
        use SliceFillState::{Full, NotFull};
        use SliceQuantState::{One, Plural, Zero};

        let quant_state = |start, end| match end - start {
            0 => Zero,
            1 => One,
            _ => Plural,
        };

        let fill_state = |start, end, dim| match (start, end) {
            (0, end) if end == dim => Full,
            (start, end) if start == end => Full,
            _ => NotFull,
        };

        let (start, end) = self.shaped_bounds(shape)?;
        let (_, valid) = itertools::multizip((start.iter(), end.iter(), shape.iter())).fold(
            (Full, true),
            |(state, valid), (start, end, dim)| match (state, valid) {
                (Full, valid) => (fill_state(start, end, dim), valid),
                (NotFull, true) => (NotFull, quant_state(start, end) < Plural),
                (NotFull, false) => (NotFull, false),
            },
        );
        if !valid {
            return Err(TensorError::SliceInvalid);
        }

        let len = Shape::from(end - start).len();
        let start = shape.linear_index(start);
        Ok((start, start + len))
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorDimension {
    #[default]
    Full,
    Auto,
    Dimension(usize),
}

impl TensorDimension {
    pub fn deduce(shape: Shape, x: Self, y: Self, z: Self, w: Self) -> Result<Shape, TensorError> {
        use TensorDimension::{Auto, Dimension, Full};
        let len = shape.len();

        let deduced = [x, y, z, w]
            .into_iter()
            .enumerate()
            .map(|(index, dim)| match dim {
                Full => Some(shape[index]),
                Auto => None,
                Dimension(dim) => Some(dim),
            });
        let remain: usize = deduced.clone().flatten().product();

        if remain == 0 || deduced.clone().filter(|x| x.is_none()).count() > 1 {
            return Err(TensorError::Deduce);
        };

        let deduced = deduced.map(|x| x.unwrap_or(len / remain)).collect_vec();
        let deduced = Shape::from_slice(&deduced);

        if deduced.len() != len {
            Err(TensorError::Size(deduced.len(), len))
        } else {
            Ok(deduced)
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use itertools::Itertools;
    use wgpu::{Instance, PowerPreference};

    use super::{Shape, TensorSlice};
    use crate::{
        context::{Context, ContextBuilder, InstanceExt},
        tensor::{shape::ShapedIndex, TensorCpu, TensorInit},
    };

    async fn create_context() -> Result<Context> {
        let instance = Instance::default();
        let adapter = instance.adapter(PowerPreference::HighPerformance).await?;
        let context = ContextBuilder::new(adapter)
            // .features(Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES)
            .build()
            .await?;
        Ok(context)
    }

    #[test]
    fn test_shaped_index() {
        let shape = Shape::new(1024, 768, 12, 1);
        let index = ShapedIndex::new(35, 42, 9, 0);
        let index = shape.linear_index(index);
        assert_eq!(index, 35 + 42 * 1024 + 9 * 1024 * 768);
    }

    #[tokio::test]
    async fn test_slice() -> Result<()> {
        let context = create_context().await?;

        let x: TensorCpu<f32> = context.tensor_init([1024, 768, 3, 1]);
        assert_eq!(
            (12..42, 7..8, 1, 0).linear_bounds(x.shape)?,
            (793612, 793642)
        );
        assert_eq!(
            (.., 42..56, 2..=2, ..).shaped_bounds(x.shape)?,
            (
                ShapedIndex::new(0, 42, 2, 0),
                ShapedIndex::new(1024, 56, 3, 1)
            )
        );
        assert!((.., 42..56, 2..3, ..).linear_bounds(x.shape).is_ok());
        assert!((0..1, 0..1, 0..1, ..).linear_bounds(x.shape).is_ok());
        assert!((.., 42..56, 0..2, ..).linear_bounds(x.shape).is_err());
        assert!((0, 0..2, 1..2, ..).linear_bounds(x.shape).is_err());

        let x: TensorCpu<f32> = context.tensor_init([1, 1024, 6, 1]);
        assert_eq!(
            (.., 0..256, 3..=3, ..).linear_bounds(x.shape)?,
            (3072, 3328)
        );

        let x: TensorCpu<f32> = context.tensor_init([1024, 768, 1, 1]);
        assert!((.., 0..256, .., ..).linear_bounds(x.shape).is_ok());

        let x: TensorCpu<f32> = context.tensor_init([1, 768, 1, 1]);
        assert!((.., 256..512, .., ..).linear_bounds(x.shape).is_ok());

        let shape = Shape::new(4, 2, 3, 1);
        let x = (0..shape.len()).map(|x| x as f32).collect_vec();
        let x = TensorCpu::from_data(shape, x)?;

        let y: Vec<_> = x.slice(.., 1..2, 1..2, ..)?.into();
        assert_eq!(y, vec![12.0, 13.0, 14.0, 15.0]);

        let y: Vec<_> = x.slice(.., .., 1..2, ..)?.into();
        assert_eq!(y, vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

        let y: Vec<_> = x.slice(2.., 1.., ..0, ..)?.into();
        assert_eq!(y, Vec::<f32>::new());

        Ok(())
    }

    #[test]
    fn test_cartesian_product() -> Result<()> {
        let shape = Shape::new(4, 3, 2, 1);
        let indices = shape.cartesian_product().collect_vec();
        assert_eq!(
            indices,
            vec![
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [2, 0, 0, 0],
                [3, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 1, 0, 0],
                [2, 1, 0, 0],
                [3, 1, 0, 0],
                [0, 2, 0, 0],
                [1, 2, 0, 0],
                [2, 2, 0, 0],
                [3, 2, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 1, 0],
                [2, 0, 1, 0],
                [3, 0, 1, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 0],
                [2, 1, 1, 0],
                [3, 1, 1, 0],
                [0, 2, 1, 0],
                [1, 2, 1, 0],
                [2, 2, 1, 0],
                [3, 2, 1, 0],
            ]
            .into_iter()
            .map(ShapedIndex::from)
            .collect_vec()
        );

        Ok(())
    }
}
