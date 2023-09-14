use std::{cmp::Ordering, hash::Hash};

use web_rwkv_derive::{Deref, DerefMut};

use super::TensorError;

pub trait IntoBytes {
    fn into_bytes(self) -> Vec<u8>;
}

/// The shape of a [`Tensor`].
/// Note that the fastest-moving axis occupies the lowest shape index, which is opposite to that in `torch`.
#[derive(Debug, Default, Clone, Copy, Deref, DerefMut, PartialEq, Eq, Hash)]
pub struct Shape([usize; 3]);

impl Shape {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self([x, y, z])
    }

    pub fn from_slice(slice: &[usize]) -> Self {
        let mut shape = Self::new(1, 1, 1);
        for (index, &dim) in slice.iter().take(3).enumerate() {
            shape[index] = dim;
        }
        shape
    }

    pub fn len(&self) -> usize {
        self.0.into_iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.0.into_iter().any(|x| x == 0)
    }

    /// Convert a shaped index into a linear index.
    pub fn shape_index(&self, indices: Shape) -> usize {
        Iterator::zip(self.0.into_iter().rev(), indices.0.into_iter().rev())
            .fold(0, |acc, (shape, index)| acc * shape + index)
    }
}

impl IntoBytes for Shape {
    fn into_bytes(self) -> Vec<u8> {
        let data = vec![self.0[0] as u32, self.0[1] as u32, self.0[2] as u32, 1];
        bytemuck::pod_collect_to_vec(&data)
    }
}

impl std::cmp::PartialOrd for Shape {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (
            self[0].cmp(&other[0]),
            self[1].cmp(&other[1]),
            self[2].cmp(&other[2]),
        ) {
            (x, y, z) if x == y && y == z => Some(x),
            (x, y, Ordering::Equal) if x == y => Some(x),
            (x, Ordering::Equal, z) if x == z => Some(x),
            (Ordering::Equal, y, z) if y == z => Some(y),
            (x, Ordering::Equal, Ordering::Equal) => Some(x),
            (Ordering::Equal, y, Ordering::Equal) => Some(y),
            (Ordering::Equal, Ordering::Equal, z) => Some(z),
            _ => None,
        }
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self[0], self[1], self[2])
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

impl std::ops::Add<Shape> for Shape {
    type Output = Self;

    fn add(self, rhs: Shape) -> Self::Output {
        Self::new(self[0] + rhs[0], self[1] + rhs[1], self[2] + rhs[2])
    }
}

impl std::ops::Sub<Shape> for Shape {
    type Output = Self;

    fn sub(self, rhs: Shape) -> Self::Output {
        Self::new(self[0] - rhs[0], self[1] - rhs[1], self[2] - rhs[2])
    }
}

impl std::ops::AddAssign<Shape> for Shape {
    fn add_assign(&mut self, rhs: Shape) {
        *self = *self + rhs;
    }
}

impl std::ops::SubAssign<Shape> for Shape {
    fn sub_assign(&mut self, rhs: Shape) {
        *self = *self - rhs;
    }
}

pub trait TensorSlice {
    fn shape_bounds(&self, shape: Shape) -> Result<(Shape, Shape), TensorError>;
    fn contiguous_bounds(&self, shape: Shape) -> Result<(usize, usize), TensorError>;
}

pub trait TensorAxis: Clone + PartialEq + Eq + Hash {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError>;
}

#[inline]
fn check_bounds(dim: usize, start: usize, end: usize) -> Result<(usize, usize), TensorError> {
    if start > end || start >= dim || end > dim {
        Err(TensorError::SliceOutOfRange { dim, start, end })
    } else {
        Ok((start, end))
    }
}

impl TensorAxis for usize {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        let start = *self;
        let end = start + 1;
        check_bounds(dim, start, end)
    }
}

impl TensorAxis for std::ops::RangeFull {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        Ok((0, dim))
    }
}

impl TensorAxis for std::ops::Range<usize> {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        check_bounds(dim, self.start, self.end)
    }
}

impl TensorAxis for std::ops::RangeInclusive<usize> {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        let start = *self.start();
        let end = self.end() + 1;
        check_bounds(dim, start, end)
    }
}

impl TensorAxis for std::ops::RangeFrom<usize> {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        check_bounds(dim, self.start, dim)
    }
}

impl TensorAxis for std::ops::RangeTo<usize> {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        check_bounds(dim, 0, self.end)
    }
}

impl TensorAxis for std::ops::RangeToInclusive<usize> {
    fn bounds(&self, dim: usize) -> Result<(usize, usize), TensorError> {
        check_bounds(dim, 0, self.end + 1)
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
    Multi,
}

enum SliceFillState {
    NotFull,
    Full,
}

impl<X, Y, Z> TensorSlice for (X, Y, Z)
where
    X: TensorAxis,
    Y: TensorAxis,
    Z: TensorAxis,
{
    fn shape_bounds(&self, shape: Shape) -> Result<(Shape, Shape), TensorError> {
        let mut start = Shape::default();
        let mut end = Shape::default();
        (start[0], end[0]) = self.0.bounds(shape[0])?;
        (start[1], end[1]) = self.1.bounds(shape[1])?;
        (start[2], end[2]) = self.2.bounds(shape[2])?;
        Ok((start, end))
    }

    fn contiguous_bounds(&self, shape: Shape) -> Result<(usize, usize), TensorError> {
        let quant_state = |start, end| match end - start {
            0 => SliceQuantState::Zero,
            1 => SliceQuantState::One,
            _ => SliceQuantState::Multi,
        };

        let fill_state = |start, end, dim| match (start, end) {
            (0, end) if end == dim => SliceFillState::Full,
            (start, end) if start == end => SliceFillState::Full,
            _ => SliceFillState::NotFull,
        };

        let (start, end) = self.shape_bounds(shape)?;
        let (_, valid) = start.iter().zip(end.iter()).zip(shape.iter()).fold(
            (SliceFillState::Full, true),
            |(state, valid), ((&start, &end), &dim)| match (state, valid) {
                (SliceFillState::Full, valid) => (fill_state(start, end, dim), valid),
                (SliceFillState::NotFull, true) => (
                    SliceFillState::NotFull,
                    quant_state(start, end) < SliceQuantState::Multi,
                ),
                (SliceFillState::NotFull, false) => (SliceFillState::NotFull, false),
            },
        );
        if !valid {
            return Err(TensorError::Contiguous);
        }

        let len = (end - start).len();
        let start = shape.shape_index(start);
        Ok((start, start + len))
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDimension {
    #[default]
    Full,
    Auto,
    Dimension(usize),
}

impl TensorDimension {
    pub fn deduce(shape: Shape, x: Self, y: Self, z: Self) -> Result<Shape, TensorError> {
        use TensorDimension::{Auto, Dimension, Full};
        let len = shape.len();
        let Shape([a, b, c]) = shape;

        let deduced = match (x, y, z) {
            (Auto, Auto, _) | (Auto, _, Auto) | (_, Auto, Auto) => Err(TensorError::Dimension),
            (Full, Full, Full) | (Full, Full, Auto) | (Full, Auto, Full) | (Auto, Full, Full) => {
                Ok(shape)
            }
            (Full, Full, Dimension(z)) => Ok(Shape([a, b, z])),
            (Full, Auto, Dimension(z)) => Ok(Shape([a, len / a / z, z])),
            (Full, Dimension(y), Full) => Ok(Shape([a, y, c])),
            (Full, Dimension(y), Auto) => Ok(Shape([a, y, len / a / y])),
            (Full, Dimension(y), Dimension(z)) => Ok(Shape([a, y, z])),
            (Auto, Full, Dimension(z)) => Ok(Shape([len / b / z, b, z])),
            (Auto, Dimension(y), Full) => Ok(Shape([len / y / c, y, c])),
            (Auto, Dimension(y), Dimension(z)) => Ok(Shape([len / y / z, y, z])),
            (Dimension(x), Full, Full) => Ok(Shape([x, b, c])),
            (Dimension(x), Full, Auto) => Ok(Shape([x, b, len / x / b])),
            (Dimension(x), Full, Dimension(z)) => Ok(Shape([x, b, z])),
            (Dimension(x), Auto, Full) => Ok(Shape([x, len / x / c, c])),
            (Dimension(x), Auto, Dimension(z)) => Ok(Shape([x, len / x / z, z])),
            (Dimension(x), Dimension(y), Full) => Ok(Shape([x, y, c])),
            (Dimension(x), Dimension(y), Auto) => Ok(Shape([x, y, len / x / y])),
            (Dimension(x), Dimension(y), Dimension(z)) => Ok(Shape([x, y, z])),
        }?;

        if deduced.len() != len {
            Err(TensorError::Size(deduced.len(), len))
        } else {
            Ok(deduced)
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use wgpu::PowerPreference;

    use super::{Shape, TensorSlice};
    use crate::{
        context::{Context, ContextBuilder, Instance},
        tensor::{TensorCpu, TensorInit},
    };

    fn create_context() -> Result<Context, anyhow::Error> {
        let adapter = pollster::block_on(async {
            let instance = Instance::new();
            instance.adapter(PowerPreference::HighPerformance).await
        })?;
        let context = pollster::block_on(async {
            ContextBuilder::new(adapter)
                .with_default_pipelines()
                .build()
                .await
        })?;
        Ok(context)
    }

    #[test]
    fn test_shape_index() {
        let shape = Shape::new(1024, 768, 12);
        let indices = Shape::new(35, 42, 9);
        let index = shape.shape_index(indices);
        assert_eq!(index, 35 + 42 * 1024 + 9 * 1024 * 768);
    }

    #[test]
    fn test_slice() -> Result<(), anyhow::Error> {
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };

        let x: TensorCpu<f32> = context.tensor_init(Shape::new(1024, 768, 3));
        assert_eq!(
            (12..42, 7..8, 1).contiguous_bounds(x.shape)?,
            (793612, 793642)
        );
        assert_eq!(
            (.., 42..56, 2..=2).shape_bounds(x.shape)?,
            (Shape::new(0, 42, 2), Shape::new(1024, 56, 3))
        );
        assert!((.., 42..56, 2..3).contiguous_bounds(x.shape).is_ok());
        assert!((0..1, 0..1, 0..1).contiguous_bounds(x.shape).is_ok());
        assert!((.., 42..56, 0..2).contiguous_bounds(x.shape).is_err());
        assert!((0, 0..2, 1..2).contiguous_bounds(x.shape).is_err());

        let x: TensorCpu<f32> = context.tensor_init(Shape::new(1, 1024, 6));
        assert_eq!(
            (.., 0..256, 3..=3).contiguous_bounds(x.shape)?,
            (3072, 3328)
        );

        let x: TensorCpu<f32> = context.tensor_init(Shape::new(1024, 768, 1));
        assert!((.., 0..256, ..).contiguous_bounds(x.shape).is_ok());

        let x: TensorCpu<f32> = context.tensor_init(Shape::new(1, 768, 1));
        assert!((.., 256..512, ..).contiguous_bounds(x.shape).is_ok());

        let shape = Shape::new(4, 2, 3);
        let x = (0..shape.len()).map(|x| x as f32).collect_vec();
        let x = TensorCpu::from_data(&context, shape, x)?;

        let y: Vec<_> = x.slice(.., 1..2, 1..2)?.into();
        assert_eq!(y, vec![12.0, 13.0, 14.0, 15.0]);

        let y: Vec<_> = x.slice(.., .., 1..2)?.into();
        assert_eq!(y, vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

        let y: Vec<_> = x.into_slice(2.., 1.., ..0)?.into();
        assert_eq!(y, Vec::<f32>::new());

        Ok(())
    }
}
