use std::{
    cmp::Ordering,
    hash::Hash,
    ops::{Bound, RangeBounds},
};

use web_rwkv_derive::Deref;

use super::TensorError;

pub trait IntoBytes {
    fn into_bytes(self) -> Vec<u8>;
}

/// The shape of a [`Tensor`].
/// Note that the fastest-moving axis occupies the lowest shape index, which is opposite to that in `torch`.
#[derive(Debug, Default, Clone, Copy, Deref, PartialEq, Eq, Hash)]
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

pub trait TensorSlice: Clone + PartialEq + Eq + Hash {
    fn convert_bounds(
        slice: impl RangeBounds<usize>,
        dim: usize,
    ) -> Result<(usize, usize), TensorError>;

    fn shape_bounds(self, shape: Shape) -> Result<(Shape, Shape), TensorError>;
    fn contiguous_bounds(self, shape: Shape) -> Result<(usize, usize), TensorError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SliceState {
    Zero,
    One,
    NotFull,
    Full,
}

impl<X, Y, Z> TensorSlice for (X, Y, Z)
where
    X: std::ops::RangeBounds<usize> + Clone + PartialEq + Eq + Hash,
    Y: std::ops::RangeBounds<usize> + Clone + PartialEq + Eq + Hash,
    Z: std::ops::RangeBounds<usize> + Clone + PartialEq + Eq + Hash,
{
    fn convert_bounds(
        slice: impl RangeBounds<usize>,
        dim: usize,
    ) -> Result<(usize, usize), TensorError> {
        let start = match slice.start_bound() {
            Bound::Included(&bound) => bound,
            Bound::Excluded(&bound) => bound + 1,
            Bound::Unbounded => 0,
        };
        let end = match slice.end_bound() {
            Bound::Included(&bound) => bound + 1,
            Bound::Excluded(&bound) => bound,
            Bound::Unbounded => dim,
        };
        if start > end || start >= dim || end > dim {
            Err(TensorError::SliceOutOfRange { dim, start, end })
        } else {
            Ok((start, end))
        }
    }

    fn shape_bounds(self, shape: Shape) -> Result<(Shape, Shape), TensorError> {
        let mut start = Shape::default();
        let mut end = Shape::default();
        (start[0], end[0]) = Self::convert_bounds(self.0, shape[0])?;
        (start[1], end[1]) = Self::convert_bounds(self.1, shape[1])?;
        (start[2], end[2]) = Self::convert_bounds(self.2, shape[2])?;
        Ok((start, end))
    }

    fn contiguous_bounds(self, shape: Shape) -> Result<(usize, usize), TensorError> {
        let mut state = SliceState::Full;

        let mut check_slice_dim = |start, end, dim| {
            let current_state = match (start, end) {
                (start, end) if start == end => SliceState::Zero,
                (start, end) if end == start + 1 => SliceState::One,
                (0, end) if end == dim => SliceState::Full,
                _ => SliceState::NotFull,
            };

            let previous_state = state;
            state = current_state;

            match previous_state {
                SliceState::NotFull => current_state < previous_state,
                _ => current_state <= previous_state,
            }
            .then_some(())
            .ok_or(TensorError::SliceNotContiguous)
        };

        let (start, end) = self.shape_bounds(shape)?;
        check_slice_dim(start[0], end[0], shape[0])
            .and(check_slice_dim(start[1], end[1], shape[1]))
            .and(check_slice_dim(start[2], end[2], shape[2]))?;

        let len = (end - start).len();
        let start = shape.shape_index(start);
        Ok((start, start + len))
    }
}

#[cfg(test)]
mod tests {
    use wgpu::PowerPreference;

    use super::{Shape, TensorSlice};
    use crate::{
        context::{Context, ContextBuilder, Instance},
        tensor::{TensorCpu, TensorExt},
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
        let context = create_context()?;

        let x: TensorCpu<f32> = context.init_tensor(Shape::new(1024, 768, 3));

        assert_eq!(
            (12..42, 7..8, 1..=1).contiguous_bounds(x.shape)?,
            (793612, 793642)
        );

        assert!((.., 42..56, 0..2).contiguous_bounds(x.shape).is_err());
        assert!((0..1, 0..2, 1..2).contiguous_bounds(x.shape).is_err());

        // x.check_slice(12..42, 7..8, 1..=1)?;
        // x.check_slice(.., .., ..)?;
        // x.check_slice(.., 42..56, 2..3)?;
        // x.check_slice(0..1, 0..1, 0..1)?;

        // assert!(x.check_slice(.., 42..56, 0..2).is_err());
        // assert!(x.check_slice(0..1, 0..2, 1..2).is_err());

        assert_eq!(
            (.., 42..56, 2..=2).shape_bounds(x.shape)?,
            (Shape::new(0, 42, 2), Shape::new(1024, 56, 3))
        );

        let shape = Shape::new(4, 2, 3);
        let x: Vec<_> = (0..shape.len()).map(|x| x as f32).collect();
        let x = TensorCpu::from_data(&context, shape, x)?;

        let y: Vec<_> = x.as_slice((.., 1..2, 1..2))?.into();
        assert_eq!(y, vec![12.0, 13.0, 14.0, 15.0]);

        let y: Vec<_> = x.as_slice((.., .., 1..2))?.into();
        assert_eq!(y, vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

        let y: Vec<_> = x.clone().into_slice((2.., 1.., ..0))?.into();
        assert_eq!(y, Vec::<f32>::new());

        Ok(())
    }
}
