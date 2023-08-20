use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::HashMap,
    hash::Hash,
    ops::Bound,
    sync::{Arc, RwLock},
};

use web_rwkv_derive::Deref;
use wgpu::Buffer;

use super::{Device, Kind, Scalar, Tensor, TensorError};

/// The shape of a [`Tensor`].
/// Note that the fastest-moving axis occupies the lowest shape index, which is opposite to that in `torch`.
#[derive(Debug, Default, Clone, Copy, Deref, PartialEq, Eq, Hash)]
pub struct Shape([usize; 3]);

impl Shape {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self([x, y, z])
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

    pub fn to_u32_slice(self) -> [u32; 4] {
        [self.0[0] as u32, self.0[1] as u32, self.0[2] as u32, 1]
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

#[derive(Debug, Default, Deref)]
pub struct ShapeCache(RefCell<RwLock<HashMap<Shape, Arc<Buffer>>>>);

impl ShapeCache {
    pub fn clear(&self) {
        let map = self.borrow_mut();
        let mut map = map.write().unwrap();
        map.clear();
    }

    pub fn query(&self, shape: Shape) -> Option<Arc<Buffer>> {
        let map = self.borrow();
        let map = map.read().unwrap();
        map.get(&shape).cloned()
    }

    pub fn request<F>(&self, shape: Shape, op: F) -> Arc<Buffer>
    where
        F: FnOnce() -> Buffer,
    {
        match self.query(shape) {
            Some(buffer) => buffer,
            None => {
                let buffer = Arc::new(op());
                let map = self.borrow_mut();
                let mut map = map.write().unwrap();
                map.insert(shape, buffer.clone());
                buffer
            }
        }
    }
}

pub trait TensorSlice: std::ops::RangeBounds<usize> + Clone + PartialEq + Eq + Hash {}

impl<T> TensorSlice for T where T: std::ops::RangeBounds<usize> + Clone + PartialEq + Eq + Hash {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SliceState {
    Zero,
    One,
    NotFull,
    Full,
}

fn slice_to_dim(slice: impl TensorSlice, dim: usize) -> (usize, usize) {
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
    (start, end)
}

fn check_slice_dim(
    slice: impl TensorSlice,
    dim: usize,
    state: &mut SliceState,
) -> Result<(), TensorError> {
    let (start, end) = slice_to_dim(slice, dim);
    let current_state = match (start, end) {
        (start, end) if start >= dim => Err(TensorError::SliceOutOfRange { dim, start, end }),
        (start, end) if end > dim => Err(TensorError::SliceOutOfRange { dim, start, end }),
        (start, end) if start >= end => Ok(SliceState::Zero),
        (start, end) if end == start + 1 => Ok(SliceState::One),
        (0, end) if end == dim => Ok(SliceState::Full),
        _ => Ok(SliceState::NotFull),
    }?;

    let previous_state = *state;
    *state = current_state;

    match previous_state {
        SliceState::NotFull => current_state < previous_state,
        _ => current_state <= previous_state,
    }
    .then_some(())
    .ok_or(TensorError::SliceNotContiguous)
}

impl<D: Device, T: Scalar, K: Kind> Tensor<'_, D, T, K> {
    pub fn check_shape(&self, shape: Shape) -> Result<(), TensorError> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(TensorError::Shape(self.shape, shape))
        }
    }

    pub fn check_shape_with(
        &self,
        shape: Shape,
        op: impl FnOnce(Shape, Shape) -> bool,
    ) -> Result<(), TensorError> {
        if op(self.shape, shape) {
            Ok(())
        } else {
            Err(TensorError::Shape(self.shape, shape))
        }
    }

    /// Check if a given slice both is not out of range and views a contiguous chunk of memory.
    pub fn check_slice(
        &self,
        x: impl TensorSlice,
        y: impl TensorSlice,
        z: impl TensorSlice,
    ) -> Result<(), TensorError> {
        let mut state = SliceState::Full;
        let x = check_slice_dim(x, self.shape[0], &mut state);
        let y = check_slice_dim(y, self.shape[1], &mut state);
        let z = check_slice_dim(z, self.shape[2], &mut state);
        x.and(y).and(z)
    }

    pub fn shape_bounds(
        &self,
        x: impl TensorSlice,
        y: impl TensorSlice,
        z: impl TensorSlice,
    ) -> (Shape, Shape) {
        let mut start = Shape::default();
        let mut end = Shape::default();
        (start[0], end[0]) = slice_to_dim(x, self.shape[0]);
        (start[1], end[1]) = slice_to_dim(y, self.shape[1]);
        (start[2], end[2]) = slice_to_dim(z, self.shape[2]);
        (start, end)
    }

    pub fn slice_shape(
        &self,
        x: impl TensorSlice,
        y: impl TensorSlice,
        z: impl TensorSlice,
    ) -> Shape {
        let (start, end) = self.shape_bounds(x, y, z);
        end - start
    }
}

#[cfg(test)]
mod tests {
    use wgpu::PowerPreference;

    use super::Shape;
    use crate::{
        context::{Context, ContextBuilder, Instance},
        tensor::{ReadWrite, TensorCpu, TensorExt},
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

        let x: TensorCpu<f32, ReadWrite> = context.init_tensor(Shape::new(1024, 768, 3));

        x.check_slice(12..42, 7..8, 1..=1)?;
        x.check_slice(.., .., ..)?;
        x.check_slice(.., 42..56, 2..3)?;
        x.check_slice(0..1, 0..1, 0..1)?;

        assert!(x.check_slice(.., 42..56, 0..2).is_err());
        assert!(x.check_slice(0..1, 0..2, 1..2).is_err());

        assert_eq!(
            x.shape_bounds(.., 42..56, 2..=2),
            (Shape::new(0, 42, 2), Shape::new(1024, 56, 3))
        );

        let shape = Shape::new(4, 2, 3);
        let x: Vec<_> = (0..shape.len()).map(|x| x as f32).collect();
        let x: TensorCpu<_, ReadWrite> = TensorCpu::from_data(&context, shape, x)?;

        let y: Vec<_> = x.clone().into_slice(.., 1..2, 1..2)?.into();
        assert_eq!(y, vec![12.0, 13.0, 14.0, 15.0]);

        let y: Vec<_> = x.clone().into_slice(.., .., 1..2)?.into();
        assert_eq!(y, vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

        let y: Vec<_> = x.into_slice(2.., 1.., ..0)?.into();
        assert_eq!(y, Vec::<f32>::new());

        Ok(())
    }
}