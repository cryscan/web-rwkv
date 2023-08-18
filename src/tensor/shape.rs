use core::ops::RangeBounds;
use std::{
    cell::RefCell,
    collections::HashMap,
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

    pub fn buffer<F>(&self, shape: Shape, op: F) -> Arc<Buffer>
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

#[derive(Debug, Default)]
pub struct Slice<X, Y, Z>(X, Y, Z);

impl<X, Y, Z> Slice<X, Y, Z>
where
    X: RangeBounds<usize>,
    Y: RangeBounds<usize>,
    Z: RangeBounds<usize>,
{
    pub fn new(x: X, y: Y, z: Z) -> Self {
        Self(x, y, z)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SliceState {
    Zero,
    One,
    NotFull,
    Full,
}

impl<D: Device, T: Scalar, K: Kind> Tensor<'_, D, T, K> {
    pub fn check_shape(&self, shape: Shape) -> Result<(), TensorError> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(TensorError::Shape(self.shape, shape))
        }
    }

    pub fn is_slice_contiguous<X, Y, Z>(&self, slice: &Slice<X, Y, Z>) -> bool
    where
        X: RangeBounds<usize>,
        Y: RangeBounds<usize>,
        Z: RangeBounds<usize>,
    {
        fn check_slice_dim<B: RangeBounds<usize>>(
            slice: &B,
            dim: usize,
            state: &mut SliceState,
        ) -> bool {
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
            let current_state = if start >= end {
                SliceState::Zero
            } else if start >= dim || end <= 0 || end > dim {
                panic!("Bad slice {}..{} of dim {}", start, end, dim);
            } else if start == 0 && end == dim {
                SliceState::Full
            } else if end == start + 1 {
                SliceState::One
            } else {
                SliceState::NotFull
            };

            let contiguous = if *state == SliceState::NotFull {
                // cannot have 2 dims that are both not full.
                current_state < *state
            } else {
                current_state <= *state
            };
            *state = current_state;
            contiguous
        }

        let mut state = SliceState::Full;
        let x = check_slice_dim(&slice.0, self.shape[0], &mut state);
        let y = check_slice_dim(&slice.1, self.shape[1], &mut state);
        let z = check_slice_dim(&slice.2, self.shape[2], &mut state);
        x && y && z
    }
}

#[cfg(test)]
mod tests {
    use wgpu::PowerPreference;

    use super::{Shape, Slice};
    use crate::{
        context::{Context, ContextBuilder, Instance},
        tensor::{ReadWrite, TensorCpu},
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
    fn test_shape_contiguous() -> Result<(), anyhow::Error> {
        let context = create_context()?;

        let x: TensorCpu<f32, ReadWrite> = context.tensor_init(None, Shape::new(1024, 768, 3));

        assert!(x.is_slice_contiguous(&Slice::new(12..42, 7..8, 1..=1)));
        assert!(x.is_slice_contiguous(&Slice::new(.., .., ..)));
        assert!(x.is_slice_contiguous(&Slice(.., 42..56, 2..3)));
        assert!(x.is_slice_contiguous(&Slice(0..1, 0..1, 0..1)));

        assert!(!x.is_slice_contiguous(&Slice(.., 42..56, 0..2)));
        assert!(!x.is_slice_contiguous(&Slice(0..1, 0..2, 1..2)));

        Ok(())
    }
}
