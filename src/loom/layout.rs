use std::fmt::Debug;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("len error: shape {0} vs. stride {1}")]
    ShapeStrideLen(Shape, Stride),
    #[error("len error: layout {0} vs. coord {1}")]
    LayoutCoordLen(Layout, Coord),
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<usize>);

impl<T> From<T> for Shape
where
    T: IntoIterator<Item = usize>,
{
    fn from(value: T) -> Self {
        Self(value.into_iter().collect())
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Shape {
    /// Dimension of the shape.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the shape is of size 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Total number of elements the shape contains.
    #[inline]
    pub fn size(&self) -> usize {
        match self.len() {
            0 => 0,
            _ => self.0.iter().product(),
        }
    }
}

/// Defines the step to add to when increase 1 along coordinates.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Stride(pub Vec<usize>);

impl<T> From<T> for Stride
where
    T: IntoIterator<Item = usize>,
{
    fn from(value: T) -> Self {
        Self(value.into_iter().collect())
    }
}

impl std::ops::Index<usize> for Stride {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::fmt::Display for Stride {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Stride {
    /// Dimension of the stride.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the stride contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A multi-dimensional coordinate.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Coord(pub Vec<usize>);

impl<T> From<T> for Coord
where
    T: IntoIterator<Item = usize>,
{
    fn from(value: T) -> Self {
        Self(value.into_iter().collect())
    }
}

impl std::ops::Index<usize> for Coord {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::fmt::Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Coord {
    /// Dimension of the coordinate.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the coordinate contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A [`Layout`] is a mapping of multi-dimensional indices.
///
/// For more information, check:
/// - [CuTe documents](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute);
/// - [A note on the algebra of CuTe Layouts](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf).
#[derive(Debug, Default, Clone)]
pub struct Layout(pub Vec<(usize, usize)>);

impl<S, D> TryFrom<(S, D)> for Layout
where
    S: Into<Shape>,
    D: Into<Stride>,
{
    type Error = LayoutError;

    fn try_from((shape, stride): (S, D)) -> Result<Self, Self::Error> {
        let shape: Shape = shape.into();
        let stride: Stride = stride.into();

        if shape.len() != stride.len() {
            Err(LayoutError::ShapeStrideLen(shape.clone(), stride.clone()))?
        }

        let value = shape.0.into_iter().zip(stride.0).collect();
        Ok(Self(value))
    }
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.shape();
        let stride = self.stride();
        write!(f, "<{shape}, {stride}>")
    }
}

impl Layout {
    /// Retrieves the shape of the layout.
    #[inline]
    pub fn shape(&self) -> Shape {
        Shape(self.0.iter().map(|&(x, _)| x).collect())
    }

    /// Retrieves the stride of the layout.
    #[inline]
    pub fn stride(&self) -> Stride {
        Stride(self.0.iter().map(|&(_, x)| x).collect())
    }

    /// Dimension of the layout.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the layout is of size 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shape().is_empty()
    }

    /// Maps a linear index to a multi-dimensional coordinate.
    #[inline]
    pub fn iota(&self, index: usize) -> Coord {
        Coord(
            self.0
                .iter()
                .fold((vec![], 1), |(mut v, p), &(m, _)| {
                    v.push((index / p) % m);
                    (v, p * m)
                })
                .0,
        )
    }

    /// Same as [`Layout::iota`], but not limited by the bound of the highest dimension.
    #[inline]
    pub fn iota_extend(&self, index: usize) -> Coord {
        Coord(
            self.0
                .iter()
                .enumerate()
                .fold((vec![], 1), |(mut v, p), (x, &(m, _))| {
                    match x {
                        x if x + 1 == self.len() => v.push(index / p),
                        _ => v.push((index / p) % m),
                    };
                    (v, p * m)
                })
                .0,
        )
    }

    /// Send an index to the layout's value.
    #[inline]
    pub fn value(&self, index: usize) -> usize {
        let coord = self.iota_extend(index);
        self.value_coord(&coord).expect("this couldn't happen")
    }

    /// Send a coordinate to the layout's value. The coordinate must have the same length as the layout.
    #[inline]
    pub fn value_coord(&self, coord: &Coord) -> Result<usize, LayoutError> {
        if self.len() != coord.len() {
            Err(LayoutError::LayoutCoordLen(self.clone(), coord.clone()))?
        }

        Ok(self
            .0
            .iter()
            .zip(coord.0.iter())
            .map(|(&(_, d), &x)| d * x)
            .sum())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::{Coord, Layout};

    #[test]
    fn test_layout_simple() {
        assert!(Layout::try_from(([2, 2], [4, 1])).is_ok());
        assert!(Layout::try_from(([2, 2, 4], [4, 1])).is_err());
    }

    #[test]
    fn test_isomorphism() -> Result<()> {
        let layout = Layout::try_from(([2, 3, 4], [3, 1, 6]))?;

        assert_eq!(layout.iota(0), Coord(vec![0, 0, 0]));
        assert_eq!(layout.iota(1), Coord(vec![1, 0, 0]));
        assert_eq!(layout.iota(2), Coord(vec![0, 1, 0]));
        assert_eq!(layout.iota(3), Coord(vec![1, 1, 0]));
        assert_eq!(layout.iota(4), Coord(vec![0, 2, 0]));
        assert_eq!(layout.iota(5), Coord(vec![1, 2, 0]));

        assert_eq!(layout.iota(6), Coord(vec![0, 0, 1]));
        assert_eq!(layout.iota(7), Coord(vec![1, 0, 1]));
        assert_eq!(layout.iota(8), Coord(vec![0, 1, 1]));
        assert_eq!(layout.iota(9), Coord(vec![1, 1, 1]));
        assert_eq!(layout.iota(10), Coord(vec![0, 2, 1]));
        assert_eq!(layout.iota(11), Coord(vec![1, 2, 1]));

        assert_eq!(layout.iota(12), Coord(vec![0, 0, 2]));
        assert_eq!(layout.iota(13), Coord(vec![1, 0, 2]));
        assert_eq!(layout.iota(14), Coord(vec![0, 1, 2]));
        assert_eq!(layout.iota(15), Coord(vec![1, 1, 2]));
        assert_eq!(layout.iota(16), Coord(vec![0, 2, 2]));
        assert_eq!(layout.iota(17), Coord(vec![1, 2, 2]));

        assert_eq!(layout.iota(18), Coord(vec![0, 0, 3]));
        assert_eq!(layout.iota(19), Coord(vec![1, 0, 3]));
        assert_eq!(layout.iota(20), Coord(vec![0, 1, 3]));
        assert_eq!(layout.iota(21), Coord(vec![1, 1, 3]));
        assert_eq!(layout.iota(22), Coord(vec![0, 2, 3]));
        assert_eq!(layout.iota(23), Coord(vec![1, 2, 3]));

        assert_eq!(layout.iota(24), Coord(vec![0, 0, 0]));
        assert_eq!(layout.iota(25), Coord(vec![1, 0, 0]));

        assert_eq!(layout.iota_extend(0), Coord(vec![0, 0, 0]));
        assert_eq!(layout.iota_extend(1), Coord(vec![1, 0, 0]));
        assert_eq!(layout.iota_extend(2), Coord(vec![0, 1, 0]));
        assert_eq!(layout.iota_extend(3), Coord(vec![1, 1, 0]));
        assert_eq!(layout.iota_extend(4), Coord(vec![0, 2, 0]));
        assert_eq!(layout.iota_extend(5), Coord(vec![1, 2, 0]));

        assert_eq!(layout.iota_extend(6), Coord(vec![0, 0, 1]));
        assert_eq!(layout.iota_extend(7), Coord(vec![1, 0, 1]));
        assert_eq!(layout.iota_extend(8), Coord(vec![0, 1, 1]));
        assert_eq!(layout.iota_extend(9), Coord(vec![1, 1, 1]));
        assert_eq!(layout.iota_extend(10), Coord(vec![0, 2, 1]));
        assert_eq!(layout.iota_extend(11), Coord(vec![1, 2, 1]));

        assert_eq!(layout.iota_extend(12), Coord(vec![0, 0, 2]));
        assert_eq!(layout.iota_extend(13), Coord(vec![1, 0, 2]));
        assert_eq!(layout.iota_extend(14), Coord(vec![0, 1, 2]));
        assert_eq!(layout.iota_extend(15), Coord(vec![1, 1, 2]));
        assert_eq!(layout.iota_extend(16), Coord(vec![0, 2, 2]));
        assert_eq!(layout.iota_extend(17), Coord(vec![1, 2, 2]));

        assert_eq!(layout.iota_extend(18), Coord(vec![0, 0, 3]));
        assert_eq!(layout.iota_extend(19), Coord(vec![1, 0, 3]));
        assert_eq!(layout.iota_extend(20), Coord(vec![0, 1, 3]));
        assert_eq!(layout.iota_extend(21), Coord(vec![1, 1, 3]));
        assert_eq!(layout.iota_extend(22), Coord(vec![0, 2, 3]));
        assert_eq!(layout.iota_extend(23), Coord(vec![1, 2, 3]));

        assert_eq!(layout.iota_extend(24), Coord(vec![0, 0, 4]));
        assert_eq!(layout.iota_extend(25), Coord(vec![1, 0, 4]));

        Ok(())
    }
}
