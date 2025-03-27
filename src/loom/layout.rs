use thiserror::Error;

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("shape and stride have different length:\nshape: {0}, stride: {1}")]
    ShapeStrideLen(usize, usize),
}

#[derive(Debug, Default, Clone)]
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

#[derive(Debug, Default, Clone)]
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
            Err(LayoutError::ShapeStrideLen(shape.len(), stride.len()))?
        }

        let value = shape.0.into_iter().zip(stride.0.into_iter()).collect();
        Ok(Self(value))
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
}

#[cfg(test)]
mod tests {
    use super::Layout;

    #[test]
    fn test_layout_from() {
        assert!(Layout::try_from(([2, 2], [4, 1])).is_ok());
        assert!(Layout::try_from(([2, 2, 4], [4, 1])).is_err());
    }
}
