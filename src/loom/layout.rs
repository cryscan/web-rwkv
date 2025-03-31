use itertools::Itertools;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("complement error: layout {0} vs. size {1}")]
    Complement(Layout, usize),
    #[error("compose error: layout composition {1} ◦ {0} is not valid")]
    Compose(Layout, Layout),
}

/// An [`IndexFunction`] is a mapping that maps an index to another.
pub trait IndexFunction<Index> {
    /// Sends an index to a mapped value.
    fn value(&self, index: Index) -> usize;
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
        write!(f, "{:?}", self.0)
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
        write!(f, "{:?}", self.0)
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
        write!(f, "{:?}", self.0)
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
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Layout(pub Vec<(usize, usize)>);

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.shape();
        let stride = self.stride();
        write!(f, "<{shape}, {stride}>")
    }
}

impl Layout {
    #[inline]
    pub fn from_shape(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let stride = Stride(
            shape
                .0
                .iter()
                .scan(1, |p, x| {
                    let q = *p;
                    *p *= x;
                    Some(q)
                })
                .collect(),
        );
        Self::from_shape_stride(shape, stride)
    }

    /// Creates a layout from shape and stride.
    /// **Panics** if lengths of the shape and the stride don't match.
    #[inline]
    pub fn from_shape_stride(shape: impl Into<Shape>, stride: impl Into<Stride>) -> Self {
        let shape: Shape = shape.into();
        let stride: Stride = stride.into();
        Self(shape.0.into_iter().zip_eq(stride.0).collect())
    }

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

    /// Number of elements in the shape of the layout.
    #[inline]
    pub fn size(&self) -> usize {
        self.shape().size()
    }

    /// The co-domain of the layout mapping.
    #[inline]
    pub fn co_size(&self) -> usize {
        match self.size() {
            0 => 0,
            x => self.value(x - 1) + 1,
        }
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

    /// Returns `true` if two layouts are totally equal as index mappings.
    /// Note that this check is exponentially slow so only use it in tests.
    #[inline]
    pub fn check_isomorphic(&self, other: &Layout) -> bool {
        if self.size() != other.size() {
            return false;
        }
        (0..self.size()).all(|index| self.value(index) == other.value(index))
    }

    /// Returns a simplified coalesce of the layout.
    pub fn coalesced(&self) -> Self {
        let mut layout = self.0.clone();

        loop {
            if layout.len() < 2 {
                break Self(layout);
            }

            let coalesced = [layout.clone(), vec![(1, 0)]]
                .concat()
                .into_iter()
                .tuples()
                .map(|(x, y)| match (x, y) {
                    ((1, _), y) => vec![y],
                    (x, (1, _)) => vec![x],
                    ((s0, d0), (s1, d1)) if d1 == s0 * d0 => vec![(s0 * s1, d0)],
                    (x, y) => vec![x, y],
                })
                .concat();

            if coalesced == layout {
                break Self(layout);
            }

            let _ = std::mem::replace(&mut layout, coalesced);
        }
    }

    /// Returns the layout whose strides are in sorted order.
    #[inline]
    pub fn sorted(&self) -> Self {
        Self(
            self.0
                .iter()
                .cloned()
                .sorted_by_key(|&(n, d)| (d, n))
                .collect(),
        )
    }

    /// Removes stride-0 or size-1 modes.
    #[inline]
    pub fn filtered(&self) -> Self {
        Self(
            self.0
                .iter()
                .cloned()
                .filter(|&(n, d)| d != 0 && n != 1)
                .collect(),
        )
    }

    /// Complements of the layout to a given size, if being admissible for complement.
    pub fn complement(&self, size: usize) -> Result<Self, LayoutError> {
        let layout = self.filtered().sorted();
        if layout.is_empty() {
            return Ok(Layout::from_shape([size]));
        }

        let shape = layout.shape();
        let stride = layout.stride();
        let product = shape
            .0
            .iter()
            .zip_eq(stride.0.iter())
            .map(|(n, d)| n * d)
            .collect_vec();

        let stride = [stride.0, vec![size]].concat(); // [d0, d1, ..., dα, M]
        let product = [vec![1], product].concat(); // [1, N0 d0, N1 d1, ..., Nα dα]

        let shape: Vec<_> = stride
            .iter()
            .zip_eq(product.iter())
            .map(|(d, p)| match d % p {
                0 => Ok(d / p),
                _ => Err(LayoutError::Complement(self.clone(), size)),
            })
            .try_collect()?;

        Ok(Self::from_shape_stride(shape, product).coalesced())
    }

    /// Layout composition. `a.compose(b)` corresponds to `B ◦ A` in layout algebra.
    pub fn compose(&self, _layout: Layout) -> Result<Self, LayoutError> {
        todo!()
    }
}

impl IndexFunction<&Coord> for Layout {
    #[inline]
    fn value(&self, index: &Coord) -> usize {
        self.0
            .iter()
            .zip_eq(index.0.iter())
            .map(|(&(_, d), &x)| d * x)
            .sum()
    }
}

impl IndexFunction<Coord> for Layout {
    #[inline]
    fn value(&self, index: Coord) -> usize {
        self.value(&index)
    }
}

impl IndexFunction<usize> for Layout {
    #[inline]
    fn value(&self, index: usize) -> usize {
        let coord = self.iota_extend(index);
        self.value(&coord)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::{Coord, IndexFunction, Layout, LayoutError};

    #[test]
    fn test_isomorphism() {
        let layout = Layout::from_shape_stride([2, 3, 4], [3, 1, 6]);

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
    }

    #[test]
    fn test_coalesce() {
        fn check(layout: Layout) {
            let coalesced = layout.coalesced();
            println!("{layout} → {coalesced}");
            assert!(layout.check_isomorphic(&coalesced))
        }

        check(Layout::from_shape_stride([1], [0]));
        check(Layout::from_shape_stride([1], [1]));

        check(Layout::from_shape([2, 4]));
        check(Layout::from_shape([2, 4, 6]));
        check(Layout::from_shape([2, 4, 6, 2]));

        check(Layout::from_shape_stride([2, 1, 6], [1, 6, 2]));
        check(Layout::from_shape_stride([2, 1, 6], [1, 7, 2]));

        check(Layout::from_shape_stride([2, 4, 6], [4, 1, 8]));
        check(Layout::from_shape_stride([2, 1, 3], [1, 1, 2]));
        check(Layout::from_shape_stride([2, 1, 3], [2, 4, 4]));
        check(Layout::from_shape_stride([2, 1, 3], [2, 0, 4]));
    }

    #[test]
    fn test_complement() -> Result<(), LayoutError> {
        fn check(layout: &Layout, co_size: usize) -> Result<(), LayoutError> {
            let complement = layout.complement(co_size)?;
            println!("{{{layout}, {co_size}}} → {complement}");

            // 1. disjoint
            assert!((1..layout.size()).all(|index| layout.value(index) != complement.value(index)));

            // 2. ordered
            assert!((0..complement.size())
                .tuple_windows::<(_, _)>()
                .all(|(x, y)| complement.value(x) < complement.value(y)));

            // 3. bounded
            if layout.size() > 0 {
                assert!(complement.size() >= co_size / layout.size());
                assert!(complement.co_size() <= co_size / layout.co_size() * layout.co_size());
            }

            Ok(())
        }

        {
            let layout = Layout::from_shape_stride([1], [0]);
            check(&layout, 1)?;
            check(&layout, 2)?;
            check(&layout, 5)?;
        }

        {
            let layout = Layout::from_shape_stride([1], [1]);
            check(&layout, 1)?;
            check(&layout, 2)?;
            check(&layout, 5)?;
        }

        {
            let layout = Layout::from_shape_stride([1], [2]);
            check(&layout, 1)?;
            check(&layout, 5)?;
            check(&layout, 2)?;
            check(&layout, 8)?;
        }

        {
            let layout = Layout::from_shape_stride([4], [0]);
            check(&layout, 1)?;
            check(&layout, 2)?;
            check(&layout, 8)?;
        }

        {
            let layout = Layout::from_shape_stride([4], [1]);
            assert!(check(&layout, 1).is_err());
            assert!(check(&layout, 2).is_err());
            check(&layout, 4)?;
            check(&layout, 8)?;
        }

        {
            let layout = Layout::from_shape_stride([4], [2]);
            check(&layout, 8)?;
            check(&layout, 16)?;
            assert!(check(&layout, 19).is_err());
        }

        {
            let layout = Layout::from_shape_stride([4], [4]);
            check(&layout, 16)?;
        }

        check(&Layout::from_shape_stride([4], [4]), 16)?;
        check(&Layout::from_shape_stride([2, 2], [4, 1]), 32)?;

        Ok(())
    }
}
