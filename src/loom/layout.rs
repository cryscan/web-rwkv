use std::sync::Arc;

use derive_more::{Deref, DerefMut, Display, From, Into};
use itertools::Itertools;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("complement error: layout {0} vs. size {1}")]
    Complement(Layout, usize),
    #[error("shape {0} is not left divisible by {1}")]
    ShapeDiv(Shape, usize),
    #[error("layout {0} is not disjoint")]
    Disjoint(Layout),
}

/// An [`IndexFunction`] is a mapping that maps an index to another.
pub trait IndexFn<Index> {
    type Output;

    /// Sends an index to a mapped value.
    fn value(&self, index: Index) -> Self::Output;
}

pub trait Compose<F> {
    type Output;

    /// Functional composition. `t.compose(f)` is `f ◦ t` in algebra.
    fn compose(&self, f: F) -> Self::Output;
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deref, DerefMut, From, Into, Display)]
#[display("{_0:?}")]
pub struct Shape(Arc<[usize]>);

impl From<usize> for Shape {
    fn from(value: usize) -> Self {
        Self([value].into())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    #[inline]
    fn from(value: [usize; N]) -> Self {
        Self(value.into_iter().collect())
    }
}

impl From<Vec<usize>> for Shape {
    #[inline]
    fn from(value: Vec<usize>) -> Self {
        Self(value.into())
    }
}

impl AsRef<Shape> for Shape {
    #[inline]
    fn as_ref(&self) -> &Shape {
        self
    }
}

impl Shape {
    #[inline]
    pub fn from_slice(slice: &[usize]) -> Self {
        Self(slice.into())
    }

    /// Returns `true` if the shape is of size 0.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.size() == 0
    }

    /// Total number of elements the shape contains.
    #[inline]
    pub fn size(&self) -> usize {
        match self.len() {
            0 => 0,
            _ => self.iter().product(),
        }
    }

    /// Performs shape division according to Definition 2.11 in [2](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf).
    ///
    /// ## Brief Explanation
    ///
    /// Suppose the shape is `(N0, N1, ..., N(α))`,
    /// and `d` can be divided "up to" `(N0, N1, ..., N(i - 1), c)`, where `c = d / (N0 × N1 × ... × N(i - 1))`.
    ///
    /// We can then divide the shape into
    /// 1. A quotient part `(N(i) / c, N(i + 1), ..., N(α))`, and
    /// 2. A remainder part `(N0, N1, ..., N(i - 1), c)`.
    pub fn shape_div(&self, d: usize) -> Result<(Self, Self), LayoutError> {
        if self.is_zero() {
            return Ok((self.clone(), Default::default()));
        }
        if d == 0 {
            assert_ne!(self.len(), 0);
            // in this case, since all nature numbers divide 0, we must have `i = α`, and c = 0
            let mut r = self.to_vec();
            r[self.len() - 1] = 0;
            return Ok((Default::default(), Shape::from(r)));
        }

        // [`1`, `N0`, `N0 × N1`, ..., `N0 × N1 × ... × N(α - 1)`]
        // [`N0`, `N1`, ..., `N(α)`]
        let product = self
            .iter()
            .scan(1, |p, &n| {
                let q = *p;
                *p *= n;
                Some((q, n))
            })
            .collect_vec();

        // find the division index
        let Some((i, &(p, n))) = product.iter().enumerate().find(|&(i, &(p, n))| {
            // 1. `p = N0 × N1 × ... × N(i-1)` divides d
            if d % p != 0 {
                return false;
            }
            // 2. if `i < α`, let `c = d / p`, we need `1 ≤ c < N(i)`, and `c` divides `N(i)`
            match (i, d / p) {
                (i, _) if i + 1 == self.len() => true,
                (_, c) => c > 0 && c < n && n % c == 0,
            }
        }) else {
            return Err(LayoutError::ShapeDiv(self.clone(), d));
        };

        assert!(i < self.len());
        let (r, q) = self.split_at(i);

        assert!(!q.is_empty());
        assert_eq!(q[0], n);

        let c = d / p;
        let r = [r, &[c]].concat();
        let q = [&[n / c], &q[1..]].concat();

        Ok((Shape::from(q), Shape::from(r)))
    }

    /// Performs weak shape division.
    /// Since `c` doesn't necessarily divides `N(i)`, we can only get the remainder here.
    pub fn shape_mod(&self, d: usize) -> Result<Self, LayoutError> {
        if self.is_zero() {
            return Ok(Default::default());
        }
        if d == 0 {
            assert_ne!(self.len(), 0);
            // in this case, since all nature numbers divide 0, we must have `i = α`, and c = 0
            let mut r = self.to_vec();
            r[self.len() - 1] = 0;
            return Ok(Shape::from(r));
        }

        // [`1`, `N0`, `N0 × N1`, ..., `N0 × N1 × ... × N(α - 1)`]
        // [`N0`, `N1`, ..., `N(α)`]
        let product = self
            .iter()
            .scan(1, |p, &n| {
                let q = *p;
                *p *= n;
                Some((q, n))
            })
            .collect_vec();

        // find the division index
        let Some((i, &(p, _))) = product.iter().enumerate().find(|&(i, &(p, n))| {
            // 1. `p = N0 × N1 × ... × N(i-1)` divides d
            if d % p != 0 {
                return false;
            }
            // 2. if `i < α`, let `c = d / p`, we need `1 ≤ c < N(i)`
            match (i, d / p) {
                (i, _) if i + 1 == self.len() => true,
                (_, c) => c > 0 && c < n,
            }
        }) else {
            return Err(LayoutError::ShapeDiv(self.clone(), d));
        };

        assert!(i < self.len());
        let r = &self[..i];

        let c = d / p;
        let r = [r, &[c]].concat();
        assert_ne!(r.len(), 0);

        Ok(Shape::from(r))
    }
}

/// Defines the step to add to when increase 1 along coordinates.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deref, DerefMut, From, Into, Display)]
#[display("{_0:?}")]
pub struct Stride(Arc<[usize]>);

impl From<usize> for Stride {
    fn from(value: usize) -> Self {
        Self([value].into())
    }
}

impl<const N: usize> From<[usize; N]> for Stride {
    #[inline]
    fn from(value: [usize; N]) -> Self {
        Self(value.into_iter().collect())
    }
}

impl From<Vec<usize>> for Stride {
    #[inline]
    fn from(value: Vec<usize>) -> Self {
        Self(value.into())
    }
}

impl AsRef<Stride> for Stride {
    #[inline]
    fn as_ref(&self) -> &Stride {
        self
    }
}

impl Stride {
    #[inline]
    pub fn from_slice(slice: &[usize]) -> Self {
        Self(slice.into())
    }
}

/// A multi-dimensional coordinate.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deref, DerefMut, From, Into, Display)]
#[display("{_0:?}")]
pub struct Coord(Arc<[usize]>);

impl From<usize> for Coord {
    fn from(value: usize) -> Self {
        Self([value].into())
    }
}

impl<const N: usize> From<[usize; N]> for Coord {
    #[inline]
    fn from(value: [usize; N]) -> Self {
        Self(value.into_iter().collect())
    }
}

impl From<Vec<usize>> for Coord {
    #[inline]
    fn from(value: Vec<usize>) -> Self {
        Self(value.into())
    }
}

impl AsRef<Coord> for Coord {
    #[inline]
    fn as_ref(&self) -> &Coord {
        self
    }
}

impl Coord {
    #[inline]
    pub fn from_slice(slice: &[usize]) -> Self {
        Self(slice.into())
    }
}

/// A [`Layout`] is a mapping of multi-dimensional indices.
///
/// For more information, check:
/// 1. [CuTe documents](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute);
/// 2. [A note on the algebra of CuTe Layouts](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf).
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deref, DerefMut, From, Into, Display)]
#[display("<{}, {}>", self.shape(), self.stride())]
pub struct Layout(Arc<[(usize, usize)]>);

impl From<Vec<(usize, usize)>> for Layout {
    #[inline]
    fn from(value: Vec<(usize, usize)>) -> Self {
        Self(value.into())
    }
}

impl AsRef<Layout> for Layout {
    #[inline]
    fn as_ref(&self) -> &Layout {
        self
    }
}

pub trait IntoLayout {
    fn into_layout(self) -> Layout;
}

impl<S, D> IntoLayout for (S, D)
where
    S: Into<Shape>,
    D: Into<Stride>,
{
    #[inline]
    fn into_layout(self) -> Layout {
        let (shape, stride) = self;
        Layout::from_shape_stride(shape.into(), stride.into())
    }
}

impl<S> IntoLayout for S
where
    S: Into<Shape>,
{
    #[inline]
    fn into_layout(self) -> Layout {
        Layout::from_shape(self.into())
    }
}

impl IntoLayout for Vec<(usize, usize)> {
    #[inline]
    fn into_layout(self) -> Layout {
        self.into()
    }
}

impl Layout {
    #[inline]
    pub fn from_shape(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let stride = Stride(
            shape
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
        Self(shape.iter().copied().zip_eq(stride.to_vec()).collect())
    }

    /// Retrieves the shape of the layout.
    #[inline]
    pub fn shape(&self) -> Shape {
        Shape(self.iter().map(|&(x, _)| x).collect())
    }

    /// Retrieves the stride of the layout.
    #[inline]
    pub fn stride(&self) -> Stride {
        Stride(self.iter().map(|&(_, x)| x).collect())
    }

    /// Retrieves the shape of a specific mode in the layout.
    #[inline]
    pub fn shape_of(&self, mode: usize) -> usize {
        self[mode].0
    }

    /// Retrieves the stride of a specific mode in the layout.
    #[inline]
    pub fn stride_of(&self, mode: usize) -> usize {
        self[mode].1
    }

    /// Returns `true` if the layout is of size 0.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.shape().is_zero()
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

    /// The whole span of the layout.
    #[inline]
    pub fn full_size(&self) -> usize {
        let layout = self.filter().sort();
        match layout.last() {
            Some((n, d)) => n * d,
            None => 0,
        }
    }

    /// Maps a linear index to a multi-dimensional coordinate.
    #[inline]
    pub fn iota(&self, index: usize) -> Coord {
        self.iter()
            .fold((vec![], 1), |(mut v, p), &(m, _)| {
                v.push((index / p) % m);
                (v, p * m)
            })
            .0
            .into()
    }

    /// Same as [`Layout::iota`], but not limited by the bound of the highest dimension.
    #[inline]
    pub fn iota_extend(&self, index: usize) -> Coord {
        self.iter()
            .enumerate()
            .fold((vec![], 1), |(mut v, p), (x, &(m, _))| {
                match x {
                    x if x + 1 == self.len() => v.push(index / p),
                    _ => v.push((index / p) % m),
                };
                (v, p * m)
            })
            .0
            .into()
    }

    /// Returns `true` if two layouts are totally equal as index mappings.
    /// Note that this check is exponentially slow so only use it in tests.
    #[inline]
    pub fn check_isomorphic(&self, other: impl AsRef<Layout>) -> bool {
        match self.size() == other.as_ref().size() {
            true => (0..self.size()).all(|index| self.value(index) == other.as_ref().value(index)),
            false => false,
        }
    }

    /// Returns `true` if all modes cover disjoint ranges.
    #[inline]
    pub fn check_disjoint(&self) -> bool {
        if self.is_zero() {
            return true;
        }
        self.iter()
            .filter(|(n, _)| *n > 1)
            .map(|&(n, d)| (d, (n - 1) * d))
            .tuple_combinations()
            .all(|((a, b), (c, d))| b < c || a > d)
    }

    /// Simplifies a layout to some length.
    #[inline]
    pub fn coalesce_to(&self, len: usize) -> Self {
        let mut layout = self.to_vec();
        loop {
            if layout.len() <= len.max(1) {
                break Self::from(layout);
            }
            let coalesced = [&layout[..], &[(1, 0)]]
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
                break Self::from(layout);
            }
            layout = coalesced;
        }
    }

    /// Returns a mostly simplified coalesce of the layout.
    #[inline]
    pub fn coalesce(&self) -> Self {
        self.coalesce_to(1)
    }

    /// Returns the layout whose strides are in sorted order.
    #[inline]
    pub fn sort(&self) -> Self {
        Self(
            self.iter()
                .cloned()
                .sorted_by_key(|&(n, d)| (d, n))
                .collect(),
        )
    }

    /// Removes stride-0 or size-1 modes.
    #[inline]
    pub fn filter(&self) -> Self {
        Self(
            self.iter()
                .cloned()
                .filter(|&(n, d)| d != 0 && n != 1)
                .collect(),
        )
    }

    /// Complements of the layout to a given size, if being admissible for complement.
    #[inline]
    pub fn complement(&self, size: usize) -> Result<Self, LayoutError> {
        let layout = self.filter().sort();
        if layout.is_zero() {
            return Ok(Layout::from_shape([size]));
        }

        let stride = layout.stride();
        let product = layout.iter().map(|(n, d)| n * d).collect_vec();

        let stride = [&stride[..], &[size]].concat(); // [d0, d1, ..., dα, M]
        let product = [vec![1], product].concat(); // [1, N0 d0, N1 d1, ..., Nα dα]

        let shape: Arc<_> = stride
            .iter()
            .zip_eq(product.iter())
            .map(|(d, p)| match d % p {
                0 => Ok(d / p),
                _ => Err(LayoutError::Complement(self.clone(), size)),
            })
            .try_collect()?;

        Ok(Self::from_shape_stride(shape, product).coalesce_to(self.len()))
    }

    /// Complement the layout to its full size, which is the least size that is admissible for completion.
    #[inline]
    pub fn complement_full(&self) -> Self {
        self.complement(self.full_size())
            .expect("this complement cannot fail")
    }

    /// Stack another layout onto `self`.
    #[inline]
    fn concat(&self, other: impl AsRef<Self>) -> Self {
        Self::from([&self[..], &other.as_ref()[..]].concat())
    }

    /// Make a tiler from this layout.
    #[inline]
    pub fn tiler(&self, tile: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let (shape, stride): (Vec<_>, Vec<_>) = tile.into_iter().unzip();
        let stride = stride
            .into_iter()
            .zip_eq(self.iter())
            .map(|(d, &(_, r))| d * r)
            .collect_vec();
        Self::from_shape_stride(shape, stride)
    }

    /// [Tile division](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#division-tiling).
    ///
    /// `A ⊘ B := A ∘ (B, B∗)`.
    #[inline]
    pub fn div(&self, tile: impl AsRef<Self>) -> Result<Self, LayoutError> {
        let tile = tile.as_ref();
        tile.concat(tile.complement(self.size())?).compose(self)
    }

    /// Shortcut for calling [`Self::div`] on `self.tiler(tile)`.
    #[inline]
    pub fn div_tiler(
        &self,
        tile: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<Self, LayoutError> {
        self.div(self.tiler(tile))
    }

    /// [Tile product](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#product-tiling).
    ///
    /// `A ⊗ B := (A, A∗ ∘ B)`.
    #[inline]
    pub fn prod(&self, tile: impl AsRef<Self>) -> Result<Self, LayoutError> {
        let tile = tile.as_ref();
        let size = self.size() * tile.full_size();
        Ok(self.concat(&tile.compose(self.complement(size)?)?))
    }
}

impl<T: AsRef<Coord>> IndexFn<T> for Layout {
    type Output = usize;

    #[inline]
    fn value(&self, index: T) -> usize {
        self.iter()
            .zip_eq(index.as_ref().iter())
            .map(|(&(_, d), &x)| d * x)
            .sum()
    }
}

impl<const N: usize> IndexFn<[usize; N]> for Layout {
    type Output = usize;

    #[inline]
    fn value(&self, index: [usize; N]) -> Self::Output {
        self.value(Coord::from(index))
    }
}

impl IndexFn<usize> for Layout {
    type Output = usize;

    #[inline]
    fn value(&self, index: usize) -> usize {
        let coord = self.iota_extend(index);
        self.value(&coord)
    }
}

impl<T: AsRef<Layout>> Compose<T> for (usize, usize) {
    type Output = Result<Layout, LayoutError>;

    fn compose(&self, f: T) -> Self::Output {
        let f = f.as_ref();
        let &(n, r) = self;
        match n {
            0 | 1 => Ok(Layout::from_shape_stride(n, r)),
            n => {
                let s = f.shape();
                let d = f.stride();
                let (q, r) = s.shape_div(r)?;
                match r.len() {
                    0 => Ok(Layout::from_shape_stride(n, 0)),
                    i if i == s.len() => Ok(Layout::from_shape_stride(n, r[i - 1] * d[i - 1])),
                    i => {
                        let i = i - 1;
                        let c = r[i];
                        let s = q.shape_mod(n)?;
                        let s = match s.last() {
                            Some(1) => Shape::from_slice(&s[..s.len() - 1]),
                            _ => s,
                        };
                        let mut d = d[i..i + s.len()].to_vec();
                        d[0] *= c;
                        Ok(Layout::from_shape_stride(s, d))
                    }
                }
            }
        }
    }
}

impl<T: AsRef<Layout>> Compose<T> for Layout {
    type Output = Result<Self, LayoutError>;

    /// Layout composition. `a.compose(b)` corresponds to `B ∘ A` in layout algebra.
    fn compose(&self, f: T) -> Self::Output {
        if !self.check_disjoint() {
            return Err(LayoutError::Disjoint(self.clone()));
        }
        let modes: Vec<_> = self
            .iter()
            .map(|&mode| mode.compose(f.as_ref()))
            .try_collect()?;
        let layout = modes
            .into_iter()
            .fold(Layout::default(), |acc, x| acc.concat(&x));
        Ok(layout)
    }
}

/// A swizzle functor.
/// See [CuTe documentation](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp#L44) for more info.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Swizzle {
    pub base: usize,
    pub bits: usize,
    pub shift: usize,
}

impl IndexFn<usize> for Swizzle {
    type Output = usize;

    fn value(&self, index: usize) -> Self::Output {
        assert!(self.shift >= self.bits);
        let mask = (1 << self.bits) - 1;
        let mask = mask << (self.base + self.shift);
        index ^ ((index & mask) >> self.shift)
    }
}

impl Compose<Swizzle> for Layout {
    type Output = ComposedFn<Layout, Swizzle>;

    #[inline]
    fn compose(&self, f: Swizzle) -> Self::Output {
        ComposedFn(self.clone(), f)
    }
}

/// Composition of 2 (possibly different types of) index functions `t` and `f`, i.e., `f ∘ t`.
#[derive(Debug, Clone)]
pub struct ComposedFn<T, F>(pub T, pub F);

impl<T, F, I, J, K> IndexFn<I> for ComposedFn<T, F>
where
    T: IndexFn<I, Output = J>,
    F: IndexFn<J, Output = K>,
{
    type Output = K;

    #[inline]
    fn value(&self, index: I) -> Self::Output {
        self.1.value(self.0.value(index))
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::{Compose, Coord, IndexFn, IntoLayout, Layout, LayoutError, Shape, Swizzle};

    #[allow(unused)]
    fn print_tensor(data: &[usize], layout: impl AsRef<Layout>) {
        let layout = layout.as_ref();
        assert_eq!(data.len(), layout.size());

        let sketch = (0..layout.size())
            .map(|index| data[layout.value(index)])
            .collect_vec();

        let n = layout.len() / 2;
        let x: usize = layout.iter().take(n).map(|&(n, _)| n).product();
        let y: usize = layout.iter().skip(n).map(|&(n, _)| n).product();

        println!("{layout}");
        for j in 0..y {
            for i in 0..x {
                print!("{}\t", sketch[i + j * x]);
            }
            println!()
        }
        println!()
    }

    #[allow(unused)]
    fn print_layout(layout: impl AsRef<Layout>) {
        let layout = layout.as_ref();
        let sketch = (0..layout.size())
            .map(|index| layout.value(index))
            .collect_vec();

        let n = layout.len() / 2;
        let x: usize = layout.iter().take(n).map(|&(n, _)| n).product();
        let y: usize = layout.iter().skip(n).map(|&(n, _)| n).product();

        println!("{layout}");
        for j in 0..y {
            for i in 0..x {
                print!("{:<3}→ {}\t", i + j * x, sketch[i + j * x]);
            }
            println!()
        }
        println!()
    }

    #[test]
    fn test_isomorphism() {
        let layout = Layout::from_shape_stride([2, 3, 4], [3, 1, 6]);

        assert_eq!(layout.iota(0), Coord::from([0, 0, 0]));
        assert_eq!(layout.iota(1), Coord::from([1, 0, 0]));
        assert_eq!(layout.iota(2), Coord::from([0, 1, 0]));
        assert_eq!(layout.iota(3), Coord::from([1, 1, 0]));
        assert_eq!(layout.iota(4), Coord::from([0, 2, 0]));
        assert_eq!(layout.iota(5), Coord::from([1, 2, 0]));

        assert_eq!(layout.iota(6), Coord::from([0, 0, 1]));
        assert_eq!(layout.iota(7), Coord::from([1, 0, 1]));
        assert_eq!(layout.iota(8), Coord::from([0, 1, 1]));
        assert_eq!(layout.iota(9), Coord::from([1, 1, 1]));
        assert_eq!(layout.iota(10), Coord::from([0, 2, 1]));
        assert_eq!(layout.iota(11), Coord::from([1, 2, 1]));

        assert_eq!(layout.iota(12), Coord::from([0, 0, 2]));
        assert_eq!(layout.iota(13), Coord::from([1, 0, 2]));
        assert_eq!(layout.iota(14), Coord::from([0, 1, 2]));
        assert_eq!(layout.iota(15), Coord::from([1, 1, 2]));
        assert_eq!(layout.iota(16), Coord::from([0, 2, 2]));
        assert_eq!(layout.iota(17), Coord::from([1, 2, 2]));

        assert_eq!(layout.iota(18), Coord::from([0, 0, 3]));
        assert_eq!(layout.iota(19), Coord::from([1, 0, 3]));
        assert_eq!(layout.iota(20), Coord::from([0, 1, 3]));
        assert_eq!(layout.iota(21), Coord::from([1, 1, 3]));
        assert_eq!(layout.iota(22), Coord::from([0, 2, 3]));
        assert_eq!(layout.iota(23), Coord::from([1, 2, 3]));

        assert_eq!(layout.iota(24), Coord::from([0, 0, 0]));
        assert_eq!(layout.iota(25), Coord::from([1, 0, 0]));

        assert_eq!(layout.iota_extend(0), Coord::from([0, 0, 0]));
        assert_eq!(layout.iota_extend(1), Coord::from([1, 0, 0]));
        assert_eq!(layout.iota_extend(2), Coord::from([0, 1, 0]));
        assert_eq!(layout.iota_extend(3), Coord::from([1, 1, 0]));
        assert_eq!(layout.iota_extend(4), Coord::from([0, 2, 0]));
        assert_eq!(layout.iota_extend(5), Coord::from([1, 2, 0]));

        assert_eq!(layout.iota_extend(6), Coord::from([0, 0, 1]));
        assert_eq!(layout.iota_extend(7), Coord::from([1, 0, 1]));
        assert_eq!(layout.iota_extend(8), Coord::from([0, 1, 1]));
        assert_eq!(layout.iota_extend(9), Coord::from([1, 1, 1]));
        assert_eq!(layout.iota_extend(10), Coord::from([0, 2, 1]));
        assert_eq!(layout.iota_extend(11), Coord::from([1, 2, 1]));

        assert_eq!(layout.iota_extend(12), Coord::from([0, 0, 2]));
        assert_eq!(layout.iota_extend(13), Coord::from([1, 0, 2]));
        assert_eq!(layout.iota_extend(14), Coord::from([0, 1, 2]));
        assert_eq!(layout.iota_extend(15), Coord::from([1, 1, 2]));
        assert_eq!(layout.iota_extend(16), Coord::from([0, 2, 2]));
        assert_eq!(layout.iota_extend(17), Coord::from([1, 2, 2]));

        assert_eq!(layout.iota_extend(18), Coord::from([0, 0, 3]));
        assert_eq!(layout.iota_extend(19), Coord::from([1, 0, 3]));
        assert_eq!(layout.iota_extend(20), Coord::from([0, 1, 3]));
        assert_eq!(layout.iota_extend(21), Coord::from([1, 1, 3]));
        assert_eq!(layout.iota_extend(22), Coord::from([0, 2, 3]));
        assert_eq!(layout.iota_extend(23), Coord::from([1, 2, 3]));

        assert_eq!(layout.iota_extend(24), Coord::from([0, 0, 4]));
        assert_eq!(layout.iota_extend(25), Coord::from([1, 0, 4]));
    }

    #[test]
    fn test_coalesce() {
        fn check(layout: Layout) {
            let coalesced = layout.coalesce();
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
        fn check(layout: &Layout, size: usize) -> Result<(), LayoutError> {
            let complement = layout.complement(size)?;
            println!("{{{layout}, {size}}} → {complement}");

            // 1. disjoint
            assert!((1..layout.size()).all(|index| layout.value(index) != complement.value(index)));

            // 2. ordered
            assert!((0..complement.size())
                .tuple_windows::<(_, _)>()
                .all(|(x, y)| complement.value(x) < complement.value(y)));

            // 3. bounded
            if layout.size() > 0 {
                assert!(complement.size() >= size / layout.size());
                assert!(complement.co_size() <= size / layout.co_size() * layout.co_size());
            }

            // 4. complement
            assert!(layout.concat(&complement).complement_full().size() <= 1);

            Ok(())
        }

        {
            let layout = Layout::from_shape_stride(1, 0);
            check(&layout, 1)?;
            check(&layout, 2)?;
            check(&layout, 5)?;
        }

        {
            let layout = Layout::from_shape_stride(1, 1);
            check(&layout, 1)?;
            check(&layout, 2)?;
            check(&layout, 5)?;
        }

        {
            let layout = Layout::from_shape_stride(1, 2);
            check(&layout, 1)?;
            check(&layout, 5)?;
            check(&layout, 2)?;
            check(&layout, 8)?;
        }

        {
            let layout = Layout::from_shape_stride(4, 0);
            check(&layout, 1)?;
            check(&layout, 2)?;
            check(&layout, 8)?;
        }

        {
            let layout = Layout::from_shape_stride(4, 1);
            assert!(check(&layout, 1).is_err());
            assert!(check(&layout, 2).is_err());
            check(&layout, 4)?;
            check(&layout, 8)?;
        }

        {
            let layout = Layout::from_shape_stride(4, 2);
            check(&layout, 8)?;
            check(&layout, 16)?;
            assert!(check(&layout, 19).is_err());
        }

        check(&Layout::from_shape_stride(4, 4), 16)?;
        check(&Layout::from_shape_stride(4, 4), 16)?;
        check(&Layout::from_shape_stride([2, 2], [4, 1]), 32)?;

        Ok(())
    }

    #[test]
    fn test_shape_div() {
        fn check(s: impl Into<Shape>) {
            let s: Shape = s.into();
            (0..=s.size())
                .filter_map(|d| s.shape_div(d).map(|(q, r)| (d, q, r)).ok())
                .for_each(|(d, q, r)| println!("{s} = {r} + {d} • {q}"));
            println!()
        }

        check([2, 0, 3]);
        check([2, 6, 4]);
    }

    #[test]
    fn test_composition() -> Result<(), LayoutError> {
        fn check(a: impl IntoLayout, b: impl IntoLayout) -> Result<(), LayoutError> {
            let a = a.into_layout();
            let b = b.into_layout();

            let c = a.compose(&b)?;
            println!("{b} ∘ {a} → {c}\n");
            print_layout(&a);
            print_layout(&b);
            print_layout(&c);
            println!("------\n");

            assert_eq!(c.size(), a.size());
            assert!((0..a.size()).all(|index| b.value(a.value(index)) == c.value(index)));

            Ok(())
        }

        check(([1], [0]), ([1], [0]))?;
        check(([1], [0]), ([1], [1]))?;
        check(([1], [1]), ([1], [0]))?;
        check(([1], [1]), ([1], [1]))?;

        check(([4], [1]), ([4], [2]))?;
        check(([4], [1]), ([4], [0]))?;
        check(([4], [0]), ([4], [1]))?;

        check(([2], [1]), ([4], [1]))?;
        check(([2], [1]), ([4], [2]))?;
        check(([2], [2]), ([4], [2]))?;

        check([12], [4, 3])?;
        check([4, 3], [12])?;
        check([4, 3], ([12], [2]))?;
        check(([4, 3], [3, 1]), [12])?;
        check(([4, 3], [3, 1]), ([12], [2]))?;
        check(([2, 3], [2, 4]), [12])?;
        check([4, 3], [4, 3])?;
        check(([6], [2]), [4, 3])?;
        check(([6, 2], [2, 1]), [4, 3])?;
        check([4, 3], ([4, 3], [3, 1]))?;
        check([12], ([4, 3], [3, 1]))?;
        check(([6], [2]), ([4, 3], [3, 1]))?;
        check(([6, 2], [2, 1]), ([4, 3], [3, 1]))?;

        check(([2, 2, 2, 2, 2, 2], [1, 16, 4, 8, 2, 32]), [8, 8])?;
        check(([2, 2, 2, 2, 2, 2], [1, 16, 4, 8, 2, 32]), ([8, 8], [8, 1]))?;

        check(([4, 2], [2, 1]), ([4, 2], [1, 16]))?;
        check(([2, 2], [2, 1]), ([2, 2], [2, 1]))?;

        check(([2, 2, 2], [2, 8, 1]), [4, 8, 2])?;
        check(([2, 2, 2], [1, 8, 2]), ([4, 8, 2], [2, 8, 1]))?;
        check(([4, 2, 2], [2, 8, 1]), ([4, 8, 2], [2, 8, 1]))?;

        // last mode gets extended
        check([24], ([4, 3], [3, 1]))?;

        // last mode extension even without last mode divisibility
        check([8], ([4, 3], [3, 1]))?;

        // capping a layout with 1:0 extends in stride-0
        check([24], ([4, 3, 1], [3, 1, 0]))?;

        // disjoint requirement
        assert!(check(([3, 2], [2, 3]), ([6, 2], [1, 7])).is_err());

        Ok(())
    }

    #[test]
    fn test_div() -> Result<(), LayoutError> {
        fn check(layout: impl IntoLayout, tile: impl IntoLayout) -> Result<(), LayoutError> {
            let a = layout.into_layout();
            let b = tile.into_layout();
            let c = a.div(&b)?;

            println!("{a} / {b} = {c}\n");
            print_layout(&a);
            print_layout(&b);
            print_layout(&c);
            println!("------\n");

            Ok(())
        }

        check([6, 1], [2, 3])?;
        check([6, 1], ([2, 3], [3, 1]))?;

        check([6, 2], [2, 3])?;
        check([6, 2], ([2, 3], [3, 1]))?;

        check(([6, 6], [1, 12]), ([6, 3], [3, 1]))?;
        check(([6, 6], [12, 1]), ([6, 3], [3, 1]))?;

        check([4, 6], ([2, 2], [1, 4]))?;
        check([4, 6], ([2, 2], [2, 4]))?;

        check([16, 4], [4])?;
        check(([4, 2, 3], [2, 1, 8]), ([4], [2]))?;

        Ok(())
    }

    #[test]
    fn test_prod() -> Result<(), LayoutError> {
        fn check(layout: impl IntoLayout, tile: impl IntoLayout) -> Result<(), LayoutError> {
            let a = layout.into_layout();
            let b = tile.into_layout();
            let c = a.prod(&b)?;

            println!("{a} • {b} = {c}\n");
            print_layout(&a);
            print_layout(&b);
            print_layout(&c);
            println!("------\n");

            Ok(())
        }

        check(([2, 2], [4, 1]), ([6], [1]))?;
        check(([2, 5], [5, 1]), ([3, 2], [1, 3]))?;

        Ok(())
    }

    #[test]
    fn test_transpose() -> Result<(), LayoutError> {
        let x = 8;
        let y = 6;

        let src = (0..x * y).collect_vec();
        let mut dst = vec![0usize; src.len()];
        let mut tmp = vec![0usize; src.len()];

        let swizzle = Swizzle {
            base: 0,
            bits: 3,
            shift: 3,
        };

        let layout_u = Layout::from_shape_stride([x, y], [1, x]);
        let layout_v = Layout::from_shape_stride([y, x], [1, y]);
        let layout_v_s = layout_v.compose(swizzle);

        let layout_t = Layout::from_shape_stride([x, y], [y, 1]);
        let layout_t_s = layout_t.compose(swizzle);

        for i in 0..src.len() {
            tmp[layout_t_s.value(i)] = src[layout_u.value(i)];
        }

        for i in 0..src.len() {
            dst[layout_v.value(i)] = tmp[layout_v_s.value(i)];
        }

        print_tensor(&src, &layout_u);
        print_tensor(&tmp, &layout_v);
        print_tensor(&dst, &layout_v);
        print_tensor(&dst, &layout_t);

        for i in 0..src.len() {
            assert_eq!(src[layout_u.value(i)], dst[layout_t.value(i)]);
        }

        Ok(())
    }

    #[test]
    fn test_gemm_nt() -> Result<(), LayoutError> {
        let (m, n, k) = (4, 8, 12);

        let a = (0..m * k).collect_vec();
        let b = (0..n * k).collect_vec();
        let mut c = vec![0; m * n];

        let layout_a = Layout::from_shape([m, k]); // M-major
        let layout_b = Layout::from_shape([n, k]); // N-major
        let layout_c = Layout::from_shape([m, n]); // M-major

        print_tensor(&a, &layout_a);
        print_tensor(&b, &layout_b);

        let (bm, bn, bk) = (2, 2, 4);

        let layout_ta = layout_a.div_tiler([(bm, 1), (bk, 1)])?;
        let layout_tb = layout_b.div_tiler([(bn, 1), (bk, 1)])?;
        let layout_tc = layout_c.div_tiler([(bm, 1), (bn, 1)])?;

        print_layout(&layout_ta);
        print_layout(&layout_tb);
        print_layout(&layout_tc);

        let mut sa = vec![0; bm * bk];
        let mut sb = vec![0; bn * bk];

        let layout_sa = Layout::from_shape([bm, bk]);
        let layout_sb = Layout::from_shape([bn, bk]);

        for (k, j, i) in itertools::iproduct!(
            0..layout_ta.shape_of(3),
            0..layout_tc.shape_of(3),
            0..layout_tc.shape_of(2)
        ) {
            for (y, x) in itertools::iproduct!(0..bk, 0..bm) {
                sa[layout_sa.value([x, y])] = a[layout_ta.value([x, y, i, k])];
            }
            for (y, x) in itertools::iproduct!(0..bk, 0..bn) {
                sb[layout_sb.value([x, y])] = b[layout_tb.value([x, y, j, k])];
            }
            for (z, y, x) in itertools::iproduct!(0..bk, 0..bn, 0..bm) {
                let ra = sa[layout_sa.value([x, z])];
                let rb = sb[layout_sb.value([y, z])];
                c[layout_tc.value([x, y, i, j])] += ra * rb;
            }
        }

        print_tensor(&c, &layout_c);

        let mut xc = vec![0; m * n];
        for (z, y, x) in itertools::iproduct!(0..k, 0..n, 0..m) {
            let ra = a[layout_a.value([x, z])];
            let rb = b[layout_b.value([y, z])];
            xc[layout_c.value([x, y])] += ra * rb;
        }

        assert_eq!(c, xc);

        Ok(())
    }
}
