use itertools::Itertools;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("complement error: layout {0} vs. size {1}")]
    Complement(Layout, usize),
    #[error("shape {0} is not left divisible by {1}")]
    ShapeDiv(Shape, usize),
}

/// An [`IndexFunction`] is a mapping that maps an index to another.
pub trait IndexFn<Index> {
    type Output;

    /// Sends an index to a mapped value.
    fn value(&self, index: Index) -> Self::Output;
}

pub trait Compose<F> {
    type Output;

    /// Apply `f` after `self`.
    fn compose(&self, f: F) -> Result<Self::Output, LayoutError>;
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<usize>);

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Self(value.into_iter().collect())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Self(value)
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
        if self.is_empty() {
            return Ok((self.clone(), Default::default()));
        }
        if d == 0 {
            assert_ne!(self.len(), 0);
            // in this case, since all nature numbers divide 0, we must have `i = α`, and c = 0
            let mut r = self.clone();
            r.0[self.len() - 1] = 0;
            return Ok((Default::default(), r));
        }

        // [`1`, `N0`, `N0 × N1`, ..., `N0 × N1 × ... × N(α - 1)`]
        // [`N0`, `N1`, ..., `N(α)`]
        let product = self
            .0
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
        let (r, q) = self.0.split_at(i);

        assert!(!q.is_empty());
        assert_eq!(q[0], n);

        let c = d / p;
        let r = [r, &[c]].concat();
        let q = [&[n / c], &q[1..]].concat();

        Ok((Shape(q), Shape(r)))
    }

    /// Performs weak shape division.
    /// Since `c` doesn't necessarily divides `N(i)`, we can only get the remainder here.
    pub fn shape_mod(&self, d: usize) -> Result<Self, LayoutError> {
        if d == 0 {
            return Err(LayoutError::ShapeDiv(self.clone(), d));
        }
        if self.is_empty() {
            return Ok(Default::default());
        }

        // [`1`, `N0`, `N0 × N1`, ..., `N0 × N1 × ... × N(α - 1)`]
        // [`N0`, `N1`, ..., `N(α)`]
        let product = self
            .0
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
        let r = &self.0[..i];

        let c = d / p;
        let r = [r, &[c]].concat();

        Ok(Shape(r))
    }
}

/// Defines the step to add to when increase 1 along coordinates.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Stride(pub Vec<usize>);

impl<const N: usize> From<[usize; N]> for Stride {
    fn from(value: [usize; N]) -> Self {
        Self(value.into_iter().collect())
    }
}

impl From<Vec<usize>> for Stride {
    fn from(value: Vec<usize>) -> Self {
        Self(value)
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

impl<const N: usize> From<[usize; N]> for Coord {
    fn from(value: [usize; N]) -> Self {
        Self(value.into_iter().collect())
    }
}

impl From<Vec<usize>> for Coord {
    fn from(value: Vec<usize>) -> Self {
        Self(value)
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
/// 1. [CuTe documents](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute);
/// 2. [A note on the algebra of CuTe Layouts](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf).
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Layout(pub Vec<(usize, usize)>);

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.shape();
        let stride = self.stride();
        write!(f, "<{shape}, {stride}>")
    }
}

impl<S, D> From<(S, D)> for Layout
where
    S: Into<Shape>,
    D: Into<Stride>,
{
    fn from((s, d): (S, D)) -> Self {
        Self::from_shape_stride(s, d)
    }
}

impl<S> From<S> for Layout
where
    S: Into<Shape>,
{
    fn from(s: S) -> Self {
        Self::from_shape(s)
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

    #[inline]
    pub fn concat(&self, other: &Layout) -> Self {
        Self([self.0.clone(), other.0.clone()].concat())
    }

    /// Simplifies a layout to some length.
    pub fn coalesce_to(&self, len: usize) -> Self {
        let mut layout = self.0.clone();

        loop {
            if layout.len() <= len.max(1) {
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

    /// Returns a mostly simplified coalesce of the layout.
    pub fn coalesce(&self) -> Self {
        self.coalesce_to(1)
    }

    /// Returns the layout whose strides are in sorted order.
    #[inline]
    pub fn sort(&self) -> Self {
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
    pub fn filter(&self) -> Self {
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
        let layout = self.filter().sort();
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

        Ok(Self::from_shape_stride(shape, product).coalesce())
    }

    /// Complement the layout to the least size for which is admissible.
    pub fn complement_full(&self) -> Self {
        let layout = self.filter().sort();
        let Some((n, d)) = layout.0.last() else {
            return Default::default();
        };
        self.complement(n * d).expect("this complement cannot fail")
    }
}

impl IndexFn<&Coord> for Layout {
    type Output = usize;

    #[inline]
    fn value(&self, index: &Coord) -> usize {
        self.0
            .iter()
            .zip_eq(index.0.iter())
            .map(|(&(_, d), &x)| d * x)
            .sum()
    }
}

impl IndexFn<Coord> for Layout {
    type Output = usize;

    #[inline]
    fn value(&self, index: Coord) -> usize {
        self.value(&index)
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

impl Compose<&Layout> for Layout {
    type Output = Layout;

    /// Layout composition. `a.compose(b)` corresponds to `B ◦ A` in layout algebra.
    fn compose(&self, f: &Layout) -> Result<Self::Output, LayoutError> {
        if self.is_empty() {
            return Ok(self.clone());
        }

        let modes: Vec<_> = self
            .0
            .iter()
            .map(|&(n, r)| {
                let s = f.shape();
                let d = f.stride();
                let (q, r) = s.shape_div(r)?;
                match r.len() {
                    0 => Ok(Layout::from_shape_stride([n], [0])),
                    i if i == s.len() => Ok(Layout::from_shape_stride([n], [r[i - 1] * d[i - 1]])),
                    i => {
                        let c = r[i - 1];
                        let s = q.shape_mod(n)?;
                        let mut d = Stride::from(d.0[i - 1..i - 1 + s.len()].to_vec());
                        d.0[0] *= c;
                        Ok(Layout::from_shape_stride(s, d))
                    }
                }
            })
            .try_collect()?;

        let layout = modes
            .into_iter()
            .fold(Layout::default(), |acc, x| acc.concat(&x));

        Ok(layout.coalesce_to(self.len()))
    }
}

impl Compose<Layout> for Layout {
    type Output = Layout;

    fn compose(&self, f: Layout) -> Result<Self::Output, LayoutError> {
        self.compose(&f)
    }
}

/// A swizzle functor.
/// See [CuTe documentation](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp#L44) for more info.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
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

    fn compose(&self, f: Swizzle) -> Result<Self::Output, LayoutError> {
        Ok(ComposedFn(self.clone(), f))
    }
}

impl Compose<&Swizzle> for Layout {
    type Output = ComposedFn<Layout, Swizzle>;

    fn compose(&self, f: &Swizzle) -> Result<Self::Output, LayoutError> {
        self.compose(f.clone())
    }
}

/// Composition of 2 (possibly different types of) index functions `t` and `f`, i.e., `f ◦ t`.
#[derive(Debug, Clone)]
pub struct ComposedFn<T, F>(pub T, pub F);

impl<T, F, I, J, K> IndexFn<I> for ComposedFn<T, F>
where
    T: IndexFn<I, Output = J>,
    F: IndexFn<J, Output = K>,
{
    type Output = K;

    fn value(&self, index: I) -> Self::Output {
        self.1.value(self.0.value(index))
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::{Compose, Coord, IndexFn, Layout, LayoutError, Shape, Swizzle};

    #[allow(unused)]
    fn print_tensor(data: &[usize], layout: &Layout) {
        assert_eq!(data.len(), layout.size());

        let mut sketch = vec![0; layout.size()];
        for i in 0..layout.size() {
            sketch[i] = data[layout.value(i)];
        }

        let n = layout.len() / 2;
        let x: usize = layout.0.iter().take(n).map(|&(n, _)| n).product();
        let y: usize = layout.0.iter().skip(n).map(|&(n, _)| n).product();

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
    fn print_layout(layout: &Layout) {
        let mut sketch = vec![0; layout.size()];
        for i in 0..layout.size() {
            sketch[i] = layout.value(i);
        }

        let n = layout.len() / 2;
        let x: usize = layout.0.iter().take(n).map(|&(n, _)| n).product();
        let y: usize = layout.0.iter().skip(n).map(|&(n, _)| n).product();

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
        fn check(layout: impl Into<Layout>) {
            let layout: Layout = layout.into();
            let coalesced = layout.coalesce();
            println!("{layout} → {coalesced}");
            assert!(layout.check_isomorphic(&coalesced))
        }

        check(([1], [0]));
        check(([1], [1]));

        check([2, 4]);
        check([2, 4, 6]);
        check([2, 4, 6, 2]);

        check(([2, 1, 6], [1, 6, 2]));
        check(([2, 1, 6], [1, 7, 2]));

        check(([2, 4, 6], [4, 1, 8]));
        check(([2, 1, 3], [1, 1, 2]));
        check(([2, 1, 3], [2, 4, 4]));
        check(([2, 1, 3], [2, 0, 4]));
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

            // 4. complement
            assert!(layout.concat(&complement).complement_full().size() <= 1);

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

    #[test]
    fn test_swizzle() -> Result<(), LayoutError> {
        let x = 16;
        let y = 8;

        let src: Vec<usize> = (0..x * y).collect();
        let mut dst = vec![0usize; src.len()];
        let mut tmp: Vec<usize> = vec![0usize; src.len()];

        let swizzle = Swizzle {
            base: 0,
            bits: 4,
            shift: 4,
        };

        let layout_u = Layout::from_shape_stride([x, y], [1, x]);
        let layout_v = Layout::from_shape_stride([y, x], [1, y]);
        let layout_v_s = Layout::from_shape_stride([y, x], [1, y]).compose(&swizzle)?;
        let layout_v_t = Layout::from_shape_stride([x, y], [y, 1]);

        for (j, i) in (0..y).cartesian_product(0..x) {
            let u = Coord::from([i, j]);
            let v = Coord::from([j, i]);
            tmp[layout_v_s.value(&v)] = src[layout_u.value(&u)];
        }

        for i in 0..src.len() {
            dst[layout_v.value(i)] = tmp[layout_v_s.value(i)];
        }

        print_tensor(&src, &layout_u);
        print_tensor(&tmp, &layout_v);
        print_tensor(&dst, &layout_v);
        print_tensor(&dst, &layout_v_t);

        for (j, i) in (0..y).cartesian_product(0..x) {
            let u = Coord::from([i, j]);
            assert_eq!(src[j * x + i], dst[i * y + j]);
            assert_eq!(src[layout_u.value(&u)], dst[layout_v_t.value(&u)]);
        }

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
    fn test_layout_composition() -> Result<(), LayoutError> {
        fn check(a: impl Into<Layout>, b: impl Into<Layout>) -> Result<(), LayoutError> {
            let a: Layout = a.into();
            let b: Layout = b.into();

            let c = a.compose(&b)?;
            println!("{b} ◦ {a} → {c}\n");
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

        Ok(())
    }
}
