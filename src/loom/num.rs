use bytemuck::{Pod, Zeroable};
use half::f16;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F16,
    U8,
    U16,
    U32,
    PackedU4x8,
    PackedU8x4,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct PackedU4x8(pub u32);

unsafe impl Zeroable for PackedU4x8 {}

unsafe impl Pod for PackedU4x8 {}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct PackedU8x4(pub u32);

unsafe impl Zeroable for PackedU8x4 {}

unsafe impl Pod for PackedU8x4 {}

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f16 {
    fn zero() -> Self {
        Self::ZERO
    }
}

impl Zero for u8 {
    fn zero() -> Self {
        0
    }
}

impl Zero for u16 {
    fn zero() -> Self {
        0
    }
}

impl Zero for u32 {
    fn zero() -> Self {
        0
    }
}

impl Zero for PackedU4x8 {
    fn zero() -> Self {
        Self(0)
    }
}

impl Zero for PackedU8x4 {
    fn zero() -> Self {
        Self(0)
    }
}

pub trait One {
    fn one() -> Self;
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}

impl One for f16 {
    fn one() -> Self {
        Self::ONE
    }
}

impl One for u8 {
    fn one() -> Self {
        1
    }
}

impl One for u16 {
    fn one() -> Self {
        1
    }
}

impl One for u32 {
    fn one() -> Self {
        1
    }
}

impl One for PackedU4x8 {
    fn one() -> Self {
        Self(0x11111111)
    }
}

impl One for PackedU8x4 {
    fn one() -> Self {
        Self(0x01010101)
    }
}

pub trait Scalar: Sized + Pod + Zero + Send + Sync {
    const DATA_TYPE: DataType;
}

pub trait Float: Scalar {}

impl Scalar for f32 {
    const DATA_TYPE: DataType = DataType::F32;
}

impl Scalar for f16 {
    const DATA_TYPE: DataType = DataType::F16;
}

impl Scalar for u8 {
    const DATA_TYPE: DataType = DataType::U8;
}

impl Scalar for u16 {
    const DATA_TYPE: DataType = DataType::U16;
}

impl Scalar for u32 {
    const DATA_TYPE: DataType = DataType::U32;
}

impl Scalar for PackedU4x8 {
    const DATA_TYPE: DataType = DataType::PackedU4x8;
}

impl Scalar for PackedU8x4 {
    const DATA_TYPE: DataType = DataType::PackedU8x4;
}

impl Float for f32 {}
impl Float for f16 {}
