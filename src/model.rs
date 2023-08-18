use bitflags::bitflags;
use derive_getters::Getters;
use half::f16;

use crate::{
    context::Context,
    tensor::{ReadWrite, TensorGpu},
};

#[derive(Debug, Getters)]
pub struct Model {
    pub(crate) info: ModelInfo,
    pub(crate) context: Context,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub num_emb: usize,
    pub num_vocab: usize,
}

bitflags! {
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct LayerFlags: u64 {
    }
}

impl LayerFlags {
    pub fn from_layer(layer: u64) -> LayerFlags {
        LayerFlags::from_bits_retain(1 << layer)
    }

    pub fn contains_layer(&self, layer: u64) -> bool {
        self.contains(LayerFlags::from_layer(layer))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum Quantization {
    /// No quantization.
    #[default]
    None,
    /// Use int8 quantization, given layers to be quantized.
    Int8(LayerFlags),
}

pub enum Matrix<'a> {
    Fp16(TensorGpu<'a, f16, ReadWrite>),
    Int8 {
        w: TensorGpu<'a, u8, ReadWrite>,
        mx: TensorGpu<'a, f32, ReadWrite>,
        rx: TensorGpu<'a, f32, ReadWrite>,
        my: TensorGpu<'a, f32, ReadWrite>,
        ry: TensorGpu<'a, f32, ReadWrite>,
    },
}

pub struct LayerNorm<'a> {
    w: TensorGpu<'a, f32, ReadWrite>,
    b: TensorGpu<'a, f32, ReadWrite>,
}

pub struct Att<'a> {
    time_decay: TensorGpu<'a, f32, ReadWrite>,
    time_first: TensorGpu<'a, f32, ReadWrite>,

    time_mix_k: TensorGpu<'a, f16, ReadWrite>,
    time_mix_v: TensorGpu<'a, f16, ReadWrite>,
    time_mix_r: TensorGpu<'a, f16, ReadWrite>,

    w_k: Matrix<'a>,
    w_v: Matrix<'a>,
    w_r: Matrix<'a>,
    w_o: Matrix<'a>,
}

pub struct Ffn<'a> {
    time_mix_k: TensorGpu<'a, f16, ReadWrite>,
    time_mix_v: TensorGpu<'a, f16, ReadWrite>,
}
