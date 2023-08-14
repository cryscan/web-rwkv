use bitflags::bitflags;
use derive_getters::Getters;

use crate::Context;

#[derive(Debug, Clone, Getters)]
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
