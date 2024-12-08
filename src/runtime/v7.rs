use half::f16;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::DeserializeSeed;

use super::model::ModelInfo;
use crate::{
    context::Context,
    tensor::{kind::ReadWrite, matrix::Matrix, serialization::Seed, TensorCpu, TensorGpu},
};

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Model {
    pub context: Context,
    pub info: ModelInfo,
    pub rescale: usize,
}

impl Model {
    pub const L2_EPS: f32 = 1.0e-12;
    pub const LN_EPS: f32 = 1.0e-5;
    pub const GN_EPS: f32 = 64.0e-5;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CustomInfo {
    pub w: usize,
    pub a: usize,
    pub g: usize,
    pub v: usize,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct ModelTensor {
    pub embed: Embed,
    pub head: Head,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct LayerNorm {
    pub w: TensorGpu<f16, ReadWrite>,
    pub b: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Att {
    pub x_r: TensorGpu<f16, ReadWrite>,
    pub x_w: TensorGpu<f16, ReadWrite>,
    pub x_k: TensorGpu<f16, ReadWrite>,
    pub x_v: TensorGpu<f16, ReadWrite>,
    pub x_a: TensorGpu<f16, ReadWrite>,
    pub x_g: TensorGpu<f16, ReadWrite>,

    pub w0: TensorGpu<f16, ReadWrite>,
    pub w1: TensorGpu<f16, ReadWrite>,
    pub w2: TensorGpu<f16, ReadWrite>,
    pub a0: TensorGpu<f16, ReadWrite>,
    pub a1: TensorGpu<f16, ReadWrite>,
    pub a2: TensorGpu<f16, ReadWrite>,
    pub g1: TensorGpu<f16, ReadWrite>,
    pub g2: TensorGpu<f16, ReadWrite>,
    pub v0: TensorGpu<f16, ReadWrite>,
    pub v1: TensorGpu<f16, ReadWrite>,
    pub v2: TensorGpu<f16, ReadWrite>,

    pub r_k: TensorGpu<f16, ReadWrite>,
    pub k_k: TensorGpu<f16, ReadWrite>,
    pub k_a: TensorGpu<f16, ReadWrite>,

    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
    pub w_o: Matrix,

    pub ln: LayerNorm,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Ffn {
    pub x_k: TensorGpu<f16, ReadWrite>,

    pub w_k: TensorGpu<f16, ReadWrite>,
    pub w_v: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Layer {
    pub att_ln: LayerNorm,
    pub ffn_ln: LayerNorm,
    pub att: Att,
    pub ffn: Ffn,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Embed {
    pub ln: LayerNorm,
    pub w: TensorCpu<f16>,
    pub u: Option<TensorGpu<f16, ReadWrite>>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Head {
    pub ln: LayerNorm,
    pub w: Matrix,
}
