use half::f16;

use crate::tensor::{ops::TensorOp, ReadWrite, TensorError, TensorGpu};

pub trait ModelHead {
    /// Tensor op that maps vectors from embed space into vocab space.
    fn head_op(
        &self,
        input: &TensorGpu<f32, ReadWrite>,
        half: &TensorGpu<f16, ReadWrite>,
        output: &TensorGpu<f32, ReadWrite>,
        turbo: bool,
    ) -> Result<TensorOp, TensorError>;
}
