use half::f16;

use crate::{
    num::Float,
    tensor::{kind::ReadWrite, ops::TensorOp, TensorError, TensorGpu},
};

pub trait ModelHead {
    /// Tensor op that maps vectors from embed space into vocab space.
    fn head_op(
        &self,
        input: &TensorGpu<f16, ReadWrite>,
        output: &TensorGpu<impl Float, ReadWrite>,
        turbo: bool,
    ) -> Result<TensorOp, TensorError>;
}
