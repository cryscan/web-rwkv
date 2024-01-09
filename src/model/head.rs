use half::f16;

use crate::tensor::{ReadWrite, TensorGpu};

pub trait ModelHead {
    fn head(&self) -> TensorGpu<f16, ReadWrite>;
}
