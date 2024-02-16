use half::f16;

use super::ops::Activation;
use crate::{
    num::Float,
    tensor::{
        kind::{ReadWrite, Uniform},
        ops::{TensorOp, TensorPass},
        shape::Shape,
        TensorError, TensorGpu, TensorShape, TensorView,
    },
};

#[derive(Debug)]
pub enum Matrix {
    Fp16(TensorGpu<f16, ReadWrite>),
    Int8 {
        w: TensorGpu<u8, ReadWrite>,
        m: TensorGpu<f16, ReadWrite>,
    },
    NF4 {
        w: TensorGpu<u8, ReadWrite>,
        q: TensorGpu<f32, Uniform>,
        m: TensorGpu<f16, ReadWrite>,
    },
}

impl Matrix {
    pub fn matmul_vec_op(
        &self,
        input: TensorView<impl Float>,
        output: TensorView<impl Float>,
        active: Activation,
    ) -> Result<TensorOp, TensorError> {
        match self {
            Matrix::Fp16(matrix) => TensorOp::matmul_vec_fp16(matrix, input, output, active),
            Matrix::Int8 { w, m } => TensorOp::matmul_vec_int8(w, m, input, output, active),
            Matrix::NF4 { w, q, m } => TensorOp::matmul_vec_nf4(w, q, m, input, output, active),
        }
    }

    pub fn matmul_mat_op(
        &self,
        input: TensorView<impl Float>,
        output: TensorView<impl Float>,
        active: Activation,
    ) -> Result<TensorOp, TensorError> {
        match self {
            Matrix::Fp16(matrix) => {
                TensorOp::matmul_mat_fp16(matrix.view(.., .., .., ..)?, input, output, active)
            }
            Matrix::Int8 { w, m } => {
                TensorOp::matmul_mat_int8(w.view(.., .., .., ..)?, m, input, output, active)
            }
            Matrix::NF4 { w, q, m } => {
                TensorOp::matmul_mat_nf4(w.view(.., .., .., ..)?, q, m, input, output, active)
            }
        }
    }

    pub fn matmul_op(
        &self,
        input: TensorView<impl Float>,
        output: TensorView<impl Float>,
        active: Activation,
        turbo: bool,
    ) -> Result<TensorOp, TensorError> {
        match turbo {
            true => self.matmul_mat_op(input, output, active),
            false => self.matmul_vec_op(input, output, active),
        }
    }

    pub fn quant_u8(matrix: &TensorGpu<f16, ReadWrite>) -> Result<Self, TensorError> {
        let context = matrix.context();
        let shape = matrix.shape();

        let w = context.tensor_init(shape);
        let m = context.tensor_init(Shape::new(
            (shape[0] << 1) / TensorOp::INT8_BLOCK_SIZE as usize,
            shape[1],
            shape[2],
            shape[3],
        ));

        let op = TensorOp::quantize_mat_int8(matrix, &m, &w)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        context.queue.submit(Some(encoder.finish()));

        Ok(Matrix::Int8 { w, m })
    }

    pub fn quant_nf4(matrix: &TensorGpu<f16, ReadWrite>) -> Result<Self, TensorError> {
        let context = matrix.context();
        let shape = matrix.shape();

        let matrix_shape = Shape::new(shape[0] / 2, shape[1], shape[2], shape[3]);
        let absmax_shape = Shape::new(
            shape[0] / TensorOp::NF4_BLOCK_SIZE as usize,
            shape[1],
            shape[2],
            shape[3],
        );

        #[allow(clippy::excessive_precision)]
        let quant = vec![
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ];
        let q = context.tensor_from_data(Shape::new(quant.len(), 1, 1, 1), quant)?;

        let w = context.tensor_init(matrix_shape);
        let m = context.tensor_init(absmax_shape);

        let op = TensorOp::quantize_mat_nf4(matrix, &q, &m, &w)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        context.queue.submit(Some(encoder.finish()));

        Ok(Matrix::NF4 { w, q, m })
    }
}
