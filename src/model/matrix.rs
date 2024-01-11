use half::f16;

use crate::tensor::{
    ops::{TensorOp, TensorPass},
    shape::Shape,
    ReadWrite, TensorError, TensorGpu, TensorShape, TensorView, Uniform,
};

#[derive(Debug)]
pub enum Matrix {
    Fp16(TensorGpu<f16, ReadWrite>),
    Int8 {
        w: TensorGpu<u8, ReadWrite>,
        mx: TensorGpu<f32, ReadWrite>,
        rx: TensorGpu<f32, ReadWrite>,
        my: TensorGpu<f32, ReadWrite>,
        ry: TensorGpu<f32, ReadWrite>,
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
        half: TensorView<f16>,
        input: TensorView<f32>,
        output: TensorView<f32>,
    ) -> Result<TensorOp, TensorError> {
        match self {
            Matrix::Fp16(matrix) => TensorOp::matmul_vec_fp16(matrix, input, output),
            Matrix::Int8 { w, mx, rx, my, ry } => {
                TensorOp::matmul_vec_int8(w, mx, rx, my, ry, input, output)
            }
            Matrix::NF4 { w, q, m } => Ok(TensorOp::List(vec![
                TensorOp::quantize_fp16(input, half.clone())?,
                TensorOp::matmul_vec_nf4(w, q, m, half, output)?,
            ])),
        }
    }

    pub fn matmul_mat_op(
        &self,
        half: TensorView<f16>,
        input: TensorView<f32>,
        output: TensorView<f32>,
    ) -> Result<TensorOp, TensorError> {
        match self {
            Matrix::Fp16(matrix) => Ok(TensorOp::List(vec![
                TensorOp::quantize_fp16(input, half.clone())?,
                TensorOp::matmul_mat_fp16(matrix.view(.., .., .., ..)?, half, output)?,
            ])),
            Matrix::Int8 { w, mx, rx, my, ry } => Ok(TensorOp::List(vec![
                TensorOp::quantize_fp16(input, half.clone())?,
                TensorOp::matmul_mat_int8(w.view(.., .., .., ..)?, mx, rx, my, ry, half, output)?,
            ])),
            Matrix::NF4 { w, q, m } => Ok(TensorOp::List(vec![
                TensorOp::quantize_fp16(input, half.clone())?,
                TensorOp::matmul_mat_nf4(w.view(.., .., .., ..)?, q, m, half, output)?,
            ])),
        }
    }

    pub fn matmul_op(
        &self,
        half: TensorView<f16>,
        input: TensorView<f32>,
        output: TensorView<f32>,
        turbo: bool,
    ) -> Result<TensorOp, TensorError> {
        match turbo {
            true => self.matmul_mat_op(half, input, output),
            false => self.matmul_vec_op(half, input, output),
        }
    }

    pub fn quant_u8(matrix: &TensorGpu<f16, ReadWrite>) -> Result<Self, TensorError> {
        let context = matrix.context();
        let shape = matrix.shape();

        // let mx_f32 = context.init_tensor(Shape::new(shape[0], 1, 1, 1));
        // let rx_f32 = context.init_tensor(Shape::new(shape[0], 1, 1, 1));
        // let my_f32 = context.init_tensor(Shape::new(shape[1], 1, 1, 1));
        // let ry_f32 = context.init_tensor(Shape::new(shape[1], 1, 1, 1));

        let w = context.tensor_init(matrix.shape());

        let mx = context.tensor_init(Shape::new(shape[0], 1, 1, 1));
        let rx = context.tensor_init(Shape::new(shape[0], 1, 1, 1));
        let my = context.tensor_init(Shape::new(shape[1], 1, 1, 1));
        let ry = context.tensor_init(Shape::new(shape[1], 1, 1, 1));

        let op = TensorOp::quantize_mat_int8(matrix, &mx, &rx, &my, &ry, &w)?;

        // ops.push(TensorOp::quantize_vec_fp16(&mx_f32, &mx)?);
        // ops.push(TensorOp::quantize_vec_fp16(&rx_f32, &rx)?);
        // ops.push(TensorOp::quantize_vec_fp16(&my_f32, &my)?);
        // ops.push(TensorOp::quantize_vec_fp16(&ry_f32, &ry)?);

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        context.queue.submit(Some(encoder.finish()));

        Ok(Matrix::Int8 { w, mx, rx, my, ry })
    }

    pub fn quant_nf4(matrix: &TensorGpu<f16, ReadWrite>) -> Result<Self, TensorError> {
        let context = matrix.context();
        let shape = matrix.shape();

        let matrix_shape = Shape::new(shape[0] / 2, shape[1], shape[2], shape[3]);
        let absmax_shape = Shape::new(
            shape[0] / TensorOp::NF4_BLOCK_SIZE,
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
