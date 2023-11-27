use half::f16;
use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor};

use crate::tensor::{
    ops::{TensorOp, TensorPass},
    shape::Shape,
    ReadWrite, TensorError, TensorGpu, TensorShape, TensorView, Uniform,
};

#[derive(Debug)]
pub enum Matrix {
    Fp16(TensorGpu<f16, ReadWrite>),
    Int8 {
        w: Box<TensorGpu<u8, ReadWrite>>,
        mx: Box<TensorGpu<f32, ReadWrite>>,
        rx: Box<TensorGpu<f32, ReadWrite>>,
        my: Box<TensorGpu<f32, ReadWrite>>,
        ry: Box<TensorGpu<f32, ReadWrite>>,
    },
    NF4 {
        w: Box<TensorGpu<u8, ReadWrite>>,
        q: Box<TensorGpu<f32, Uniform>>,
        m: Box<TensorGpu<f16, ReadWrite>>,
    },
}

impl Matrix {
    pub fn matmul_vec_op<'a>(
        &'a self,
        half: TensorView<'a, f16>,
        input: TensorView<'a, f32>,
        output: TensorView<'a, f32>,
    ) -> Result<TensorOp<'a>, TensorError> {
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

    pub fn matmul_mat_op<'a>(
        &'a self,
        half: TensorView<'a, f16>,
        input: TensorView<'a, f32>,
        output: TensorView<'a, f32>,
    ) -> Result<TensorOp<'a>, TensorError> {
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

    pub fn matmul_op<'a>(
        &'a self,
        half: TensorView<'a, f16>,
        input: TensorView<'a, f32>,
        output: TensorView<'a, f32>,
        turbo: bool,
    ) -> Result<TensorOp<'a>, TensorError> {
        match turbo {
            true => self.matmul_mat_op(half, input, output),
            false => self.matmul_vec_op(half, input, output),
        }
    }

    pub fn quant_u8(matrix: &TensorGpu<f16, ReadWrite>) -> Result<Self, TensorError> {
        let context = &matrix.context;
        let shape = matrix.shape();

        // let mx_f32 = context.init_tensor(Shape::new(shape[0], 1, 1, 1));
        // let rx_f32 = context.init_tensor(Shape::new(shape[0], 1, 1, 1));
        // let my_f32 = context.init_tensor(Shape::new(shape[1], 1, 1, 1));
        // let ry_f32 = context.init_tensor(Shape::new(shape[1], 1, 1, 1));

        let w = Box::new(context.tensor_init(matrix.shape()));

        let mx = Box::new(context.tensor_init(Shape::new(shape[0], 1, 1, 1)));
        let rx = Box::new(context.tensor_init(Shape::new(shape[0], 1, 1, 1)));
        let my = Box::new(context.tensor_init(Shape::new(shape[1], 1, 1, 1)));
        let ry = Box::new(context.tensor_init(Shape::new(shape[1], 1, 1, 1)));

        let op = TensorOp::quantize_mat_int8(matrix, &mx, &rx, &my, &ry, &w)?;

        // ops.push(TensorOp::quantize_vec_fp16(&mx_f32, &mx)?);
        // ops.push(TensorOp::quantize_vec_fp16(&rx_f32, &rx)?);
        // ops.push(TensorOp::quantize_vec_fp16(&my_f32, &my)?);
        // ops.push(TensorOp::quantize_vec_fp16(&ry_f32, &ry)?);

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        context.queue.submit(Some(encoder.finish()));

        Ok(Matrix::Int8 { w, mx, rx, my, ry })
    }

    pub fn quant_nf4(matrix: &TensorGpu<f16, ReadWrite>) -> Result<Self, TensorError> {
        let context = &matrix.context;
        let shape = matrix.shape();

        let matrix_shape = Shape::new(shape[0] / 2, shape[1], shape[2], shape[3]);
        let absmax_shape = Shape::new(
            shape[0] / TensorOp::NF4_BLOCK_SIZE,
            shape[1],
            shape[2],
            shape[3],
        );

        let quant = vec![
            -1.0,
            -0.696_192_8,
            -0.525_073_05,
            -0.394_917_5,
            -0.284_441_38,
            -0.184_773_43,
            -0.091_050_036,
            0.0,
            0.079_580_3,
            0.160_930_2,
            0.246_112_3,
            0.337_915_24,
            0.440_709_83,
            0.562_617,
            0.722_956_84,
            1.0,
        ];
        let q = Box::new(context.tensor_from_data(Shape::new(quant.len(), 1, 1, 1), quant)?);

        let w = Box::new(context.tensor_init(matrix_shape));
        let m = Box::new(context.tensor_init(absmax_shape));

        let op = TensorOp::quantize_mat_nf4(matrix, &q, &m, &w)?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        context.queue.submit(Some(encoder.finish()));

        Ok(Matrix::NF4 { w, q, m })
    }
}
