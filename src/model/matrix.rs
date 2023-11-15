use half::f16;
use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor};

use crate::tensor::{
    ops::{TensorOp, TensorPass},
    shape::Shape,
    ReadWrite, TensorError, TensorGpu, TensorShape, TensorView,
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
}

impl Matrix {
    pub fn matmul_vec_op<'a>(
        &'a self,
        input: TensorView<'a, f32>,
        output: TensorView<'a, f32>,
    ) -> Result<TensorOp<'a>, TensorError> {
        match self {
            Matrix::Fp16(matrix) => TensorOp::matmul_vec_fp16(matrix, input, output),
            Matrix::Int8 { w, mx, rx, my, ry } => {
                TensorOp::matmul_vec_int8(w, mx, rx, my, ry, input, output)
            }
        }
    }

    pub fn matmul_mat_op<'a>(
        &'a self,
        buffer: &'a TensorGpu<f16, ReadWrite>,
        input: TensorView<'a, f32>,
        output: TensorView<'a, f32>,
    ) -> Result<TensorOp<'a>, TensorError> {
        match self {
            Matrix::Fp16(matrix) => Ok(TensorOp::List(vec![
                TensorOp::quantize_fp16(input.tensor, buffer)?,
                TensorOp::matmul_mat_fp16(
                    matrix.view(.., .., .., ..)?,
                    buffer.view(.., .., .., ..)?,
                    output,
                )?,
            ])),
            Matrix::Int8 { w, mx, rx, my, ry } => Ok(TensorOp::List(vec![
                TensorOp::quantize_fp16(input.tensor, buffer)?,
                TensorOp::matmul_mat_int8(
                    w.view(.., .., .., ..)?,
                    mx,
                    rx,
                    my,
                    ry,
                    buffer.view(.., .., .., ..)?,
                    output,
                )?,
            ])),
        }
    }

    pub fn quant_u8(matrix: TensorGpu<f16, ReadWrite>) -> Result<Self, TensorError> {
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

        let op = TensorOp::quantize_mat_int8(&matrix, &mx, &rx, &my, &ry, &w)?;

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
        matrix.destroy();

        Ok(Matrix::Int8 { w, mx, rx, my, ry })
    }
}
