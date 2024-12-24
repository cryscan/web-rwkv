use half::f16;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::DeserializeSeed;

use super::{ops::Activation, TensorCpu, TensorInit, TensorInto};
use crate::{
    context::Context,
    num::Float,
    tensor::{
        kind::{ReadWrite, Uniform},
        ops::TensorOp,
        serialization::Seed,
        shape::Shape,
        TensorError, TensorGpu, TensorGpuView, TensorShape,
    },
};

#[derive(Debug, Clone)]
pub struct Nf4Quant(pub TensorCpu<f32>);

impl Default for Nf4Quant {
    fn default() -> Self {
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
        let shape = Shape::new(quant.len(), 1, 1, 1);
        Self(TensorCpu::from_data(shape, quant).unwrap())
    }
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub enum Matrix {
    Fp16(TensorGpu<f16, ReadWrite>),
    Int8 {
        w: TensorGpu<u8, ReadWrite>,
        m: TensorGpu<f16, ReadWrite>,
    },
    NF4 {
        q: TensorGpu<f32, Uniform>,
        w: TensorGpu<u8, ReadWrite>,
        m: TensorGpu<f16, ReadWrite>,
    },
}

impl Matrix {
    pub fn matmul_vec_op<'a, 'b, F0: Float, F1: Float>(
        &self,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
    ) -> Result<TensorOp, TensorError> {
        match self {
            Matrix::Fp16(matrix) => TensorOp::matmul_vec_fp16(matrix, input, output, act),
            Matrix::Int8 { w, m } => TensorOp::matmul_vec_int8(w, m, input, output, act),
            Matrix::NF4 { w, q, m } => TensorOp::matmul_vec_nf4(w, q, m, input, output, act),
        }
    }

    pub fn matmul_mat_op<'a, 'b, F0: Float, F1: Float>(
        &self,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
    ) -> Result<TensorOp, TensorError> {
        match self {
            Matrix::Fp16(matrix) => TensorOp::matmul_mat_fp16(matrix, input, output, act),
            Matrix::Int8 { w, m } => TensorOp::matmul_mat_int8(w, m, input, output, act),
            Matrix::NF4 { w, q, m } => TensorOp::matmul_mat_nf4(w, q, m, input, output, act),
        }
    }

    pub fn matmul_op<'a, 'b, F0: Float, F1: Float>(
        &self,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
        turbo: bool,
    ) -> Result<TensorOp, TensorError> {
        match turbo {
            true => self.matmul_mat_op(input, output, act),
            false => self.matmul_vec_op(input, output, act),
        }
    }

    pub fn quant_u8(matrix: &TensorGpu<f16, ReadWrite>) -> Result<Self, TensorError> {
        let context = matrix.context();
        let shape = matrix.shape();

        let w = context.tensor_init(shape);
        let m = context.tensor_init(Shape::new(
            (shape.len() << 1).div_ceil(TensorOp::INT8_BLOCK_SIZE as usize),
            1,
            1,
            1,
        ));

        let op = TensorOp::quantize_mat_int8(matrix, &m, &w)?;
        context.queue.submit(context.encode(&op));

        Ok(Matrix::Int8 { w, m })
    }

    pub fn quant_nf4(matrix: &TensorGpu<f16, ReadWrite>) -> Result<Self, TensorError> {
        let context = matrix.context();
        let shape = matrix.shape();

        let matrix_shape = Shape::new(shape[0] / 2, shape[1], shape[2], shape[3]);
        let absmax_shape = Shape::new(
            shape.len().div_ceil(TensorOp::NF4_BLOCK_SIZE as usize),
            1,
            1,
            1,
        );

        let q = Nf4Quant::default().0.to(context);
        let w = context.tensor_init(matrix_shape);
        let m = context.tensor_init(absmax_shape);

        let op = TensorOp::quantize_mat_nf4(matrix, &q, &m, &w)?;
        context.queue.submit(context.encode(&op));

        Ok(Matrix::NF4 { w, q, m })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixStatistics {
    /// Quantile values: `min`, `q_005`, `q_25`, `q_50`, `q_75`, `q_995`, `max`.
    pub quantile: [f32; 7],
}

impl<F: Float> TensorCpu<F> {
    pub fn statistics(&self) -> MatrixStatistics {
        let values: Vec<f32> = self
            .iter()
            .map(|x| x.hom())
            .sorted_unstable_by(|x: &f32, y: &f32| x.total_cmp(y))
            .collect();
        assert!(values.len() > 2);
        let p0 = 0;
        let p4 = values.len() - 1;
        let p2 = (p0 + p4) / 2;
        let p1 = (p0 + p2) / 2;
        let p3 = (p2 + p4) / 2;
        let p_005 = ((p4 as f32) * 0.005) as usize;
        let p_995 = ((p4 as f32) * 0.995) as usize;
        let quantile = [p0, p1, p2, p3, p4, p_005, p_995].map(|p| values[p]);
        MatrixStatistics { quantile }
    }
}
