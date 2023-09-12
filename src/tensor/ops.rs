use half::f16;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, CommandEncoder, ComputePass, ComputePipeline,
};

use super::{Kind, ReadWrite, Shape, TensorError, TensorExt, TensorGpu, TensorView};
use crate::num::Scalar;

pub trait TensorCommand<T: Scalar, K: Kind> {
    fn copy_tensor(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
    ) -> Result<(), TensorError>;

    fn copy_tensor_batch(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
        batch: usize,
    ) -> Result<(), TensorError>;
}

impl<T: Scalar, K: Kind> TensorCommand<T, K> for CommandEncoder {
    fn copy_tensor(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
    ) -> Result<(), TensorError> {
        destination.check_shape(source.shape())?;
        let size = destination.size() as u64;
        self.copy_buffer_to_buffer(&source.buffer, 0, &destination.buffer, 0, size);
        Ok(())
    }

    fn copy_tensor_batch(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
        batch: usize,
    ) -> Result<(), TensorError> {
        destination.check_shape(Shape::new(source.shape[0], source.shape[1], 1))?;
        if batch >= source.shape[2] {
            return Err(TensorError::BatchOutOfRange(batch, source.shape[2]));
        }
        let size = destination.size() as u64;
        let offset = (T::size() * source.shape[0] * source.shape[1] * batch) as u64;
        self.copy_buffer_to_buffer(&source.buffer, offset, &destination.buffer, 0, size);
        Ok(())
    }
}

pub trait TensorPass<'a> {
    fn execute_tensor_op(&mut self, op: &'a TensorOp);
}

impl<'b, 'a: 'b> TensorPass<'a> for ComputePass<'b> {
    fn execute_tensor_op(&mut self, op: &'a TensorOp) {
        self.set_pipeline(op.pipeline);
        op.bindings
            .iter()
            .enumerate()
            .for_each(|(index, bind_group)| self.set_bind_group(index as u32, bind_group, &[]));
        self.dispatch_workgroups(op.dispatch[0], op.dispatch[1], op.dispatch[2]);
    }
}

pub struct TensorOp<'a> {
    pub pipeline: &'a ComputePipeline,
    pub bindings: Vec<BindGroup>,
    pub dispatch: [u32; 3],
}

impl<'a> TensorOp<'a> {
    const BLOCK_SIZE: u32 = 128;

    #[inline]
    fn round(x: u32, div: u32) -> u32 {
        (x + div - 1) / div
    }

    #[inline]
    fn block_count(x: u32) -> u32 {
        Self::round(x, Self::BLOCK_SIZE)
    }

    /// Softmax operator applied on `x`.
    pub fn softmax(x: &'a TensorGpu<f32, ReadWrite>) -> Result<Self, TensorError> {
        let shape = x.shape();
        let context = &x.context;
        let pipeline = context.pipeline("softmax")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Layer norm applied on `x`, with weight `w` and bias `b`.
    pub fn layer_norm(
        w: &'a TensorGpu<f16, ReadWrite>,
        b: &'a TensorGpu<f16, ReadWrite>,
        x: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = x.shape();
        w.check_shape(Shape::new(shape[0], 1, 1))?;
        b.check_shape(Shape::new(shape[0], 1, 1))?;

        let context = &x.context;
        let pipeline = context.pipeline("layer_norm")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: w.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: b.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Fp32 matrix-vector multiplication.
    /// - `matrix` shape: `[C, R, 1]`.
    /// - `input` shape: `[C, T, B]`.
    /// - `output` shape: `[R, T, B]`.
    pub fn matmul_vec(
        matrix: &'a TensorGpu<f16, ReadWrite>,
        input: TensorView<'a, f32>,
        output: TensorView<'a, f32>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape();
        matrix.check_shape(Shape::new(input.shape()[0], shape[0], 1))?;
        input.check_shape(Shape::new(matrix.shape[0], shape[1], shape[2]))?;

        let context = &output.tensor.context;
        let pipeline = context.pipeline("matmul_vec")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [matrix.shape[1] as u32 / 4, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Int8 matrix-vector multiplication.
    /// - `matrix` shape: `[C, R, 1]`.
    /// - `mx` and `rx` shape: `[C, 1, 1]`.
    /// - `my` and `ry` shape: `[R, 1, 1]`.
    /// - `input` shape: `[C, T, B]`.
    /// - `output` shape: `[R, T, B]`.
    pub fn matmul_vec_int8(
        matrix: &'a TensorGpu<u8, ReadWrite>,
        mx: &'a TensorGpu<f32, ReadWrite>,
        rx: &'a TensorGpu<f32, ReadWrite>,
        my: &'a TensorGpu<f32, ReadWrite>,
        ry: &'a TensorGpu<f32, ReadWrite>,
        input: TensorView<'a, f32>,
        output: TensorView<'a, f32>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape();
        matrix.check_shape(Shape::new(input.shape()[0], shape[0], 1))?;
        input.check_shape(Shape::new(matrix.shape[0], shape[1], shape[2]))?;
        mx.check_shape(Shape::new(matrix.shape[0], 1, 1))?;
        rx.check_shape(Shape::new(matrix.shape[0], 1, 1))?;
        my.check_shape(Shape::new(matrix.shape[1], 1, 1))?;
        ry.check_shape(Shape::new(matrix.shape[1], 1, 1))?;

        let context = &matrix.context;
        let pipeline = context.pipeline("matmul_vec_int8")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: mx.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: rx.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: my.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: ry.binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [matrix.shape[1] as u32 / 4, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Fp32 matrix-matrix multiplication.
    /// - `matrix` shape: `[K, M, B]`.
    /// - `input` shape: `[K, N, B]`.
    /// - `output` shape: `[M, N, B]`.
    pub fn matmul_mat(
        xa: TensorView<'a, f16>,
        xb: TensorView<'a, f32>,
        output: TensorView<'a, f32>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape();
        xa.check_shape(Shape::new(xa.shape()[0], shape[0], shape[2]))?;
        xb.check_shape(Shape::new(xb.shape()[0], shape[1], shape[2]))?;

        let context = &output.tensor.context;
        let pipeline = context.pipeline("matmul_mat")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: xa.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: xb.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: xa.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: xb.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [
                Self::round(Self::round(shape[0] as u32, 4), 8),
                Self::round(Self::round(shape[1] as u32, 4), 8),
                shape[2] as u32,
            ],
        })
    }

    /// Fp16 matrix-matrix multiplication.
    /// - `matrix` shape: `[K, M, B]`.
    /// - `input` shape: `[K, N, B]`.
    /// - `output` shape: `[M, N, B]`.
    pub fn matmul_mat_fp16(
        xa: TensorView<'a, f16>,
        xb: TensorView<'a, f16>,
        output: TensorView<'a, f32>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape();
        xa.check_shape(Shape::new(xa.shape()[0], shape[0], shape[2]))?;
        xb.check_shape(Shape::new(xb.shape()[0], shape[1], shape[2]))?;

        let context = &output.tensor.context;
        let pipeline = context.pipeline("matmul_mat_fp16")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: xa.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: xb.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: xa.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: xb.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [
                Self::round(Self::round(shape[0] as u32, 4), 8),
                Self::round(Self::round(shape[1] as u32, 4), 8),
                shape[2] as u32,
            ],
        })
    }

    /// Add `input` onto `output`.
    pub fn add(
        input: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        input.check_shape(shape)?;

        let context = &output.context;
        let pipeline = context.pipeline("add")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn token_shift(
        cursors: &'a TensorGpu<u32, ReadWrite>,
        time_mix: &'a TensorGpu<f16, ReadWrite>,
        x: &'a TensorGpu<f32, ReadWrite>,
        sx: TensorView<f32>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        let max_batch = sx.shape()[2];
        cursors.check_shape(Shape::new(shape[1], 1, 1))?;
        time_mix.check_shape(Shape::new(shape[0], 1, 1))?;
        x.check_shape(shape)?;
        sx.check_shape(Shape::new(shape[0], 1, max_batch))
            .or(sx.check_shape(Shape::new(shape[0], 4, max_batch)))?;

        let context = &output.context;
        let pipeline = context.pipeline("token_shift")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: sx.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cursors.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: time_mix.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: sx.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [Self::block_count(shape[0] as u32 / 4), shape[1] as u32, 1],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn time_mix(
        stack: &'a TensorGpu<u32, ReadWrite>,
        time_decay: &'a TensorGpu<f32, ReadWrite>,
        time_first: &'a TensorGpu<f32, ReadWrite>,
        k: &'a TensorGpu<f32, ReadWrite>,
        v: &'a TensorGpu<f32, ReadWrite>,
        r: &'a TensorGpu<f32, ReadWrite>,
        x: &'a TensorGpu<f32, ReadWrite>,
        state: TensorView<f32>,
    ) -> Result<Self, TensorError> {
        let shape = x.shape;
        let max_batch = state.shape()[2];
        let num_batch = stack.shape[0];
        stack.check_shape(Shape::new(num_batch, 1, 1))?;
        k.check_shape(shape)?;
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        time_decay.check_shape(Shape::new(shape[0], 1, 1))?;
        time_first.check_shape(Shape::new(shape[0], 1, 1))?;
        state.check_shape(Shape::new(shape[0], 4, max_batch))?;

        let context = &x.context;
        let pipeline = context.pipeline("time_mix")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: state.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: stack.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: time_decay.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: time_first.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: k.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: v.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: r.binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: state.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [Self::block_count(shape[0] as u32 / 4), num_batch as u32, 1],
        })
    }

    pub fn squared_relu(x: &'a TensorGpu<f32, ReadWrite>) -> Result<Self, TensorError> {
        let shape = x.shape;
        let context = &x.context;
        let pipeline = context.pipeline("squared_relu")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn channel_mix(
        cursors: &'a TensorGpu<u32, ReadWrite>,
        r: &'a TensorGpu<f32, ReadWrite>,
        v: &'a TensorGpu<f32, ReadWrite>,
        x: &'a TensorGpu<f32, ReadWrite>,
        state: TensorView<'a, f32>,
    ) -> Result<Self, TensorError> {
        let shape = x.shape;
        let max_batch = state.shape()[2];
        cursors.check_shape(Shape::new(shape[1], 1, 1))?;
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        state.check_shape(Shape::new(shape[0], 1, max_batch))?;

        let context = &x.context;
        let pipeline = context.pipeline("channel_mix")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: state.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cursors.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: r.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: v.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: state.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [Self::block_count(shape[0] as u32 / 4), shape[1] as u32, 1],
        })
    }

    /// Copy the content of `input` into `output`, given an `offset`.
    pub fn blit(
        input: TensorView<'a, f32>,
        output: TensorView<'a, f32>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape();
        input.check_shape(shape)?;

        let context = &input.tensor.context;
        let pipeline = context.pipeline("blit")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn quantize_mat_int8(
        input: &'a TensorGpu<f16, ReadWrite>,
        mx: &'a TensorGpu<f32, ReadWrite>,
        rx: &'a TensorGpu<f32, ReadWrite>,
        my: &'a TensorGpu<f32, ReadWrite>,
        ry: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<u8, ReadWrite>,
    ) -> Result<Vec<Self>, TensorError> {
        let shape = output.shape;
        input.check_shape(shape)?;
        mx.check_shape(Shape::new(shape[0], 1, 1))?;
        rx.check_shape(Shape::new(shape[0], 1, 1))?;
        my.check_shape(Shape::new(shape[1], 1, 1))?;
        ry.check_shape(Shape::new(shape[1], 1, 1))?;

        let context = &output.context;
        let entries = &[
            BindGroupEntry {
                binding: 0,
                resource: output.meta_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: input.binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: mx.binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: rx.binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: my.binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: ry.binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: output.binding(),
            },
        ];
        let create_op = |name: &'static str, dispatch| -> Result<Self, TensorError> {
            let pipeline = context.pipeline(name)?;
            let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            })];
            Ok(Self {
                pipeline,
                bindings,
                dispatch,
            })
        };

        let my = create_op("quant_mat_int8_my", [1, shape[1] as u32, 1])?;
        let ry = create_op("quant_mat_int8_ry", [1, shape[1] as u32, 1])?;
        let mx = create_op("quant_mat_int8_mx", [1, shape[0] as u32 / 4, 1])?;
        let rx = create_op("quant_mat_int8_rx", [1, shape[0] as u32 / 4, 1])?;
        let quantize = create_op("quant_mat_int8", [shape[0] as u32 / 4, shape[1] as u32, 1])?;

        if shape[1] > shape[0] {
            Ok(vec![my, mx, rx, ry, quantize])
        } else {
            Ok(vec![mx, my, rx, ry, quantize])
        }
    }

    pub fn quantize_vec_fp16(
        input: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f16, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        input.check_shape(shape)?;

        let context = &output.context;
        let pipeline = context.pipeline("quant_vec_fp16")?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use half::f16;
    use itertools::Itertools;
    use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor, Features, PowerPreference};
    use wgpu_profiler::GpuProfiler;

    use super::{TensorOp, TensorPass};
    use crate::{
        context::{Context, ContextBuilder, Instance},
        tensor::{ops::TensorCommand, Shape, TensorCpu, TensorExt, TensorGpu, TensorInit},
    };

    fn is_approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= f32::max(f32::EPSILON, f32::max(a.abs(), b.abs()) * f32::EPSILON)
    }

    fn is_approx_eps(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= f32::max(eps, f32::max(a.abs(), b.abs()) * eps)
    }

    fn create_context() -> Result<Context, anyhow::Error> {
        let adapter = pollster::block_on(async {
            let instance = Instance::new();
            instance.adapter(PowerPreference::HighPerformance).await
        })?;
        let context = pollster::block_on(async {
            ContextBuilder::new(adapter)
                .with_default_pipelines()
                .with_quant_pipelines()
                .with_features(Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES)
                .build()
                .await
        })?;
        Ok(context)
    }

    #[test]
    fn test_copy() -> Result<(), anyhow::Error> {
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };

        let x = vec![0.0, 1.5, 2.0, -1.0];
        let shape = Shape::new(x.len(), 1, 1);

        let x_device: TensorGpu<_, _> = context.tensor_from_data(shape, x.clone())?;
        let x_map = context.init_tensor(x_device.shape());

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_tensor(&x_device, &x_map)?;
        context.queue.submit(Some(encoder.finish()));

        let x_host = TensorCpu::from(x_map);
        let x_host = Vec::from(x_host);

        assert_eq!(x, x_host);
        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<(), anyhow::Error> {
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };

        const C: usize = 1000;
        const T: usize = 3;
        const B: usize = 2;

        let x = [(); C * T * B]
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .to_vec();
        let shape = Shape::new(C, T, B);

        let x_dev: TensorGpu<_, _> = context.tensor_from_data(shape, x.clone())?;
        let x_map = context.init_tensor(x_dev.shape());

        let softmax = TensorOp::softmax(&x_dev)?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&softmax);
        drop(pass);

        encoder.copy_tensor(&x_dev, &x_map)?;
        context.queue.submit(Some(encoder.finish()));

        let x_host = TensorCpu::from(x_map);
        let x_host = Vec::from(x_host);

        let mut ans = vec![];
        for x in &x.into_iter().chunks(C) {
            let x = x.collect_vec().into_iter();
            let max = x.clone().reduce(f32::max).unwrap_or_default();
            let x = x.map(|x| (x - max).exp());
            let sum: f32 = x.clone().sum();
            let mut x: Vec<_> = x.map(|x| x / sum).collect();
            ans.append(&mut x);
        }

        for (index, (a, b)) in Iterator::zip(x_host.into_iter(), ans.into_iter()).enumerate() {
            assert!(
                is_approx(a, b),
                "Failed at index {index}, computed: {a} vs. answer: {b}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_layer_norm() -> Result<(), anyhow::Error> {
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };

        const C: usize = 1000;
        const T: usize = 3;
        const B: usize = 2;

        let x = [(); C * T * B]
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .to_vec();
        let w = [(); C]
            .map(|_| f16::from_f32(fastrand::f32() - 0.5))
            .repeat(T * B)
            .to_vec();
        let b = [(); C]
            .map(|_| f16::from_f32(fastrand::f32() - 0.5))
            .repeat(T * B)
            .to_vec();

        let shape = Shape::new(C, T, B);
        let x_dev = TensorGpu::from_data(&context, shape, &x)?;
        let x_map = context.init_tensor(shape);

        let shape = Shape::new(C, 1, 1);
        let w_dev = TensorGpu::from_data(&context, shape, &w[..1000])?;
        let b_dev = TensorGpu::from_data(&context, shape, &b[..1000])?;

        let layer_norm = TensorOp::layer_norm(&w_dev, &b_dev, &x_dev)?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&layer_norm);
        drop(pass);

        encoder.copy_tensor(&x_dev, &x_map)?;
        context.queue.submit(Some(encoder.finish()));

        let x_host = TensorCpu::from(x_map);
        let x_host = Vec::from(x_host);

        let mut ans = vec![];
        for chunk in &x
            .into_iter()
            .zip(w.into_iter())
            .zip(b.into_iter())
            .chunks(C)
        {
            let chunk = chunk.collect_vec();
            let x = chunk.iter().map(|((x, _), _)| x).copied();
            let sum: f32 = x.clone().sum();
            let squared_sum: f32 = x.clone().map(|x| x.powi(2)).sum();

            let mean = sum / C as f32;
            let deviation = ((squared_sum / C as f32) - mean.powi(2)).sqrt();

            let mut x: Vec<_> = chunk
                .into_iter()
                .map(|((x, w), b)| (x - mean) / deviation * w.to_f32() + b.to_f32())
                .collect();
            ans.append(&mut x);
        }

        for (index, (a, b)) in Iterator::zip(x_host.into_iter(), ans.into_iter()).enumerate() {
            assert!(
                is_approx_eps(a, b, 1.0e-3),
                "Failed at index {index}, computed: {a} vs. answer: {b}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_matmul() -> Result<(), anyhow::Error> {
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
        let mut profiler = GpuProfiler::new(&context.adapter, &context.device, &context.queue, 1);

        const C: usize = 1024;
        const R: usize = 768;
        const T: usize = 255;

        let matrix = vec![(); C * R]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .map(f16::from_f32)
            .collect_vec();
        let input_f32 = vec![(); C * T]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .collect_vec();
        let input_f16 = input_f32.iter().copied().map(f16::from_f32).collect_vec();

        let matrix_dev = context.tensor_from_data(Shape::new(C, R, 1), matrix.clone())?;
        let input_f32_dev = TensorGpu::from_data(&context, Shape::new(C, T, 1), input_f32.clone())?;
        // let input_f16_dev = TensorGpu::from_data(&context, Shape::new(C, T, 1), input_f16.clone())?;
        let input_f16_dev = TensorGpu::init(&context, input_f32_dev.shape());
        let output_dev = TensorGpu::init(&context, Shape::new(R, T, 3));
        let output_map = TensorGpu::init(&context, output_dev.shape());

        let quant_input = TensorOp::quantize_vec_fp16(&input_f32_dev, &input_f16_dev)?;
        let matmul_vec = TensorOp::matmul_vec(
            &matrix_dev,
            input_f32_dev.view((.., .., ..))?,
            output_dev.view((.., .., 0))?,
        )?;
        let matmul_mat = TensorOp::matmul_mat(
            matrix_dev.view((.., .., ..))?,
            input_f32_dev.view((.., .., ..))?,
            output_dev.view((.., .., 1))?,
        )?;
        let matmul_mat_fp16 = TensorOp::matmul_mat_fp16(
            matrix_dev.view((.., .., ..))?,
            input_f16_dev.view((.., .., ..))?,
            output_dev.view((.., .., 2))?,
        )?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        {
            let mut pass = wgpu_profiler::scope::Scope::start(
                "quant_fp16",
                &mut profiler,
                &mut pass,
                &context.device,
            );
            pass.execute_tensor_op(&quant_input);
        }
        {
            let mut pass = wgpu_profiler::scope::Scope::start(
                "matmul_vec",
                &mut profiler,
                &mut pass,
                &context.device,
            );
            pass.execute_tensor_op(&matmul_vec);
        }
        {
            let mut pass = wgpu_profiler::scope::Scope::start(
                "matmul_mat",
                &mut profiler,
                &mut pass,
                &context.device,
            );
            pass.execute_tensor_op(&matmul_mat);
        }
        {
            let mut pass = wgpu_profiler::scope::Scope::start(
                "matmul_mat_fp16",
                &mut profiler,
                &mut pass,
                &context.device,
            );
            pass.execute_tensor_op(&matmul_mat_fp16);
        }
        drop(pass);

        profiler.resolve_queries(&mut encoder);

        encoder.copy_tensor(&output_dev, &output_map)?;
        context.queue.submit(Some(encoder.finish()));

        let output_host = TensorCpu::from(output_map);
        let output_host = Vec::from(output_host);

        profiler.end_frame().unwrap();
        context.device.poll(wgpu::MaintainBase::Wait);

        if let Some(results) = profiler.process_finished_frame() {
            wgpu_profiler::chrometrace::write_chrometrace(
                std::path::Path::new(&format!("./trace/matmul_{T}.json")),
                &results,
            )
            .expect("failed to write trace");
        }

        let mut ans = vec![0.0; output_host.len()];
        for token in 0..T {
            for line in 0..R {
                let matrix = &matrix[line * C..(line + 1) * C];
                let input = &input_f32[token * C..(token + 1) * C];
                let product = matrix
                    .iter()
                    .zip(input.iter())
                    .fold(0.0f32, |acc, x| acc + x.0.to_f32() * *x.1);
                ans[(0 * T + token) * R + line] = product;
                ans[(1 * T + token) * R + line] = product;

                let input = &input_f16[token * C..(token + 1) * C];
                let product = matrix
                    .iter()
                    .zip(input.iter())
                    .fold(0.0f32, |acc, x| acc + x.0.to_f32() * x.1.to_f32());
                ans[(2 * T + token) * R + line] = product;
            }
        }

        for (index, (a, b)) in Iterator::zip(output_host.into_iter(), ans.into_iter()).enumerate() {
            assert!(
                is_approx_eps(a, b, 1.0e-3),
                "Failed at index {index}, computed: {a} vs. answer: {b}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_blit() -> Result<(), anyhow::Error> {
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };

        let output = vec![0.0; 24];
        let output = TensorGpu::from_data(&context, Shape::new(4, 3, 2), output)?;

        let map = TensorGpu::init(&context, output.shape());
        let mut ops = vec![];

        let input = (0..8).map(|x| x as f32).collect_vec();
        let input = TensorGpu::from_data(&context, Shape::new(4, 1, 2), input)?;
        ops.push(TensorOp::blit(
            input.view((.., .., ..))?,
            output.view((.., 1, ..))?,
        )?);

        let input = (8..12).map(|x| x as f32).collect_vec();
        let input = TensorGpu::from_data(&context, Shape::new(4, 1, 1), input)?;
        let input = input.view((.., .., ..))?;
        ops.push(TensorOp::blit(input, output.view((.., 2.., 1..2))?)?);

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        ops.iter().for_each(|op| pass.execute_tensor_op(op));
        drop(pass);

        encoder.copy_tensor(&output, &map)?;
        context.queue.submit(Some(encoder.finish()));

        let output_host = TensorCpu::from(map);
        let output_host = Vec::from(output_host);

        assert_eq!(
            output_host,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0
            ]
        );

        Ok(())
    }
}
