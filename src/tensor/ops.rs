use half::f16;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BufferAddress, CommandEncoder, ComputePass,
    ComputePipeline, Queue,
};

use super::{Kind, ReadWrite, Shape, TensorCpu, TensorError, TensorGpu, Uniform};
use crate::num::Scalar;

pub trait TensorCommand<T: Scalar, K: Kind> {
    fn copy_tensor(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
    ) -> Result<(), TensorError>;
}

impl<T: Scalar, K: Kind> TensorCommand<T, K> for CommandEncoder {
    fn copy_tensor(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
    ) -> Result<(), TensorError> {
        source.check_shape(destination.shape)?;
        let size = source.size() as BufferAddress;
        self.copy_buffer_to_buffer(
            &source.buffer,
            source.offset,
            &destination.buffer,
            destination.offset,
            size,
        );
        Ok(())
    }
}

pub trait TensorQueue<T: Scalar, K: Kind> {
    fn write_tensor(
        &mut self,
        host: &TensorCpu<T, K>,
        device: &TensorGpu<T, K>,
    ) -> Result<(), TensorError>;
}

impl<T: Scalar, K: Kind> TensorQueue<T, K> for Queue {
    fn write_tensor(
        &mut self,
        host: &TensorCpu<T, K>,
        device: &TensorGpu<T, K>,
    ) -> Result<(), TensorError> {
        host.check_shape(device.shape)?;
        self.write_buffer(
            &device.buffer,
            device.offset,
            bytemuck::cast_slice(&host.data),
        );
        Ok(())
    }
}

pub trait TensorPass<'a> {
    fn execute_tensor_op(&mut self, op: &'a TensorOp);
}

impl<'a, 'b> TensorPass<'a> for ComputePass<'b>
where
    'a: 'b,
{
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

    fn block_count(x: u32) -> u32 {
        (x + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE
    }

    /// Softmax operator applied on `x`.
    pub fn softmax(x: &'a TensorGpu<f32, ReadWrite>) -> Result<Self, TensorError> {
        let shape = x.shape;
        let context = x.context;
        let pipeline = context
            .pipelines
            .get("softmax")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.shape_binding(),
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
        let shape = x.shape;
        w.check_shape(Shape::new(shape[0], 1, 1))?;
        b.check_shape(Shape::new(shape[0], 1, 1))?;

        let context = x.context;
        let pipeline = context
            .pipelines
            .get("layer_norm")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.shape_binding(),
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

    /// Fp16 matrix multiplication.
    /// - `matrix` shape: `[C, R, 1]`.
    /// - `input` shape: `[C, T, B]`.
    /// - `output` shape: `[R, T, B]`.
    pub fn matmul(
        matrix: &'a TensorGpu<f16, ReadWrite>,
        input: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        matrix.check_shape(Shape::new(input.shape[0], shape[0], 1))?;
        input.check_shape(Shape::new(matrix.shape[0], shape[1], shape[2]))?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("matmul")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 4,
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

    /// Int8 matrix multiplication.
    /// - `matrix` shape: `[C, R, 1]`.
    /// - `mx` and `rx` shape: `[C, 1, 1]`.
    /// - `my` and `ry` shape: `[R, 1, 1]`.
    /// - `input` shape: `[C, T, B]`.
    /// - `output` shape: `[R, T, B]`.
    pub fn matmul_int8(
        matrix: &'a TensorGpu<u8, ReadWrite>,
        mx: &'a TensorGpu<f16, ReadWrite>,
        rx: &'a TensorGpu<f16, ReadWrite>,
        my: &'a TensorGpu<f16, ReadWrite>,
        ry: &'a TensorGpu<f16, ReadWrite>,
        input: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        matrix.check_shape(Shape::new(input.shape[0], shape[0], 1))?;
        input.check_shape(Shape::new(matrix.shape[0], shape[1], shape[2]))?;
        mx.check_shape(Shape::new(matrix.shape[0], 1, 1))?;
        rx.check_shape(Shape::new(matrix.shape[0], 1, 1))?;
        my.check_shape(Shape::new(matrix.shape[1], 1, 1))?;
        ry.check_shape(Shape::new(matrix.shape[1], 1, 1))?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("matmul_int8")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: mx.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: rx.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: my.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: ry.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 8,
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

    /// Add `input` onto `output`.
    pub fn add(
        input: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        input.check_shape(shape)?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("add")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.shape_binding(),
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
        time_mix: &'a TensorGpu<f16, ReadWrite>,
        x: &'a TensorGpu<f32, ReadWrite>,
        sx: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        x.check_shape(shape)?;
        time_mix.check_shape(Shape::new(shape[0], 1, 1))?;
        sx.check_shape(Shape::new(shape[0], 1, shape[2]))?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("token_shift")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: time_mix.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: sx.binding(),
                },
                BindGroupEntry {
                    binding: 4,
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

    #[allow(clippy::too_many_arguments)]
    pub fn token_mix(
        mask: &'a TensorGpu<u32, Uniform>,
        time_decay: &'a TensorGpu<f32, ReadWrite>,
        time_first: &'a TensorGpu<f32, ReadWrite>,
        k: &'a TensorGpu<f32, ReadWrite>,
        v: &'a TensorGpu<f32, ReadWrite>,
        r: &'a TensorGpu<f32, ReadWrite>,
        x: &'a TensorGpu<f32, ReadWrite>,
        state: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = x.shape;
        mask.check_shape(Shape::new(1, 1, 1))?;
        k.check_shape(shape)?;
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        time_decay.check_shape(Shape::new(shape[0], 1, 1))?;
        time_first.check_shape(Shape::new(shape[0], 1, 1))?;
        state.check_shape(Shape::new(shape[0], 4, shape[2]))?;

        let context = x.context;
        let pipeline = context
            .pipelines
            .get("token_mix")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: mask.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: time_decay.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: time_first.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: k.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: v.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: r.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: state.binding(),
                },
            ],
        })];

        Ok(Self {
            pipeline,
            bindings,
            dispatch: [Self::block_count(shape[0] as u32 / 4), 1, shape[2] as u32],
        })
    }

    pub fn squared_relu(x: &'a TensorGpu<f32, ReadWrite>) -> Result<Self, TensorError> {
        let shape = x.shape;
        let context = x.context;
        let pipeline = context
            .pipelines
            .get("squared_relu")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.shape_binding(),
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
        mask: &'a TensorGpu<u32, Uniform>,
        r: &'a TensorGpu<f32, ReadWrite>,
        v: &'a TensorGpu<f32, ReadWrite>,
        x: &'a TensorGpu<f32, ReadWrite>,
        state: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = x.shape;
        mask.check_shape(Shape::new(1, 1, 1))?;
        x.check_shape(shape)?;
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        state.check_shape(Shape::new(shape[0], 1, shape[2]))?;

        let context = x.context;
        let pipeline = context
            .pipelines
            .get("channel_mix")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: mask.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: r.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: v.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: x.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: state.binding(),
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

    /// Copy the content of `input` into `output`, given an `offset`.
    pub fn blit(
        offset: &'a TensorGpu<u32, Uniform>,
        input: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f32, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = input.shape;
        offset.check_shape(Shape::new(4, 1, 1))?;
        output.check_shape_with(shape, |output, input| {
            output
                .partial_cmp(&input)
                .is_some_and(|x| x != std::cmp::Ordering::Less)
        })?;

        let context = input.context;
        let pipeline = context
            .pipelines
            .get("blit")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.shape_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.shape_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: offset.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 4,
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
    ) -> Result<[Self; 5], TensorError> {
        let shape = output.shape;
        input.check_shape(shape)?;
        mx.check_shape(Shape::new(shape[0], 1, 1))?;
        rx.check_shape(Shape::new(shape[0], 1, 1))?;
        my.check_shape(Shape::new(shape[1], 1, 1))?;
        ry.check_shape(Shape::new(shape[1], 1, 1))?;

        let context = output.context;
        let entries = &[
            BindGroupEntry {
                binding: 0,
                resource: output.shape_binding(),
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
        let create_op = |name: &str, dispatch| -> Result<Self, TensorError> {
            let pipeline = context
                .pipelines
                .get(name)
                .ok_or(TensorError::PipelineError)?;
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
            Ok([my, mx, rx, ry, quantize])
        } else {
            Ok([mx, my, rx, ry, quantize])
        }
    }

    pub fn quantize_vec_fp16(
        input: &'a TensorGpu<f32, ReadWrite>,
        output: &'a TensorGpu<f16, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape;
        input.check_shape(shape)?;

        let context = output.context;
        let pipeline = context
            .pipelines
            .get("quant_vec_fp16")
            .ok_or(TensorError::PipelineError)?;
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.shape_binding(),
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
    use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor, PowerPreference};

    use super::{TensorOp, TensorPass};
    use crate::{
        context::{Context, ContextBuilder, Instance},
        tensor::{Shape, TensorCommand, TensorCpu, TensorExt, TensorGpu},
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
                .build()
                .await
        })?;
        Ok(context)
    }

    #[test]
    fn test_copy() -> Result<(), anyhow::Error> {
        let context = create_context()?;

        let x = vec![0.0, 1.5, 2.0, -1.0];
        let shape = Shape::new(x.len(), 1, 1);

        let x_device: TensorGpu<_, _> = context.tensor_from_data(None, shape, x.clone())?;
        let x_map = context.tensor_init(None, x_device.shape());

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
        let context = create_context()?;

        let x = [(); 6000].map(|_| 10.0 * (fastrand::f32() - 0.5)).to_vec();
        let shape = Shape::new(x.len() / 6, 3, 2);

        let x_dev: TensorGpu<_, _> = context.tensor_from_data(None, shape, x.clone())?;
        let x_map = context.tensor_init(None, x_dev.shape());

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
        for x in &x.into_iter().chunks(1000) {
            let x: Vec<_> = x.collect();
            let x = x.into_iter();
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
    fn test_matmul() -> Result<(), anyhow::Error> {
        let context = create_context()?;

        const C: usize = 1024;
        const R: usize = 768;
        const T: usize = 7;
        const B: usize = 3;

        let matrix: Vec<_> = vec![(); C * R]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .map(f16::from_f32)
            .collect();
        let input: Vec<_> = vec![(); C * T * B]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .collect();

        let matrix_dev = TensorGpu::from_data(
            &context,
            Some("matrix"),
            Shape::new(C, R, 1),
            matrix.clone(),
        )?;
        let input_dev =
            TensorGpu::from_data(&context, Some("input"), Shape::new(C, T, B), input.clone())?;
        let output_dev = TensorGpu::init(&context, None, Shape::new(R, T, B));
        let output_map = TensorGpu::init(&context, None, output_dev.shape());

        let matmul = TensorOp::matmul(&matrix_dev, &input_dev, &output_dev)?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&matmul);
        drop(pass);

        encoder.copy_tensor(&output_dev, &output_map)?;
        context.queue.submit(Some(encoder.finish()));

        let output_host = TensorCpu::from(output_map);
        let output_host = Vec::from(output_host);

        let mut ans = vec![0.0; output_host.len()];
        for batch in 0..B {
            for token in 0..T {
                for line in 0..R {
                    let matrix = &matrix[line * C..(line + 1) * C];
                    let input = &input[(batch * T + token) * C..(batch * T + token + 1) * C];
                    let product = matrix
                        .iter()
                        .map(|x| (*x).to_f32())
                        .zip(input.iter())
                        .fold(0.0f32, |acc, x| acc + x.0 * *x.1);
                    ans[(batch * T + token) * R + line] = product;
                }
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
        let context = create_context()?;

        let output = vec![0.0; 24];
        let output = TensorGpu::from_data(&context, None, Shape::new(4, 3, 2), output)?;

        let map = TensorGpu::init(&context, None, output.shape());

        let offset_shape = Shape::new(4, 1, 1);

        let input: Vec<_> = (0..8).map(|x| x as f32).collect();
        let input = TensorGpu::from_data(&context, None, Shape::new(4, 1, 2), input)?;
        let offset = TensorGpu::from_data(&context, None, offset_shape, vec![0, 1, 0, 0])?;
        let blit_1 = TensorOp::blit(&offset, &input, &output)?;

        let input: Vec<_> = (8..12).map(|x| x as f32).collect();
        let input = TensorGpu::from_data(&context, None, Shape::new(4, 1, 1), input)?;
        let offset = TensorGpu::from_data(&context, None, offset_shape, vec![0, 2, 1, 0])?;
        let blit_2 = TensorOp::blit(&offset, &input, &output)?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&blit_1);
        pass.execute_tensor_op(&blit_2);
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
