use web_rwkv::{
    context::Macros,
    num::Float,
    tensor::{ops::TensorOp, TensorError, TensorGpuView, TensorShape},
    wgpu::{BindGroupDescriptor, BindGroupEntry},
};

pub trait TensorOpExt: Sized {
    /// Multiply `input` to exponential of `output`.
    /// - `input` shape: `[C, 1, B]` or `[C, T, B]`.
    /// - `output` shape: `[C, T, B]`.
    fn mul_exp<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
    ) -> Result<Self, TensorError>;
}

impl TensorOpExt for TensorOp {
    fn mul_exp<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let shape = {
            let [index, token, batch, _] = output.shape().into();
            input
                .check_shape([index, 1, batch, 1])
                .or(input.check_shape([index, token, batch, 1]))?;
            output.check_shape([index, token, batch, 1])?;
            output.shape()
        };

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "mul_exp",
            include_str!("mul_exp.wgsl"),
            "mul_exp",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
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

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }
}
