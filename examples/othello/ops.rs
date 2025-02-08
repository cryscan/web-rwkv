use web_rwkv::{
    context::{BindGroupBuilder, Macros, PipelineKey},
    num::Float,
    tensor::{ops::TensorOp, TensorError, TensorGpuView, TensorShape},
};

pub trait TensorOpExt: Sized {
    fn mul_w<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
    ) -> Result<Self, TensorError>;
}

impl TensorOpExt for TensorOp {
    fn mul_w<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = output.context();
        let shape = {
            let [index, token, batch, _] = output.shape().into();
            input
                .check_shape([index, 1, batch, 1])
                .or(input.check_shape([index, token, batch, 1]))?;
            output.check_shape([index, token, batch, 1])?;
            output.shape()
        };

        let key = PipelineKey::new(
            "mul_w",
            "mul_w",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("mul_w.wgsl"),
            &[
                input.meta_layout(0),
                output.meta_layout(1),
                input.layout(2, true),
                output.layout(3, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .bind_meta(0, &input)
            .bind_meta(1, &output)
            .bind(2, &input)
            .bind(3, &output)
            .build()];

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
