use crate::{
    context::Context,
    num::Float,
    tensor::{
        ops::{TensorOp, TensorPass},
        TensorCpu, TensorError, TensorGpu, TensorInitContext, TensorShape,
    },
};

pub async fn softmax<'a, T: Float>(
    context: &Context,
    input: &TensorCpu<'a, T>,
) -> Result<TensorCpu<'static, T>, TensorError> {
    let tensor = TensorGpu::init(context, input.shape());
    tensor.load(input)?;
    let op = TensorOp::softmax(&tensor)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.execute_tensor_op(&op);
    drop(pass);

    let output = tensor.back().await;
    Ok(output)
}
