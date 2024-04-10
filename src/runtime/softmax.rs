use crate::{
    context::Context,
    num::Float,
    tensor::{
        ops::{TensorOp, TensorPass},
        TensorCpu, TensorError, TensorGpu, TensorInitContext, TensorShape,
    },
};

pub async fn softmax<T: Float>(
    context: &Context,
    input: &TensorCpu<T>,
) -> Result<TensorCpu<T>, TensorError> {
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
