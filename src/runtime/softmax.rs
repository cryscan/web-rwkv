use crate::{
    context::Context,
    num::Float,
    tensor::{
        ops::{TensorOp, TensorPass},
        TensorCpu, TensorError, TensorGpu, TensorInto,
    },
};

pub async fn softmax_one<T: Float>(
    context: &Context,
    input: TensorCpu<T>,
) -> Result<TensorCpu<T>, TensorError> {
    if input.size() == 0 {
        return Ok(input);
    }

    let tensor: TensorGpu<_, _> = input.transfer_into(context);
    let op = TensorOp::softmax(&tensor)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.execute_tensor_op(&op);
    drop(pass);
    context.queue.submit(Some(encoder.finish()));

    let output = tensor.back().await;
    Ok(output)
}

pub async fn softmax<T: Float>(
    context: &Context,
    input: Vec<TensorCpu<T>>,
) -> Result<Vec<TensorCpu<T>>, TensorError> {
    let mut tensors = Vec::with_capacity(input.len());
    let mut ops = Vec::with_capacity(input.len());

    let mut encoder = context.device.create_command_encoder(&Default::default());
    for input in input.into_iter() {
        let tensor: TensorGpu<_, _> = input.transfer_into(context);
        if tensor.size() > 0 {
            ops.push(TensorOp::softmax(&tensor)?);
        }
        tensors.push(tensor);
    }

    let ops = TensorOp::List(ops);
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.execute_tensor_op(&ops);
    drop(pass);
    context.queue.submit(Some(encoder.finish()));

    let mut output = Vec::with_capacity(tensors.len());
    for tensor in tensors.into_iter() {
        output.push(tensor.back().await);
    }
    Ok(output)
}
