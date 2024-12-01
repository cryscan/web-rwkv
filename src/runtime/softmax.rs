use crate::{
    context::Context,
    num::Float,
    tensor::{ops::TensorOp, TensorCpu, TensorError, TensorGpu, TensorInto},
};

pub async fn softmax_one<T: Float>(
    context: &Context,
    input: TensorCpu<T>,
) -> Result<TensorCpu<T>, TensorError> {
    if input.size() == 0 {
        return Ok(input);
    }

    let tensor: TensorGpu<_, _> = input.to(context);
    let op = TensorOp::softmax(&tensor)?;
    context.queue.submit(context.encode(&op));

    let output = tensor.back().await;
    Ok(output)
}

pub async fn softmax<T: Float>(
    context: &Context,
    input: Vec<TensorCpu<T>>,
) -> Result<Vec<TensorCpu<T>>, TensorError> {
    let mut tensors = Vec::with_capacity(input.len());
    let mut ops = Vec::with_capacity(input.len());

    for input in input.into_iter() {
        let tensor: TensorGpu<_, _> = input.to(context);
        if tensor.size() > 0 {
            ops.push(TensorOp::softmax(&tensor)?);
        }
        tensors.push(tensor);
    }
    context.queue.submit(context.encode(&TensorOp::List(ops)));

    let mut output = Vec::with_capacity(tensors.len());
    for tensor in tensors.into_iter() {
        output.push(tensor.back().await);
    }
    Ok(output)
}
