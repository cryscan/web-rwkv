use std::{future::Future, sync::Arc};

use anyhow::Result;
use itertools::Itertools;

use super::{ModelBase, ModelInfo, ModelOutput};
use crate::{
    context::Context,
    tensor::{
        kind::{ReadBack, ReadWrite},
        ops::{TensorCommand, TensorOp, TensorPass},
        shape::Shape,
        TensorCpu, TensorError, TensorGpu, TensorInit, TensorShape,
    },
};

#[derive(Debug)]
pub struct Softmax {
    pub buffer: TensorGpu<f32, ReadWrite>,
    pub map: TensorGpu<f32, ReadBack>,
}

impl Softmax {
    pub fn new(context: &Context, info: &ModelInfo, num_batch: usize) -> Self {
        let shape = Shape::new(info.num_vocab, 1, num_batch, 1);
        Self {
            buffer: context.tensor_init(shape),
            map: context.tensor_init(shape),
        }
    }
}

pub(crate) trait ModelSoftmaxInternal: ModelBase {
    fn checkout_softmax(&self, num_batch: usize) -> Arc<Softmax>;
}

pub trait ModelSoftmax {
    /// Softmax of the input tensors.
    fn softmax(
        &self,
        input: Vec<ModelOutput>,
    ) -> impl Future<Output = Result<Vec<ModelOutput>, TensorError>>;
}

impl<Model: ModelSoftmaxInternal> ModelSoftmax for Model {
    async fn softmax(&self, input: Vec<ModelOutput>) -> Result<Vec<ModelOutput>, TensorError> {
        let context = self.context();
        let info = self.info();

        if input.iter().all(ModelOutput::is_none) {
            return Ok(input);
        }

        let mut redirect = vec![0..0; input.len()];
        let input: Vec<_> = input
            .into_iter()
            .enumerate()
            .filter_map(|(batch, data)| match data {
                ModelOutput::None => None,
                ModelOutput::Last(data) => Some((batch, vec![data])),
                ModelOutput::Full(data) => Some((batch, data)),
            })
            .map(|(batch, data)| {
                let shape = Shape::new(info.num_vocab, 1, data.len(), 1);
                TensorCpu::from_data(context, shape, data.concat()).map(|tensor| (batch, tensor))
            })
            .try_collect()?;
        let input = TensorCpu::stack(
            input
                .into_iter()
                .fold((0, vec![]), |(index, mut tensors), (batch, tensor)| {
                    let len = tensor.shape()[2];
                    redirect[batch] = index..index + len;
                    tensors.push(tensor);
                    (index + len, tensors)
                })
                .1,
        )?;

        let num_batch = input.shape()[2];
        let softmax = self.checkout_softmax(num_batch);
        softmax.buffer.load(&input)?;

        let op = TensorOp::softmax(&softmax.buffer)?;

        let mut encoder = self
            .context()
            .device
            .create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        encoder.copy_tensor(&softmax.buffer, &softmax.map)?;
        self.context().queue.submit(Some(encoder.finish()));

        let output = softmax.map.clone().back_async().await;
        Ok(redirect
            .into_iter()
            .map(|r| match r.len() {
                0 => ModelOutput::None,
                1 => ModelOutput::Last(output.slice(.., .., r.start, ..).unwrap().to_vec()),
                _ => ModelOutput::Full(
                    r.map(|index| output.slice(.., .., index, ..).unwrap().to_vec())
                        .collect(),
                ),
            })
            .collect())
    }
}
