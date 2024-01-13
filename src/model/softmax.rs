use std::{future::Future, sync::Arc};

use anyhow::Result;
use itertools::Itertools;

use super::{ModelBase, ModelInfo};
use crate::{
    context::Context,
    tensor::{
        kind::{ReadBack, ReadWrite},
        ops::{TensorCommand, TensorOp, TensorPass},
        shape::Shape,
        TensorCpu, TensorGpu, TensorInit, TensorShape,
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

pub(crate) trait ModelSoftmaxInternal: ModelBase + Sync {
    fn request_softmax(&self, num_batch: usize) -> Arc<Softmax>;
}

pub trait ModelSoftmax {
    /// Softmax of the input tensors.
    fn softmax(
        &self,
        input: Vec<Option<Vec<f32>>>,
    ) -> impl Future<Output = Result<Vec<Option<Vec<f32>>>>> + Send;
}

impl<Model: ModelSoftmaxInternal> ModelSoftmax for Model {
    async fn softmax(&self, input: Vec<Option<Vec<f32>>>) -> Result<Vec<Option<Vec<f32>>>> {
        let max_batch = input.len();
        let context = self.context();
        let info = self.info();

        let mut redirect = vec![None; max_batch];
        let input: Vec<_> = input
            .into_iter()
            .enumerate()
            .filter_map(|(batch, data)| data.map(|data| (batch, data)))
            .map(|(batch, data)| {
                let shape = Shape::new(info.num_vocab, 1, 1, 1);
                TensorCpu::from_data(context, shape, data).map(|tensor| (batch, tensor))
            })
            .try_collect()?;
        let input = TensorCpu::stack(
            input
                .into_iter()
                .enumerate()
                .map(|(index, (batch, tensor))| {
                    redirect[batch] = Some(index);
                    tensor
                })
                .collect_vec(),
        )?;

        let num_batch = input.shape()[2];
        let softmax = self.request_softmax(num_batch);
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

        let mut output = softmax
            .map
            .clone()
            .back_async()
            .await
            .split(2)
            .expect("split buffer map")
            .into_iter()
            .map(|tensor| Some(tensor.to_vec()))
            .collect_vec();

        let mut probs = vec![None; max_batch];
        for (probs, redirect) in probs.iter_mut().zip_eq(redirect.into_iter()) {
            if let Some(redirect) = redirect {
                std::mem::swap(probs, &mut output[redirect]);
            }
        }

        Ok(probs)
    }
}
