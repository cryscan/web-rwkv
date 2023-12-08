use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use super::{ModelBase, ModelError, ModelInfo, ModelState};
use crate::{
    context::Context,
    tensor::{shape::Shape, ReadBack, ReadWrite, TensorGpu},
};

#[derive(Debug)]
pub struct Output {
    pub head_x: TensorGpu<f32, ReadWrite>,
    pub head_o: TensorGpu<f32, ReadWrite>,
    pub map: TensorGpu<f32, ReadBack>,
}

impl Output {
    pub fn new(context: &Context, info: &ModelInfo, num_batch: usize) -> Self {
        let head_shape = Shape::new(info.num_emb, num_batch, 1, 1);
        let output_shape = Shape::new(info.num_vocab, num_batch, 1, 1);

        Self {
            head_x: context.tensor_init(head_shape),
            head_o: context.tensor_init(output_shape),
            map: context.tensor_init(output_shape),
        }
    }
}

#[async_trait]
pub trait ModelRun: ModelBase {
    fn request_output(&self, num_batch: usize) -> Arc<Output>;

    /// Actual implementation of the model's inference.
    #[allow(clippy::type_complexity)]
    fn run_internal(
        &self,
        tokens: Vec<Vec<u16>>,
        state: &Self::ModelState,
        last: Option<usize>,
    ) -> Result<(Arc<Output>, Vec<Option<usize>>)>;

    /// Run the model for a batch of tokens as input.
    /// The length of `tokens` must match the number of batches in `state`.
    /// `tokens` may have slots with no tokens, for which `run` won't compute that batch and will return an empty vector in that corresponding slot.
    async fn run(
        &self,
        tokens: &mut Vec<Vec<u16>>,
        state: &Self::ModelState,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let num_token: usize = tokens.iter().map(Vec::len).sum();
        let max_batch = state.max_batch();

        if tokens.len() != max_batch {
            return Err(ModelError::BatchSize(tokens.len(), max_batch).into());
        }
        if num_token == 0 {
            return Err(ModelError::EmptyInput.into());
        }

        // we only infer at most `token_chunk_size` tokens at a time
        let mut num_token = num_token.min(self.token_chunk_size());
        let mut inputs = vec![vec![]; max_batch];
        let mut last = None;

        // take `num_token` tokens out of all the inputs and put into `input`
        for (index, (batch, input)) in tokens.iter_mut().zip(inputs.iter_mut()).enumerate() {
            let mid = batch.len().min(num_token);
            num_token -= mid;

            let (head, tail) = batch.split_at(mid);
            last = (!tail.is_empty()).then_some(index);
            *input = head.to_vec();
            *batch = tail.to_vec();

            if num_token == 0 {
                break;
            }
        }

        let (output, redirect) = self.run_internal(inputs, state, last)?;
        let output = output.map.clone().back_async().await;

        Ok(redirect
            .into_iter()
            .map(|index| {
                index.map(|index| {
                    output
                        .slice(.., index, .., ..)
                        .expect("this never happens")
                        .to_vec()
                })
            })
            .collect())
    }
}
