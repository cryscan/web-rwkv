use wgpu::CommandBuffer;

use crate::{
    num::Float,
    tensor::{kind::ReadWrite, TensorCpu, TensorError, TensorGpu},
};

#[cfg(feature = "tokio")]
pub mod tokio;

#[derive(Debug)]
pub struct Job<F: Float> {
    input: TensorGpu<F, ReadWrite>,
    output: TensorGpu<F, ReadWrite>,
    command: CommandBuffer,
}

impl<F: Float> Job<F> {
    pub fn load(&self, input: &TensorCpu<F>) -> Result<(), TensorError> {
        self.input.load(input)
    }

    pub async fn submit(self) -> TensorCpu<'static, F> {
        let context = self.output.context();
        context.queue.submit(Some(self.command));
        self.output.back().await
    }
}

#[derive(Debug, Default, Clone)]
pub struct RunInput(pub Vec<(usize, RunOutput)>);

impl RunInput {
    pub fn num_token(&self) -> usize {
        self.0.iter().map(|(x, _)| x).sum()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum RunOutput {
    #[default]
    None,
    Last,
    Full,
}

pub trait Run {
    type Model;
    type State;

    fn run<F: Float>(model: &Self::Model, state: &Self::State, input: RunInput) -> Job<F>;
}

#[derive(Debug, Default, Clone)]
pub struct SoftmaxInput(pub Vec<usize>);

pub trait Softmax {
    fn softmax<F: Float>(input: SoftmaxInput) -> Job<F>;
}
