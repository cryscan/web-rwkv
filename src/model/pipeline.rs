use std::{
    future::{Future, IntoFuture},
    pin::Pin,
};

use wgpu::CommandBuffer;

use super::OutputType;
use crate::{
    num::Float,
    tensor::{kind::ReadWrite, TensorCpu, TensorError, TensorGpu},
};

type Batch<T> = Vec<T>;
type BoxedFuture<Output> = Pin<Box<dyn Future<Output = Output>>>;

pub trait Pipeline {
    type Output;
    type Error;

    /// Wait for required data to be ready and submit the GPU jobs.
    fn execute(self) -> impl Future<Output = Result<Self::Output, Self::Error>>;
}

pub trait Prompt: IntoFuture<Output = Vec<u16>, IntoFuture = BoxedFuture<Vec<u16>>> {
    /// Number of tokens in the prompt.
    fn num_token(&self) -> usize;

    /// Output type of the prompt, can either only compute the last token's prediction or compute all tokens'.
    fn output_type(&self) -> OutputType;

    /// Split the prompt into two at `mid`. `mid` must be less than `num_token`.
    fn split_at(self, mid: usize) -> (Box<dyn Prompt>, Box<dyn Prompt>);
}

#[derive(Debug, Default, Clone)]
pub struct PresentPrompt {
    pub tokens: Vec<u16>,
    pub output_type: OutputType,
}

impl IntoFuture for PresentPrompt {
    type Output = Vec<u16>;
    type IntoFuture = BoxedFuture<Vec<u16>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.tokens })
    }
}

impl Prompt for PresentPrompt {
    fn num_token(&self) -> usize {
        self.tokens.len()
    }

    fn output_type(&self) -> OutputType {
        self.output_type
    }

    fn split_at(self, mid: usize) -> (Box<dyn Prompt>, Box<dyn Prompt>) {
        let output_type = self.output_type;
        let (head, tail) = self.tokens.split_at(mid);
        let head = Self {
            tokens: head.to_vec(),
            output_type,
        };
        let tail = Self {
            tokens: tail.to_vec(),
            output_type,
        };
        (Box::new(head), Box::new(tail))
    }
}

pub struct FuturePrompt<F: Future<Output = u16> + 'static>(pub F);

impl<F: Future<Output = u16> + 'static> IntoFuture for FuturePrompt<F> {
    type Output = Vec<u16>;
    type IntoFuture = BoxedFuture<Vec<u16>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { vec![self.0.await] })
    }
}

impl<F: Future<Output = u16> + 'static> Prompt for FuturePrompt<F> {
    fn num_token(&self) -> usize {
        1
    }

    fn output_type(&self) -> OutputType {
        OutputType::Last
    }

    fn split_at(self, mid: usize) -> (Box<dyn Prompt>, Box<dyn Prompt>) {
        match mid {
            0 => (Box::<PresentPrompt>::default(), Box::new(self)),
            1 => (Box::new(self), Box::<PresentPrompt>::default()),
            _ => panic!("split at {mid} out of range"),
        }
    }
}

pub struct Run<F: Float> {
    command: CommandBuffer,
    future: BoxedFuture<TensorCpu<'static, F>>,
    input: TensorGpu<F, ReadWrite>,
    output: Batch<TensorGpu<F, ReadWrite>>,
}

impl<F: Float> Pipeline for Run<F> {
    type Output = Batch<TensorGpu<F, ReadWrite>>;
    type Error = TensorError;

    async fn execute(self) -> Result<Self::Output, Self::Error> {
        let context = self.input.context();
        let tensor = self.future.await;
        self.input.load(&tensor)?;
        context.queue.submit(Some(self.command));
        Ok(self.output)
    }
}

pub trait ModelRun<F: Float, M, S> {
    fn run(model: &M, state: &S, tokens: &mut Batch<Box<dyn Prompt>>) -> Run<F>;
}
