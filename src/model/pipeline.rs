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

type BoxedFuture<Output> = Pin<Box<dyn Future<Output = Output>>>;

pub trait Pipeline {
    type Input;
    type Output;
    type Error;

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

pub struct Batch<T: Float> {
    pub redirect: Vec<(usize, usize)>,
    pub tensor: TensorCpu<'static, T>,
}

pub struct Run<T, F>
where
    T: Float,
    F: Future<Output = Result<TensorCpu<'static, T>, TensorError>>,
{
    command: CommandBuffer,
    future: F,
    redirect: Vec<(usize, usize)>,
    input: TensorGpu<T, ReadWrite>,
    output: TensorGpu<T, ReadWrite>,
}

impl<T, F> Pipeline for Run<T, F>
where
    T: Float,
    F: Future<Output = Result<TensorCpu<'static, T>, TensorError>>,
{
    type Input = TensorCpu<'static, T>;
    type Output = Batch<T>;
    type Error = TensorError;

    async fn execute(self) -> Result<Self::Output, Self::Error> {
        let tensor = self.future.await?;
        self.input.load(&tensor)?;

        let context = self.input.context();
        context.queue.submit(Some(self.command));

        let redirect = self.redirect;
        let tensor = self.output.back_async().await;
        Ok(Batch { redirect, tensor })
    }
}

impl<T, Fut> Run<T, Fut>
where
    T: Float,
    Fut: Future<Output = Result<TensorCpu<'static, T>, TensorError>>,
{
    pub fn transform<Output, F>(self, f: F) -> impl Pipeline
    where
        F: FnMut(<Self as Pipeline>::Output) -> Result<Output, <Self as Pipeline>::Error>,
    {
        Transform { parent: self, f }
    }
}

pub struct Transform<P, Input, Output, F>
where
    P: Pipeline<Output = Input>,
    F: FnMut(Input) -> Result<Output, P::Error>,
{
    parent: P,
    f: F,
}

impl<P, Input, Output, F> Pipeline for Transform<P, Input, Output, F>
where
    P: Pipeline<Output = Input>,
    F: FnMut(Input) -> Result<Output, P::Error>,
{
    type Input = P::Input;
    type Output = Output;
    type Error = P::Error;

    async fn execute(mut self) -> Result<Self::Output, Self::Error> {
        let output = self.parent.execute().await?;
        (self.f)(output)
    }
}
