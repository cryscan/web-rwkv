use std::{
    future::{Future, IntoFuture},
    pin::Pin,
};

use wgpu::CommandBuffer;

use super::OutputType;
use crate::{
    context::Context,
    num::Float,
    tensor::{kind::ReadWrite, TensorCpu, TensorError, TensorGpu},
};

type Batch<T> = Vec<T>;
type BoxedFuture<Output> = Pin<Box<dyn Future<Output = Output>>>;

pub trait Pipeline {
    type Output;
    type Backed;
    type Error;

    fn execute(self) -> impl Future<Output = Result<Self::Output, Self::Error>>;
    fn back(output: Self::Output) -> impl Future<Output = Self::Backed>;
}

pub trait ModelInput: IntoFuture<Output = Vec<u16>, IntoFuture = BoxedFuture<Vec<u16>>> {
    fn num_token(&self) -> usize;
    fn output_type(&self) -> OutputType;
    fn split_at(self, mid: usize) -> (Box<dyn ModelInput>, Box<dyn ModelInput>);
}

#[derive(Debug, Default, Clone)]
pub struct ModelInputLastOutput(pub Vec<u16>);

impl IntoFuture for ModelInputLastOutput {
    type Output = Vec<u16>;
    type IntoFuture = BoxedFuture<Vec<u16>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.0 })
    }
}

impl ModelInput for ModelInputLastOutput {
    fn num_token(&self) -> usize {
        self.0.len()
    }

    fn output_type(&self) -> OutputType {
        OutputType::Last
    }

    fn split_at(self, mid: usize) -> (Box<dyn ModelInput>, Box<dyn ModelInput>) {
        let (head, tail) = self.0.split_at(mid);
        (Box::new(Self(head.to_vec())), Box::new(Self(tail.to_vec())))
    }
}

pub struct ModelInputFullOutput(pub Vec<u16>);

impl IntoFuture for ModelInputFullOutput {
    type Output = Vec<u16>;
    type IntoFuture = BoxedFuture<Vec<u16>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.0 })
    }
}

impl ModelInput for ModelInputFullOutput {
    fn num_token(&self) -> usize {
        self.0.len()
    }

    fn output_type(&self) -> OutputType {
        OutputType::Full
    }

    fn split_at(self, mid: usize) -> (Box<dyn ModelInput>, Box<dyn ModelInput>) {
        let (head, tail) = self.0.split_at(mid);
        (Box::new(Self(head.to_vec())), Box::new(Self(tail.to_vec())))
    }
}

pub struct ModelInputFutureToken<F>(pub F)
where
    F: Future<Output = u16> + 'static;

impl<F> IntoFuture for ModelInputFutureToken<F>
where
    F: Future<Output = u16> + 'static,
{
    type Output = Vec<u16>;
    type IntoFuture = BoxedFuture<Vec<u16>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { vec![self.0.await] })
    }
}

impl<F> ModelInput for ModelInputFutureToken<F>
where
    F: Future<Output = u16> + 'static,
{
    fn num_token(&self) -> usize {
        1
    }

    fn output_type(&self) -> OutputType {
        OutputType::Last
    }

    fn split_at(self, mid: usize) -> (Box<dyn ModelInput>, Box<dyn ModelInput>) {
        match mid {
            0 => (Box::<ModelInputLastOutput>::default(), Box::new(self)),
            1 => (Box::new(self), Box::<ModelInputLastOutput>::default()),
            _ => panic!("split at {mid} out of range"),
        }
    }
}

pub struct Run<F: Float> {
    context: Context,
    command: CommandBuffer,

    future: BoxedFuture<TensorCpu<'static, F>>,
    input: TensorGpu<F, ReadWrite>,
    output: TensorGpu<F, ReadWrite>,
}

impl<F: Float> Pipeline for Run<F> {
    type Output = TensorGpu<F, ReadWrite>;
    type Backed = TensorCpu<'static, F>;
    type Error = TensorError;

    async fn execute(self) -> Result<Self::Output, Self::Error> {
        let tensor = self.future.await;
        self.input.load(&tensor)?;
        self.context.queue.submit(Some(self.command));
        Ok(self.output)
    }

    async fn back(output: Self::Output) -> Self::Backed {
        output.back_async().await
    }
}

pub trait ModelRun<F: Float, M, S> {
    fn run(model: &M, state: &S, tokens: &mut Batch<Box<dyn ModelInput>>) -> Run<F>;
}
