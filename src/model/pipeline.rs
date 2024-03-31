use std::{
    future::{Future, IntoFuture},
    pin::Pin,
};

use crate::{
    context::Context,
    num::Float,
    tensor::{kind::ReadWrite, shape::Shape, TensorCpu, TensorError, TensorGpu},
};

use super::{ModelInfo, ModelState, OutputType};

type Batch<T> = Vec<T>;
type BoxedFuture<Output> = Pin<Box<dyn Future<Output = Output>>>;

pub trait Pipeline {
    type Output;
    type Error;

    fn execute(self) -> impl Future<Output = Result<Self::Output, Self::Error>>;
}

pub trait RunInput: IntoFuture<Output = Vec<u16>, IntoFuture = BoxedFuture<Vec<u16>>> {
    fn num_token(&self) -> usize;
    fn output_type(&self) -> OutputType;
    fn split_at(self, mid: usize) -> (Box<dyn RunInput>, Box<dyn RunInput>);
}

#[derive(Debug, Default, Clone)]
pub struct RunInputLastOutput(pub Vec<u16>);

impl IntoFuture for RunInputLastOutput {
    type Output = Vec<u16>;
    type IntoFuture = BoxedFuture<Vec<u16>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.0 })
    }
}

impl RunInput for RunInputLastOutput {
    fn num_token(&self) -> usize {
        self.0.len()
    }

    fn output_type(&self) -> OutputType {
        OutputType::Last
    }

    fn split_at(self, mid: usize) -> (Box<dyn RunInput>, Box<dyn RunInput>) {
        let (head, tail) = self.0.split_at(mid);
        (Box::new(Self(head.to_vec())), Box::new(Self(tail.to_vec())))
    }
}

pub struct RunInputFullOutput(pub Vec<u16>);

impl IntoFuture for RunInputFullOutput {
    type Output = Vec<u16>;
    type IntoFuture = BoxedFuture<Vec<u16>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.0 })
    }
}

impl RunInput for RunInputFullOutput {
    fn num_token(&self) -> usize {
        self.0.len()
    }

    fn output_type(&self) -> OutputType {
        OutputType::Full
    }

    fn split_at(self, mid: usize) -> (Box<dyn RunInput>, Box<dyn RunInput>) {
        let (head, tail) = self.0.split_at(mid);
        (Box::new(Self(head.to_vec())), Box::new(Self(tail.to_vec())))
    }
}

pub struct RunInputFutureToken<F>(pub F)
where
    F: Future<Output = u16> + 'static;

impl<F> IntoFuture for RunInputFutureToken<F>
where
    F: Future<Output = u16> + 'static,
{
    type Output = Vec<u16>;
    type IntoFuture = BoxedFuture<Vec<u16>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { vec![self.0.await] })
    }
}

impl<F> RunInput for RunInputFutureToken<F>
where
    F: Future<Output = u16> + 'static,
{
    fn num_token(&self) -> usize {
        1
    }

    fn output_type(&self) -> OutputType {
        OutputType::Last
    }

    fn split_at(self, mid: usize) -> (Box<dyn RunInput>, Box<dyn RunInput>) {
        match mid {
            0 => (Box::<RunInputLastOutput>::default(), Box::new(self)),
            1 => (Box::new(self), Box::<RunInputLastOutput>::default()),
            _ => panic!("split at {mid} out of range"),
        }
    }
}

pub struct Run<F: Float> {
    future: BoxedFuture<TensorCpu<'static, F>>,
    input: TensorGpu<F, ReadWrite>,
    output: TensorGpu<F, ReadWrite>,
}

impl<F: Float> Pipeline for Run<F> {
    type Output = TensorGpu<F, ReadWrite>;
    type Error = TensorError;

    async fn execute(self) -> Result<Self::Output, Self::Error> {
        let tensor = self.future.await;
        self.input.load(&tensor)?;
        Ok(self.output)
    }
}

#[derive(Debug)]
pub struct Header<F: Float> {
    pub head_x: TensorGpu<F, ReadWrite>,
    pub head_o: TensorGpu<f32, ReadWrite>,
}

impl<F: Float> Header<F> {
    pub fn new(context: &Context, info: &ModelInfo, num_batch: usize) -> Self {
        let head_shape = Shape::new(info.num_emb, num_batch, 1, 1);
        let output_shape = Shape::new(info.num_vocab, num_batch, 1, 1);

        Self {
            head_x: context.tensor_init(head_shape),
            head_o: context.tensor_init(output_shape),
        }
    }
}

pub trait ModelRun<F: Float> {
    type State: ModelState;

    fn run(&self, state: &Self::State, tokens: &mut Batch<Box<dyn RunInput>>) -> Run<F>;
}
