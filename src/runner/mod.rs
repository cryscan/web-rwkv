use std::{collections::HashMap, future::Future};

use anyhow::Result;
use flume::{Receiver, Sender};

use self::{
    loader::{Lora, Reader},
    model::{EmbedDevice, Quant},
};
use crate::context::Context;

pub mod loader;
pub mod model;
pub mod run;
pub mod v6;

/// A [`Job`] to be executed on GPU.
pub trait Job: Sized {
    type Input;
    type Output;

    fn load(self, input: &Self::Input) -> Result<Self>;
    fn submit(self) -> impl Future<Output = Result<Self::Output>> + Send + 'static;
}

pub trait JobBuilder {
    type Seed;
    type Input: JobInput;
    type Output;

    /// Build a [`Job`] from the given seed.
    /// This usually involves creating a list of GPU commands (but not actually execution).
    fn build(
        &self,
        seed: Self::Seed,
    ) -> Result<impl Job<Input = <Self::Input as JobInput>::Chunk, Output = Self::Output>>;
}

pub struct Submission<I, O> {
    pub input: I,
    pub sender: Sender<(I, O)>,
}

pub trait JobInput {
    /// One chunk of the whole input at a step.
    type Chunk;

    /// Advance the input for a step.
    fn step(&mut self);
    /// The current step's chunk to feed into the job.
    fn chunk(&self) -> Self::Chunk;
}

pub struct JobRunner<I, O>(Receiver<Submission<I, O>>);

impl<I, O> JobRunner<I, O> {
    pub fn new(input: Receiver<Submission<I, O>>) -> Self {
        Self(input)
    }
}

impl<T, I, O> JobRunner<I, O>
where
    for<'a> &'a I: IntoIterator<Item = T>,
    I: JobInput,
    O: Send + 'static,
{
    pub async fn run(
        &self,
        builder: impl JobBuilder<Seed = T, Input = I, Output = O>,
    ) -> Result<()> {
        let mut predict = None;
        while let Ok(Submission { mut input, sender }) = self.0.recv_async().await {
            let mut iter = (&input).into_iter();
            let Some(info) = iter.next() else {
                continue;
            };

            let chunk = input.chunk();
            fn load<J: Job>(job: J, input: &J::Input) -> Option<J> {
                job.load(input).ok()
            }
            let job = match predict.take().and_then(|job| load(job, &chunk)) {
                Some(job) => job,
                None => builder.build(info)?.load(&chunk)?,
            };
            let handle = tokio::spawn(job.submit());

            predict = match iter.next() {
                Some(info) => Some(builder.build(info)?),
                None => None,
            };
            drop(iter);

            let output = handle.await??;
            input.step();
            let _ = sender.send_async((input, output)).await;
        }
        Ok(())
    }
}

pub trait Build<T> {
    fn build(self) -> impl Future<Output = Result<T>>;
}

pub struct RunnerBuilder<R: Reader> {
    context: Context,
    model: R,
    lora: Vec<Lora<R>>,
    quant: HashMap<usize, Quant>,
    embed_device: EmbedDevice,
    num_batch: usize,
}

impl<R: Reader> RunnerBuilder<R> {
    pub fn new(context: &Context, model: R) -> Self {
        Self {
            context: context.clone(),
            model,
            lora: vec![],
            quant: Default::default(),
            embed_device: Default::default(),
            num_batch: 1,
        }
    }

    pub fn with_quant(mut self, value: HashMap<usize, Quant>) -> Self {
        self.quant = value;
        self
    }

    pub fn add_lora(mut self, value: Lora<R>) -> Self {
        self.lora.push(value);
        self
    }

    pub fn with_embed_device(mut self, value: EmbedDevice) -> Self {
        self.embed_device = value;
        self
    }

    pub fn with_num_batch(mut self, value: usize) -> Self {
        let value = value.max(1);
        self.num_batch = value;
        self
    }
}
