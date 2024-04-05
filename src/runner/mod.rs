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
pub trait Job: Sized + Send + 'static {
    type Input;
    type Output;

    fn load(self, input: &Self::Input) -> Result<Self>;
    fn submit(self) -> impl Future<Output = Result<Self::Output>> + Send + 'static;
}

pub trait JobBuilder: Send + 'static {
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

#[derive(Debug, Clone)]
pub struct Submission<I, O> {
    pub input: I,
    pub sender: Sender<(I, O)>,
}

pub trait JobInput: Send + 'static {
    /// One chunk of the whole input at a step.
    type Chunk;

    /// Advance the input for a step.
    fn step(&mut self);
    /// The current step's chunk to feed into the job.
    fn chunk(&self) -> Self::Chunk;
}

#[derive(Debug, Clone)]
pub struct JobRuntime<I, O>(Receiver<Submission<I, O>>);

impl<T, I, O> JobRuntime<I, O>
where
    for<'a> &'a I: IntoIterator<Item = T>,
    I: JobInput,
    O: Send + 'static,
{
    pub async fn start(
        builder: impl JobBuilder<Seed = T, Input = I, Output = O>,
    ) -> Sender<Submission<I, O>> {
        let (sender, receiver) = flume::unbounded();
        let runtime = JobRuntime(receiver);
        tokio::spawn(async move {
            runtime.run(builder).await.expect("runtime error");
        });
        sender
    }

    async fn run(&self, builder: impl JobBuilder<Seed = T, Input = I, Output = O>) -> Result<()> {
        let mut predict = None;
        while let Ok(Submission { mut input, sender }) = self.0.recv_async().await {
            let mut iter = (&input).into_iter();
            let Some(info) = iter.next() else {
                continue;
            };
            let next = iter.next();
            drop(iter);

            let chunk = input.chunk();
            fn load<J: Job>(job: J, input: &J::Input) -> Option<J> {
                job.load(input).ok()
            }
            let job = match predict.take().and_then(|job| load(job, &chunk)) {
                Some(job) => job,
                None => builder.build(info)?.load(&chunk)?,
            };

            let output = job.submit();
            tokio::spawn(async move {
                let output = output.await.expect("job execution error");
                input.step();
                let _ = sender.send_async((input, output)).await;
            });

            predict = match next {
                Some(info) => Some(builder.build(info)?),
                None => None,
            };
        }
        Ok(())
    }
}

pub trait Build<T> {
    fn build(self) -> impl Future<Output = Result<T>>;
}

pub struct ModelBuilder<R: Reader> {
    context: Context,
    model: R,
    lora: Vec<Lora<R>>,
    quant: HashMap<usize, Quant>,
    embed_device: EmbedDevice,
    num_batch: usize,
}

impl<R: Reader> ModelBuilder<R> {
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
