use std::future::Future;

use anyhow::Result;
use web_rwkv_derive::Deref;

pub mod loader;
pub mod model;
pub mod run;
pub mod softmax;
pub mod v5;
pub mod v6;

/// A [`Job`] to be executed on GPU.
pub trait Job: Sized + Send + 'static {
    type Input;
    type Output;

    /// Load the data from CPU to GPU.
    fn load(self, input: &Self::Input) -> Result<Self>;
    /// Submit the job to GPU and execute it immediately.
    fn submit(&mut self);
    /// Wait for the job to finish and read the data back.
    fn back(self) -> impl Future<Output = Result<Self::Output>> + Send + 'static;
}

pub trait JobBuilder: Send + 'static {
    type Seed;
    type Job: Job;

    /// Build a [`Job`] from the given seed.
    /// This usually involves creating a list of GPU commands (but not actually execution).
    fn build(&self, seed: Self::Seed) -> Result<Self::Job>;
}

#[derive(Debug)]
pub struct Submission<I, O> {
    pub input: I,
    pub sender: tokio::sync::oneshot::Sender<(I, O)>,
}

pub trait JobInput: Send + 'static {
    /// One chunk of the whole input at a step.
    type Chunk: Send + 'static;

    /// Advance the input for a step.
    fn step(&mut self);
    /// The current step's chunk to feed into the job.
    fn chunk(&self) -> Self::Chunk;
}

#[derive(Debug, Clone, Deref)]
pub struct JobRuntime<I, O>(tokio::sync::mpsc::Sender<Submission<I, O>>);

#[allow(clippy::type_complexity)]
impl<I, O, T, F> JobRuntime<I, O>
where
    T: Send + 'static,
    F: Iterator<Item = T> + Send + 'static,
    I: JobInput,
    O: Send + 'static,
    for<'a> &'a I: IntoIterator<Item = T, IntoIter = F>,
{
    pub async fn new<J>(builder: impl JobBuilder<Seed = T, Job = J>) -> Self
    where
        J: Job<Input = I::Chunk, Output = O>,
    {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        tokio::spawn(Self::run(builder, receiver));
        Self(sender)
    }

    async fn run<J>(
        builder: impl JobBuilder<Seed = T, Job = J>,
        mut receiver: tokio::sync::mpsc::Receiver<Submission<I, O>>,
    ) -> Result<()>
    where
        J: Job<Input = I::Chunk, Output = O>,
    {
        let mut predict: Option<J> = None;
        while let Some(Submission { input, sender }) = receiver.recv().await {
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
            let mut job = match predict.take().and_then(|job| load(job, &chunk)) {
                Some(job) => job,
                None => builder.build(info)?.load(&chunk)?,
            };

            async fn back<J: Job, I: JobInput>(
                job: J,
                mut input: I,
                sender: tokio::sync::oneshot::Sender<(I, J::Output)>,
            ) -> Result<()> {
                let output = job.back().await?;
                input.step();
                let _ = sender.send((input, output));
                Ok(())
            }

            job.submit();
            tokio::spawn(back(job, input, sender));

            predict = match next {
                Some(info) => Some(builder.build(info)?),
                None => None,
            };
        }
        Ok(())
    }
}
