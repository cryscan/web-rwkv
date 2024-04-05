use std::future::Future;

use anyhow::Result;
use flume::{Receiver, Sender};

pub mod model;
pub mod run;
pub mod v6;

pub trait Job: Sized {
    type Input;
    type Output;

    fn load(self, input: &Self::Input) -> Result<Self>;
    fn submit(self) -> impl Future<Output = Result<Self::Output>> + Send + 'static;
}

pub trait JobBuilder {
    type Seed;
    type Input;
    type Output;

    fn build(
        &self,
        seed: Self::Seed,
    ) -> Result<impl Job<Input = Self::Input, Output = Self::Output>>;
}

pub struct Submission<I, O> {
    pub input: I,
    pub sender: Sender<(I, O)>,
}

pub trait JobInput {
    /// Advance the input for a step.
    fn step(self) -> Self;
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
        let mut speculation = None;
        while let Ok(Submission { input, sender }) = self.0.recv_async().await {
            let mut iter = (&input).into_iter();
            let Some(info) = iter.next() else {
                continue;
            };

            fn load<J: Job>(job: J, input: &J::Input) -> Option<J> {
                job.load(input).ok()
            }
            let job = match speculation.take().and_then(|job| load(job, &input)) {
                Some(job) => job,
                None => builder.build(info)?.load(&input)?,
            };
            let handle = tokio::spawn(job.submit());

            speculation = match iter.next() {
                Some(info) => Some(builder.build(info)?),
                None => None,
            };
            drop(iter);

            let output = handle.await??;
            let _ = sender.send_async((input.step(), output)).await;
        }
        Ok(())
    }
}
