use std::{future::Future, marker::PhantomData};

use flume::{Receiver, Sender};

pub mod model;
pub mod run;
pub mod v6;

pub trait Job: Sized {
    type Input;
    type Output;
    type Error;

    fn load(self, input: &Self::Input) -> Result<Self, Self::Error>;
    fn submit(self) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send + 'static;
}

pub trait JobBuilder {
    type Info;
    type Input;
    type Output;
    type Error;

    fn build(
        &self,
        input: Self::Info,
    ) -> Result<
        impl Job<Input = Self::Input, Output = Self::Output, Error = Self::Error>,
        Self::Error,
    >;
}

pub struct Submission<I, O> {
    pub input: I,
    pub sender: Sender<O>,
}

pub struct JobRunner<I, O, E>(Receiver<Submission<I, O>>, PhantomData<E>);

impl<I, O, E> JobRunner<I, O, E> {
    pub fn new(input: Receiver<Submission<I, O>>) -> Self {
        Self(input, PhantomData)
    }
}

impl<F, I, O, E> JobRunner<I, O, E>
where
    for<'a> &'a I: IntoIterator<Item = F>,
    O: Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    pub async fn run(
        &self,
        builder: impl JobBuilder<Info = F, Input = I, Output = O, Error = E>,
    ) -> Result<(), E> {
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

            let output = job.submit();

            tokio::spawn(async move {
                let output = output.await;
                let _ = sender.send_async(output.unwrap()).await;
            });

            speculation = match iter.next() {
                Some(info) => Some(builder.build(info)?),
                None => None,
            };
        }
        Ok(())
    }
}
