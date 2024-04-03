use std::future::Future;

use flume::{Receiver, Sender};

pub mod run;

pub trait Job {
    type Input;
    type Output;
    type Error: std::error::Error;

    fn load(&self, input: Self::Input) -> Result<(), Self::Error>;
    fn submit(self) -> impl Future<Output = Self::Output> + Send;
}

pub trait JobBuilder {
    type Info;
    type Input;
    type Output;

    fn build(
        &self,
        info: Self::Info,
    ) -> impl Job<Input = Self::Input, Output = Self::Output> + 'static;
}

pub struct Submission<Input, Output> {
    pub input: Input,
    pub sender: Sender<Output>,
}

pub struct JobRunner<Input, Output> {
    input: Receiver<Submission<Input, Output>>,
}

impl<Input, Output> JobRunner<Input, Output> {
    pub fn new(input: Receiver<Submission<Input, Output>>) -> Self {
        Self { input }
    }
}

impl<Info, Input, Output> JobRunner<Input, Output>
where
    Input: IntoIterator<Item = Info> + Copy,
    Output: Send + 'static,
{
    pub async fn run(
        &self,
        builder: &impl JobBuilder<Info = Info, Input = Input, Output = Output>,
    ) {
        let mut speculation = None;
        while let Ok(Submission { input, sender }) = self.input.recv_async().await {
            let mut iter = input.into_iter();
            let Some(info) = iter.next() else {
                continue;
            };

            fn load<J: Job>(job: J, input: J::Input) -> Option<J> {
                job.load(input).ok().and(Some(job))
            }

            let job = match speculation.take().and_then(|job| load(job, input)) {
                Some(job) => job,
                None => {
                    let job = builder.build(info);
                    job.load(input).unwrap();
                    job
                }
            };

            let output = job.submit();

            #[cfg(feature = "tokio")]
            tokio::spawn(async move {
                let output = output.await;
                let _ = sender.send_async(output).await;
            });

            speculation = iter.next().map(|info| builder.build(info));

            #[cfg(not(feature = "tokio"))]
            let _ = sender.send_async(output.await).await;
        }
    }
}
