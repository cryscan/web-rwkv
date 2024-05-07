use std::future::Future;

use anyhow::Result;
use web_rwkv_derive::Deref;

pub mod infer;
pub mod loader;
pub mod model;
pub mod softmax;
pub mod v4;
pub mod v5;
pub mod v6;

const MAX_QUEUE_SIZE: usize = 2;

pub trait JobInfo: Send + Clone + 'static {
    /// Check if the info are compatible.
    fn check(&self, info: &Self) -> bool;
}

/// A [`Job`] to be executed on GPU.
pub trait Job: Sized + Send + 'static {
    type Info: JobInfo;
    type Input;
    type Output;

    /// Load the data from CPU to GPU.
    fn load(self, input: &Self::Input) -> Result<Self>;
    /// Submit the job to GPU and execute it immediately.
    fn submit(&mut self);
    /// Wait for the job to finish and read the data back.
    fn back(self) -> impl Future<Output = Result<Self::Output>> + Send;
}

pub trait JobBuilder<J: Job>: Send + Clone + 'static {
    type Info;

    /// Build a [`Job`] from the given info.
    /// This usually involves creating a list of GPU commands (but not actually execution).
    fn build(&self, info: Self::Info) -> Result<J>;
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
    T: JobInfo,
    F: Iterator<Item = T> + Send + 'static,
    I: JobInput,
    O: Send + 'static,
    for<'a> &'a I: IntoIterator<Item = T, IntoIter = F>,
{
    pub async fn new<J>(builder: impl JobBuilder<J, Info = T>) -> Self
    where
        J: Job<Info = T, Input = I::Chunk, Output = O>,
    {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        let handle = tokio::spawn(Self::run(builder, receiver));
        tokio::spawn(async move {
            match handle.await {
                Ok(_) => {}
                Err(err) => log::error!("{}", err),
            }
        });
        Self(sender)
    }

    async fn run<J>(
        builder: impl JobBuilder<J, Info = T>,
        mut receiver: tokio::sync::mpsc::Receiver<Submission<I, O>>,
    ) -> Result<()>
    where
        J: Job<Info = T, Input = I::Chunk, Output = O>,
    {
        let mut queue: Vec<(T, tokio::task::JoinHandle<Result<J>>)> = vec![];

        while let Some(Submission { input, sender }) = receiver.recv().await {
            let mut iter = (&input).into_iter();
            let Some(info) = iter.next() else {
                continue;
            };

            let chunk = input.chunk();

            let mut job = loop {
                let mut job = None;
                let mut remain = vec![];
                for (_info, handle) in queue.drain(..) {
                    match (job.is_none(), info.check(&_info)) {
                        (true, true) => job = Some(handle),
                        (true, false) => handle.abort(),
                        (false, _) => remain.push((_info, handle)),
                    }
                }
                queue.append(&mut remain);

                let predict = MAX_QUEUE_SIZE - MAX_QUEUE_SIZE.min(queue.len());
                for info in (&input).into_iter().take(predict) {
                    let _info = info.clone();
                    let builder = builder.clone();
                    let handle = tokio::task::spawn_blocking(move || builder.build(_info));
                    queue.push((info.clone(), handle));
                }

                if let Some(job) = job {
                    break job.await??;
                }
            }
            .load(&chunk)?;

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
        }
        Ok(())
    }
}
