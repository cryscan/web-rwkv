use std::{future::Future, marker::PhantomData};

use anyhow::{bail, Result};

pub mod infer;
pub mod loader;
pub mod model;
pub mod softmax;
pub mod v4;
pub mod v5;
pub mod v6;

// const MAX_QUEUE_SIZE: usize = 2;

pub trait JobInfo: Send + Clone + 'static {
    /// Check if the info are compatible.
    fn check(&self, info: &Self) -> bool;
}

pub trait JobInput: Send + 'static {
    /// One chunk of the whole input at a step.
    type Chunk: Send + 'static;

    /// Advance the input for a step.
    fn step(&mut self);
    /// The current step's chunk to feed into the job.
    fn chunk(&self) -> Self::Chunk;
}

/// A [`Job`] to be executed on GPU.
pub trait Job: Sized + Send + 'static {
    type Input: JobInput;
    type Output;

    /// Load the data from CPU to GPU.
    fn load(self, input: &<<Self as Job>::Input as JobInput>::Chunk) -> Result<Self>;
    /// Submit the job to GPU and execute it immediately.
    fn submit(&mut self);
    /// Wait for the job to finish and read the data back.
    fn back(self) -> impl Future<Output = Result<Self::Output>> + Send;
}

pub trait Dispatcher<J: Job>: Send + Clone + 'static {
    type Info;

    /// Build a [`Job`] from the given info.
    /// This usually involves creating a list of GPU commands (but not actually execution).
    fn dispatch(&self, info: Self::Info) -> Result<J>;
}

#[derive(Debug)]
struct Submission<I, O> {
    input: I,
    sender: flume::Sender<(I, O)>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub struct TokioRuntime<I, O>(tokio::sync::mpsc::Sender<Submission<I, O>>);

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::type_complexity)]
impl<I, O, T, F> TokioRuntime<I, O>
where
    I: JobInput,
    O: Send + 'static,
    T: JobInfo,
    F: Iterator<Item = T> + Send + 'static,
    for<'a> &'a I: IntoIterator<Item = T, IntoIter = F>,
{
    pub async fn new<J: Job<Input = I, Output = O>>(model: impl Dispatcher<J, Info = T>) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        let handle = tokio::spawn(Self::run(model, receiver));
        tokio::spawn(async move {
            match handle.await {
                Ok(_) => {}
                Err(err) => log::error!("{}", err),
            }
        });
        Self(sender)
    }

    async fn run<J: Job<Input = I, Output = O>>(
        model: impl Dispatcher<J, Info = T>,
        mut receiver: tokio::sync::mpsc::Receiver<Submission<I, O>>,
    ) -> Result<()> {
        let mut queue: Vec<(T, tokio::task::JoinHandle<Result<J>>)> = vec![];
        let mut iter: Option<F> = None;
        let mut predict: usize = 0;

        while let Some(Submission { input, sender }) = receiver.recv().await {
            let Some(info) = (&input).into_iter().next() else {
                continue;
            };

            let chunk = input.chunk();

            let mut job = loop {
                let mut candidates = vec![];
                let mut remain = vec![];
                for (key, handle) in queue {
                    match (candidates.is_empty(), info.check(&key)) {
                        (true, false) => handle.abort(),
                        (false, false) => remain.push((key, handle)),
                        (_, true) => candidates.push(handle),
                    }
                }
                queue = remain;

                predict = match predict {
                    2 => 1,
                    1 => 0,
                    0 => 2,
                    _ => unreachable!(),
                };

                // we have a cache miss, restart the pipeline
                if candidates.is_empty() || iter.is_none() {
                    iter = Some((&input).into_iter());
                    predict = 2;
                }
                let iter = iter.as_mut().unwrap();

                for info in iter.take(predict) {
                    #[cfg(feature = "trace")]
                    tracing::event!(
                        tracing::Level::TRACE,
                        "launch ({queue}, {candidates}, {predict})",
                        queue = queue.len(),
                        candidates = candidates.len(),
                        predict = predict
                    );

                    let key = info.clone();
                    let model = model.clone();
                    let handle = tokio::task::spawn_blocking(move || model.dispatch(key));
                    queue.push((info.clone(), handle));
                }

                if !candidates.is_empty() {
                    let (job, _, remain) = futures::future::select_all(candidates).await;
                    let mut remain = remain
                        .into_iter()
                        .map(|handle| (info.clone(), handle))
                        .collect();
                    std::mem::swap(&mut queue, &mut remain);
                    queue.append(&mut remain);
                    break job??;
                }
            }
            .load(&chunk)?;

            async fn back<J: Job, I: JobInput>(
                job: J,
                mut input: I,
                sender: flume::Sender<(I, J::Output)>,
            ) -> Result<()> {
                let output = job.back().await?;
                input.step();
                let _ = sender.send((input, output));
                Ok(())
            }

            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("submit").entered();
            job.submit();
            tokio::spawn(back(job, input, sender));
        }
        Ok(())
    }

    /// Perform (partial) inference and return the remaining input and (perhaps partial) output.
    /// The amount of input processed during one call is bound by the input chunk size.
    pub async fn infer(&self, input: I) -> (I, O) {
        let (sender, receiver) = flume::bounded(1);
        let submission = Submission { input, sender };
        let _ = self.0.send(submission).await;
        receiver
            .recv_async()
            .await
            .expect("receive infer output error")
    }
}

#[derive(Debug, Clone)]
pub struct SimpleRuntime<M, J, I, O> {
    model: M,
    _phantom: PhantomData<(J, I, O)>,
}

impl<I, O, J, M, T, F> SimpleRuntime<M, J, I, O>
where
    I: JobInput,
    O: Send + 'static,
    J: Job<Input = I, Output = O>,
    M: Dispatcher<J, Info = T>,
    T: JobInfo,
    F: Iterator<Item = T> + Send + 'static,
    for<'a> &'a I: IntoIterator<Item = T, IntoIter = F>,
{
    pub fn new(model: M) -> Self {
        Self {
            model,
            _phantom: PhantomData,
        }
    }

    pub async fn infer(&self, mut input: I) -> Result<(I, O)> {
        let Some(info) = (&input).into_iter().next() else {
            bail!("input iterator exhausted")
        };
        let chunk = input.chunk();

        let mut job = self.model.dispatch(info)?.load(&chunk)?;
        job.submit();

        let output = job.back().await?;
        input.step();

        Ok((input, output))
    }
}
