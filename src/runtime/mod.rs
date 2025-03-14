use std::{future::Future, marker::PhantomData};

#[cfg(not(target_arch = "wasm32"))]
use futures::future::BoxFuture;
#[cfg(target_arch = "wasm32")]
use futures::future::LocalBoxFuture;
use thiserror::Error;

pub mod infer;
pub mod loader;
pub mod model;
pub mod softmax;
pub mod v4;
pub mod v5;
pub mod v6;
pub mod v7;

// const MAX_QUEUE_SIZE: usize = 2;

pub trait JobInfo: Clone + Send + Sync + 'static {
    /// Check if the info are compatible.
    fn check(&self, info: &Self) -> bool;
}

pub trait JobInput: Send + Sync + 'static {
    /// One chunk of the whole input at a step.
    type Chunk: Send + Sync + 'static;

    /// Advance the input for a step.
    fn step(&mut self);
    /// The current step's chunk to feed into the job.
    fn chunk(&self) -> Self::Chunk;
}

/// A [`Job`] to be executed on GPU.
pub trait Job: Sized {
    type Input: JobInput;
    type Output: Send + Sync + 'static;

    /// Load the data from CPU to GPU.
    fn load(self, input: &<Self::Input as JobInput>::Chunk) -> Result<Self, RuntimeError>;
    /// Submit the job to GPU and execute it immediately.
    fn submit(&mut self);
    #[cfg(not(target_arch = "wasm32"))]
    /// Wait for the job to finish and read the data back.
    fn back(self) -> impl Future<Output = Result<Self::Output, RuntimeError>> + Send;
    #[cfg(target_arch = "wasm32")]
    /// Wait for the job to finish and read the data back.
    fn back(self) -> impl Future<Output = Result<Self::Output, RuntimeError>>;
}

pub trait Dispatcher<J: Job>: Clone {
    type Info;

    /// Build a [`Job`] from the given info.
    /// This usually involves creating a list of GPU commands (but not actually execution).
    fn dispatch(&self, info: Self::Info) -> Result<J, RuntimeError>;
}

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
#[allow(clippy::type_complexity)]
#[derive(Debug)]
struct Submission<I: infer::Infer> {
    input: I::Input,
    sender: flume::Sender<Result<(I::Input, I::Output), RuntimeError>>,
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("input iterator exhausted")]
    InputExhausted,
    #[error("tensor error")]
    TensorError(#[from] crate::tensor::TensorError),
    #[error("recv error")]
    RecvError(#[from] flume::RecvError),
    #[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
    #[error("join error")]
    JoinError(#[from] tokio::task::JoinError),
}

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
#[derive(Debug, Clone)]
pub struct TokioRuntime<I: infer::Infer>(tokio::sync::mpsc::Sender<Submission<I>>);

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
#[allow(clippy::type_complexity)]
impl<I, T, F> TokioRuntime<I>
where
    I: infer::Infer,
    T: JobInfo,
    F: Iterator<Item = T> + Send + 'static,
    for<'a> &'a I::Input: IntoIterator<Item = T, IntoIter = F>,
{
    pub async fn new<J>(bundle: impl Dispatcher<J, Info = T> + Send + 'static) -> Self
    where
        J: Job<Input = I::Input, Output = I::Output> + Send + 'static,
    {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        let handle = tokio::spawn(Self::run(bundle, receiver));
        tokio::spawn(async move {
            match handle.await {
                Ok(_) => {}
                Err(err) => log::error!("{}", err),
            }
        });
        Self(sender)
    }

    async fn run<J>(
        model: impl Dispatcher<J, Info = T> + Send + 'static,
        mut receiver: tokio::sync::mpsc::Receiver<Submission<I>>,
    ) where
        J: Job<Input = I::Input, Output = I::Output> + Send + 'static,
    {
        let mut queue: Vec<(T, tokio::task::JoinHandle<Result<J, RuntimeError>>)> = vec![];
        let mut iter: Option<F> = None;
        let mut predict: usize = 0;

        'main: while let Some(Submission { input, sender }) = receiver.recv().await {
            let Some(info) = (&input).into_iter().next() else {
                let _ = sender.send(Err(RuntimeError::InputExhausted));
                continue 'main;
            };

            let chunk = input.chunk();

            let job = loop {
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

                    break match job {
                        Ok(Ok(job)) => job,
                        Ok(Err(error)) => {
                            let _ = sender.send(Err(error));
                            continue 'main;
                        }
                        Err(error) => {
                            let _ = sender.send(Err(error.into()));
                            continue 'main;
                        }
                    };
                }
            }
            .load(&chunk);

            let mut job = match job {
                Ok(job) => job,
                Err(error) => {
                    let _ = sender.send(Err(error));
                    continue 'main;
                }
            };

            async fn back<J: Job>(
                job: J,
                mut input: J::Input,
                sender: flume::Sender<Result<(J::Input, J::Output), RuntimeError>>,
            ) {
                let output = job.back().await;
                input.step();
                let _ = sender.send(output.map(|output| (input, output)));
            }

            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("submit").entered();
            job.submit();
            tokio::spawn(back(job, input, sender));
        }
    }

    /// Perform (partial) inference and return the remaining input and (perhaps partial) output.
    /// The amount of input processed during one call is bound by the input chunk size.
    pub async fn infer(&self, input: I::Input) -> Result<(I::Input, I::Output), RuntimeError> {
        let (sender, receiver) = flume::bounded(1);
        let submission = Submission { input, sender };
        let _ = self.0.send(submission).await;
        receiver.recv_async().await?
    }
}

#[derive(Debug, Clone)]
pub struct SimpleRuntime<M, I, J>(M, PhantomData<(I, J)>);

impl<M, I, J> SimpleRuntime<M, I, J> {
    #[inline]
    pub fn new<T, F>(bundle: M) -> Self
    where
        I: infer::Infer,
        J: Job<Input = I::Input, Output = I::Output>,
        T: JobInfo,
        F: Iterator<Item = T> + Send + 'static,
        M: Dispatcher<J, Info = T>,
        for<'a> &'a I::Input: IntoIterator<Item = T, IntoIter = F>,
    {
        Self(bundle, PhantomData)
    }

    pub async fn infer<T, F>(
        &self,
        mut input: I::Input,
    ) -> Result<(I::Input, I::Output), RuntimeError>
    where
        I: infer::Infer,
        J: Job<Input = I::Input, Output = I::Output>,
        T: JobInfo,
        F: Iterator<Item = T> + Send + 'static,
        M: Dispatcher<J, Info = T>,
        for<'a> &'a I::Input: IntoIterator<Item = T, IntoIter = F>,
    {
        let Some(info) = (&input).into_iter().next() else {
            return Err(RuntimeError::InputExhausted);
        };
        let chunk = input.chunk();

        let mut job = self.0.dispatch(info)?.load(&chunk)?;
        job.submit();

        let output = job.back().await?;
        input.step();

        Ok((input, output))
    }
}

#[allow(clippy::type_complexity)]
pub trait Runtime<I: infer::Infer> {
    #[cfg(not(target_arch = "wasm32"))]
    fn infer(&self, input: I::Input) -> BoxFuture<Result<(I::Input, I::Output), RuntimeError>>;

    #[cfg(target_arch = "wasm32")]
    fn infer(&self, input: I::Input)
        -> LocalBoxFuture<Result<(I::Input, I::Output), RuntimeError>>;
}

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
#[allow(clippy::type_complexity)]
impl<I, T, F> Runtime<I> for TokioRuntime<I>
where
    I: infer::Infer,
    T: JobInfo,
    F: Iterator<Item = T> + Send + 'static,
    for<'a> &'a I::Input: IntoIterator<Item = T, IntoIter = F>,
{
    #[cfg(not(target_arch = "wasm32"))]
    fn infer(&self, input: I::Input) -> BoxFuture<Result<(I::Input, I::Output), RuntimeError>> {
        Box::pin(self.infer(input))
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::type_complexity)]
impl<M, I, J, T, F> Runtime<I> for SimpleRuntime<M, I, J>
where
    I: infer::Infer,
    J: Job<Input = I::Input, Output = I::Output> + Send + Sync + 'static,
    M: Dispatcher<J, Info = T> + Sync,
    T: JobInfo,
    F: Iterator<Item = T> + Send + 'static,
    for<'a> &'a I::Input: IntoIterator<Item = T, IntoIter = F>,
{
    fn infer(&self, input: I::Input) -> BoxFuture<Result<(I::Input, I::Output), RuntimeError>> {
        Box::pin(self.infer(input))
    }
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::type_complexity)]
impl<M, I, J, T, F> Runtime<I> for SimpleRuntime<M, I, J>
where
    I: infer::Infer,
    J: Job<Input = I::Input, Output = I::Output>,
    M: Dispatcher<J, Info = T>,
    T: JobInfo,
    F: Iterator<Item = T> + Send + 'static,
    for<'a> &'a I::Input: IntoIterator<Item = T, IntoIter = F>,
{
    fn infer(
        &self,
        input: I::Input,
    ) -> LocalBoxFuture<Result<(I::Input, I::Output), RuntimeError>> {
        Box::pin(self.infer(input))
    }
}
