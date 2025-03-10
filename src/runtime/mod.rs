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

pub trait JobInfo: Clone {
    /// Check if the info are compatible.
    fn check(&self, info: &Self) -> bool;
}

pub trait JobInput {
    /// One chunk of the whole input at a step.
    type Chunk;

    /// Advance the input for a step.
    fn step(&mut self);
    /// The current step's chunk to feed into the job.
    fn chunk(&self) -> Self::Chunk;
}

/// A [`Job`] to be executed on GPU.
pub trait Job: Sized {
    type Input: JobInput;
    type Output;

    /// Load the data from CPU to GPU.
    fn load(self, input: &<<Self as Job>::Input as JobInput>::Chunk) -> Result<Self, RuntimeError>;
    /// Submit the job to GPU and execute it immediately.
    fn submit(&mut self);
    #[cfg(not(target_arch = "wasm32"))]
    /// Wait for the job to finish and read the data back.
    fn back(self) -> impl Future<Output = Result<Self::Output, RuntimeError>> + Send;
    #[cfg(target_arch = "wasm32")]
    /// Wait for the job to finish and read the data back.
    fn back(self) -> impl Future<Output = Result<Self::Output, RuntimeError>>;
}

pub trait Dispatcher<J: Job> {
    type Info;

    /// Build a [`Job`] from the given info.
    /// This usually involves creating a list of GPU commands (but not actually execution).
    fn dispatch(&self, info: Self::Info) -> Result<J, RuntimeError>;
}

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
#[derive(Debug)]
struct Submission<I, O> {
    input: I,
    sender: flume::Sender<Result<(I, O), RuntimeError>>,
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
pub struct TokioRuntime<I, O>(tokio::sync::mpsc::Sender<Submission<I, O>>);

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
#[allow(clippy::type_complexity)]
impl<I, O, T, F> TokioRuntime<I, O>
where
    I: JobInput + Send + 'static,
    O: Send + 'static,
    T: JobInfo + Send + 'static,
    F: Iterator<Item = T> + Send + 'static,
    <I as JobInput>::Chunk: Send,
    for<'a> &'a I: IntoIterator<Item = T, IntoIter = F>,
{
    pub async fn new<J>(bundle: impl Dispatcher<J, Info = T> + Send + Clone + 'static) -> Self
    where
        J: Job<Input = I, Output = O> + Send + 'static,
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
        model: impl Dispatcher<J, Info = T> + Send + Clone + 'static,
        mut receiver: tokio::sync::mpsc::Receiver<Submission<I, O>>,
    ) where
        J: Job<Input = I, Output = O> + Send + 'static,
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

            async fn back<J: Job, I: JobInput>(
                job: J,
                mut input: I,
                sender: flume::Sender<Result<(I, J::Output), RuntimeError>>,
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
    pub async fn infer(&self, input: I) -> Result<(I, O), RuntimeError> {
        let (sender, receiver) = flume::bounded(1);
        let submission = Submission { input, sender };
        let _ = self.0.send(submission).await;
        receiver.recv_async().await?
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
    pub fn new(bundle: M) -> Self {
        Self {
            model: bundle,
            _phantom: PhantomData,
        }
    }

    pub async fn infer(&self, mut input: I) -> Result<(I, O), RuntimeError> {
        let Some(info) = (&input).into_iter().next() else {
            return Err(RuntimeError::InputExhausted);
        };
        let chunk = input.chunk();

        let mut job = self.model.dispatch(info)?.load(&chunk)?;
        job.submit();

        let output = job.back().await?;
        input.step();

        Ok((input, output))
    }
}

pub trait Runtime {
    #[cfg(not(target_arch = "wasm32"))]
    fn infer(
        &self,
        input: infer::InferInput,
    ) -> BoxFuture<Result<(infer::InferInput, infer::InferOutput), RuntimeError>>;

    #[cfg(target_arch = "wasm32")]
    fn infer(
        &self,
        input: infer::InferInput,
    ) -> LocalBoxFuture<Result<(infer::InferInput, infer::InferOutput), RuntimeError>>;
}

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
#[allow(clippy::type_complexity)]
impl Runtime for TokioRuntime<infer::InferInput, infer::InferOutput> {
    #[cfg(not(target_arch = "wasm32"))]
    fn infer(
        &self,
        input: infer::InferInput,
    ) -> BoxFuture<Result<(infer::InferInput, infer::InferOutput), RuntimeError>> {
        Box::pin(self.infer(input))
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::type_complexity)]
impl<M, J> Runtime for SimpleRuntime<M, J, infer::InferInput, infer::InferOutput>
where
    J: Job<Input = infer::InferInput, Output = infer::InferOutput> + Send + Sync,
    M: Dispatcher<J, Info = infer::InferInfo> + Send + Sync,
{
    fn infer(
        &self,
        input: infer::InferInput,
    ) -> BoxFuture<Result<(infer::InferInput, infer::InferOutput), RuntimeError>> {
        Box::pin(self.infer(input))
    }
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::type_complexity)]
impl<M, J> Runtime for SimpleRuntime<M, J, infer::InferInput, infer::InferOutput>
where
    J: Job<Input = infer::InferInput, Output = infer::InferOutput>,
    M: Dispatcher<J, Info = infer::InferInfo>,
{
    fn infer(
        &self,
        input: infer::InferInput,
    ) -> LocalBoxFuture<Result<(infer::InferInput, infer::InferOutput), RuntimeError>> {
        Box::pin(self.infer(input))
    }
}
