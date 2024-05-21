use std::future::Future;

use anyhow::Result;

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
struct Submission<I, O> {
    input: I,
    sender: tokio::sync::oneshot::Sender<(I, O)>,
}

pub trait JobInput: Send + 'static {
    /// One chunk of the whole input at a step.
    type Chunk: Send + 'static;

    /// Advance the input for a step.
    fn step(&mut self);
    /// The current step's chunk to feed into the job.
    fn chunk(&self) -> Self::Chunk;
}

#[derive(Debug, Clone)]
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
                for (key, handle) in queue.drain(..) {
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
                let iter = iter.as_mut().expect("iter should be assigned");

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
                    let builder = builder.clone();
                    let handle = tokio::task::spawn_blocking(move || builder.build(key));
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
                sender: tokio::sync::oneshot::Sender<(I, J::Output)>,
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
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let submission = Submission { input, sender };
        let _ = self.0.send(submission).await;
        receiver.await.expect("receive infer output error")
    }
}
