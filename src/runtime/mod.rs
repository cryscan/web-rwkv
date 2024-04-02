use std::future::Future;

#[cfg(feature = "tokio")]
pub mod tokio;

pub trait Job {
    type Input;
    type Output;
    type Error;

    fn load(&self, input: &Self::Input) -> Result<(), Self::Error>;
    fn submit(self) -> impl Future<Output = Self::Output> + Send + 'static;
}

#[derive(Debug, Default, Clone)]
pub struct RunInput(pub Vec<(usize, RunOutput)>);

impl RunInput {
    pub fn num_token(&self) -> usize {
        self.0.iter().map(|(x, _)| x).sum()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum RunOutput {
    #[default]
    None,
    Last,
    Full,
}

pub trait Run {
    fn run(&self, input: RunInput) -> impl Job;
}

#[derive(Debug, Default, Clone)]
pub struct SoftmaxInput(pub Vec<usize>);

pub trait Softmax {
    fn softmax(&self, input: SoftmaxInput) -> impl Job;
}
