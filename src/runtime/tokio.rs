use super::{Run, Softmax};

pub struct Runtime<Model, State> {
    model: Model,
    state: State,
}

impl<Model, State> Runtime<Model, State>
where
    Runtime<Model, State>: Run + Softmax,
{
    pub fn new(model: Model, state: State) -> Self {
        Self { model, state }
    }
}
