use web_rwkv_derive::{Deref, DerefMut};

use crate::{
    runtime::{JobInfo, JobInput},
    tensor::TensorCpu,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct Encoder;

impl super::Infer for Encoder {
    type Info = EncoderInfo;
    type Input = EncoderInput;
    type Output = EncoderOutput;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderInfo {
    pub len: usize,
}

impl JobInfo for EncoderInfo {
    fn check(&self, info: &Self) -> bool {
        self.eq(info)
    }
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct EncoderInput(pub Vec<super::Token>);

impl JobInput for EncoderInput {
    type Chunk = Self;

    fn step(&mut self) {}

    fn chunk(&self) -> Self::Chunk {
        self.clone()
    }
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct EncoderOutput(pub TensorCpu<f32>);
