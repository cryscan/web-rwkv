use crate::runtime::{JobInfo, JobInput};

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
    fn check(&self, _info: &Self) -> bool {
        todo!()
    }
}

pub struct EncoderInput;

impl JobInput for EncoderInput {
    type Chunk = ();

    fn step(&mut self) {
        todo!()
    }

    fn chunk(&self) -> Self::Chunk {
        todo!()
    }
}

pub struct EncoderOutput;
