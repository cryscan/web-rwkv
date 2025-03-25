use web_rwkv_derive::{Deref, DerefMut};

use crate::{
    runtime::{JobInfo, JobInput},
    tensor::{TensorCpu, TensorShape},
};

#[derive(Debug, Default, Clone, Copy)]
pub struct Vision;

impl super::Infer for Vision {
    type Info = VisionInfo;
    type Input = VisionInput;
    type Output = VisionOutput;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deref, DerefMut)]
pub struct VisionInfo([usize; 4]);

impl JobInfo for VisionInfo {
    fn check(&self, info: &Self) -> bool {
        self.eq(info)
    }
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct VisionInput(pub TensorCpu<f32>);

impl VisionInput {
    #[inline]
    pub fn x(&self) -> usize {
        self.0.shape()[0]
    }

    #[inline]
    pub fn y(&self) -> usize {
        self.0.shape()[1]
    }

    #[inline]
    pub fn num_channel(&self) -> usize {
        self.0.shape()[2]
    }

    #[inline]
    pub fn num_patch(&self) -> usize {
        self.0.shape()[3]
    }

    #[inline]
    pub fn num_emb(&self) -> usize {
        self.x() * self.y() * self.num_channel()
    }

    #[inline]
    pub fn info(&self) -> VisionInfo {
        VisionInfo(self.shape().into())
    }
}

impl JobInput for VisionInput {
    type Chunk = Self;

    fn step(&mut self) {}

    fn chunk(&self) -> Self::Chunk {
        self.clone()
    }
}

impl Iterator for &VisionInput {
    type Item = VisionInfo;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.info())
    }
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct VisionOutput(pub TensorCpu<f32>);
