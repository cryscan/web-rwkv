use std::{borrow::Cow, future::Future};

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use regex::Regex;
use safetensors::{Dtype, SafeTensorError, SafeTensors};
use web_rwkv_derive::{Deref, DerefMut};

use super::model::{ModelError, ModelInfo, ModelVersion, Quant};
use crate::{
    context::Context,
    num::Scalar,
    tensor::{
        kind::ReadWrite,
        matrix::Matrix,
        ops::{TensorCommand, TensorOp, TensorPass},
        shape::{Shape, TensorDimension},
        TensorCpu, TensorError, TensorGpu, TensorInit, TensorInto, TensorReshape, TensorShape,
    },
};

pub type ReaderTensor<'a> = (Dtype, Vec<usize>, Cow<'a, [u8]>);

/// Interface accessing a safetensors data blob.
#[trait_variant::make(ReaderSend: Send)]
pub trait Reader {
    fn names(&self) -> Vec<&str>;
    fn contains(&self, name: &str) -> bool;
    fn shape(&self, name: &str) -> Result<Vec<usize>, SafeTensorError>;
    fn tensor(&self, name: &str) -> impl Future<Output = Result<ReaderTensor, SafeTensorError>>;
}

impl ReaderSend for SafeTensors<'_> {
    #[inline]
    fn names(&self) -> Vec<&str> {
        self.names().into_iter().map(AsRef::as_ref).collect()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.names().contains(&&name.to_string())
    }

    #[inline]
    fn shape(&self, name: &str) -> Result<Vec<usize>, SafeTensorError> {
        Ok(self.tensor(name)?.shape().to_vec())
    }

    #[inline]
    async fn tensor(&self, name: &str) -> Result<ReaderTensor, SafeTensorError> {
        let tensor = self.tensor(name)?;
        let shape = tensor.shape().to_vec();
        let data = tensor.data().into();
        Ok((tensor.dtype(), shape, data))
    }
}

pub trait TensorFromReader<T: Scalar> {
    /// Create a tensor from safetensors reader.
    fn from_reader(reader: ReaderTensor) -> Result<TensorCpu<T>, TensorError>;
}

impl<T: Scalar> TensorFromReader<T> for TensorCpu<T> {
    fn from_reader((dt, shape, data): ReaderTensor) -> Result<Self, TensorError> {
        if T::DATA_TYPE != dt {
            return Err(TensorError::Type);
        }
        let shape = Shape::from_slice_rev(&shape)?;
        match data {
            Cow::Borrowed(data) => Self::from_data(shape, bytemuck::cast_slice(data)),
            Cow::Owned(data) => {
                let data = bytemuck::cast_slice(&data);
                let data = Cow::Owned(data.to_vec());
                Self::from_data(shape, data)
            }
        }
    }
}

/// A LoRA that adds to the model when loading.
#[derive(Clone)]
pub struct Lora<R> {
    /// Binary safetensors LoRA content.
    pub data: R,
    /// A list of LoRA blend patterns.
    /// A blend pattern is a regex that matches the name of multiple tensors, and a blend factor.
    /// When applying the patterns, they are applied in order.
    pub blend: LoraBlend,
}

/// A list of LoRA blend patterns.
#[derive(Debug, Default, Clone, Deref, DerefMut)]
pub struct LoraBlend(pub Vec<LoraBlendPattern>);

impl LoraBlend {
    /// Build a blend pattern that replaces all vectors, and adds to all matrices with `alpha`.
    #[inline]
    pub fn full(alpha: f32) -> Self {
        Self::default().add_nominal(1.0).add_matrices(alpha)
    }

    /// Add a blend pattern that interpolates tensors with factor `alpha` from 0 to 1.
    #[inline]
    pub fn add_nominal(mut self, alpha: f32) -> Self {
        let pattern = LoraBlendPattern::new(r".+", alpha).unwrap();
        self.push(pattern);
        self
    }

    /// Add a blend pattern that adds to all matrices with `alpha`.
    #[inline]
    pub fn add_matrices(mut self, alpha: f32) -> Self {
        let pattern = LoraBlendPattern::new(
            r"blocks\.([0-9]+)\.(att|ffn)\.(key|value|receptance|gate|output)\.weight",
            alpha,
        )
        .unwrap();
        self.push(pattern);
        self
    }

    /// Add a blend pattern that interpolates tensors in a layer with factor `alpha` from 0 to 1.
    pub fn add_layer_nominal(mut self, layer: usize, alpha: f32) -> Self {
        let pattern = format!(r"blocks\.{layer}");
        let pattern = LoraBlendPattern::new(&pattern, alpha).unwrap();
        self.push(pattern);
        self
    }

    /// Add a blend pattern that adds to all matrices in a layer with `alpha`.
    pub fn add_layer_matrices(mut self, layer: usize, alpha: f32) -> Self {
        let pattern =
            format!(r"blocks\.{layer}\.(att|ffn)\.(key|value|receptance|gate|output)\.weight");
        let pattern = LoraBlendPattern::new(&pattern, alpha).unwrap();
        self.push(pattern);
        self
    }
}

/// A blend pattern is a regex that matches the name of multiple tensors, and a blend factor.
#[derive(Debug, Clone)]
pub struct LoraBlendPattern {
    /// A regex pattern that matches tensors in the model.
    pattern: Regex,
    /// The blend factor.
    alpha: f32,
}

impl LoraBlendPattern {
    #[inline]
    pub fn new(pattern: &str, alpha: f32) -> Result<Self> {
        Ok(Self {
            pattern: Regex::new(pattern)?,
            alpha,
        })
    }

    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

struct LoraVector {
    tensor: TensorGpu<f16, ReadWrite>,
    alpha: f32,
}

struct LoraMatrix {
    x: TensorGpu<f16, ReadWrite>,
    y: TensorGpu<f16, ReadWrite>,
    rank: usize,
    alpha: f32,
}

#[derive(Clone)]
pub struct Loader<R> {
    pub context: Context,
    pub model: R,
    pub lora: Vec<Lora<R>>,
}

impl<R: Reader> Loader<R> {
    pub fn info(model: &R) -> Result<ModelInfo> {
        let num_layer = {
            let mut r: usize = 0;
            for i in model.names() {
                const PREFIX: &str = "blocks.";
                if let Some(i) = i.strip_prefix(PREFIX) {
                    let i = &i[..i.find('.').unwrap_or(0)];
                    r = r.max(i.parse::<usize>()?)
                }
            }
            r + 1
        };

        let embed = model.shape("emb.weight")?;
        let ffn = model.shape("blocks.0.ffn.key.weight")?;
        let time_first = model.shape("blocks.0.att.time_first")?;

        let v5 = [
            "blocks.0.att.gate.weight",
            "blocks.0.att.ln_x.weight",
            "blocks.0.att.ln_x.bias",
        ]
        .into_iter()
        .all(|name| model.contains(name));
        let v6 = [
            "blocks.0.att.time_mix_x",
            "blocks.0.att.time_mix_w",
            "blocks.0.att.time_mix_k",
            "blocks.0.att.time_mix_v",
            "blocks.0.att.time_mix_r",
            "blocks.0.att.time_mix_g",
            "blocks.0.att.time_mix_w1",
            "blocks.0.att.time_mix_w2",
            "blocks.0.att.time_decay_w1",
            "blocks.0.att.time_decay_w2",
            "blocks.0.ffn.time_mix_k",
            "blocks.0.ffn.time_mix_r",
        ]
        .into_iter()
        .all(|name| model.contains(name));

        let version = match (v5, v6) {
            (false, false) => ModelVersion::V4,
            (true, false) => ModelVersion::V5,
            (true, true) => ModelVersion::V6,
            _ => return Err(ModelError::InvalidVersion.into()),
        };

        let num_emb = embed[1];
        let num_hidden = ffn[0];
        let num_vocab = embed[0];
        let num_head = time_first[0];

        let time_mix_adapter_size = model
            .shape("blocks.0.att.time_mix_w1")
            .map(|shape| shape[0] / 5)
            .unwrap_or_default();
        let time_decay_adapter_size = model
            .shape("blocks.0.att.time_decay_w1")
            .map(|shape| shape[0])
            .unwrap_or_default();

        Ok(ModelInfo {
            version,
            num_layer,
            num_emb,
            num_hidden,
            num_vocab,
            num_head,
            time_mix_adapter_size,
            time_decay_adapter_size,
        })
    }

    /// Load all lora and blend factors about the vector with a given name.
    /// In each LoRA, only the last matched pattern is loaded.
    async fn lora_vectors(&self, name: impl AsRef<str>) -> Result<Vec<LoraVector>> {
        let context = &self.context;
        let name = name.as_ref();

        let mut vectors = vec![];
        for lora in self.lora.iter() {
            let Some(blend) = lora
                .blend
                .iter()
                .filter(|blend| blend.pattern.is_match(name))
                .last()
            else {
                continue;
            };

            let Ok(tensor) = lora.data.tensor(name).await else {
                continue;
            };
            let tensor = TensorCpu::<f16>::from_reader(tensor)?.transfer_into(context);
            let alpha = blend.alpha;
            vectors.push(LoraVector { tensor, alpha });

            log::info!("vector (LoRA) {name}, alpha: {alpha}");
        }
        Ok(vectors)
    }

    /// Load all lora and blend factors about the matrix with a given name.
    /// In each LoRA, only the last matched pattern is loaded.
    async fn lora_matrices(&self, name: impl AsRef<str>) -> Result<Vec<LoraMatrix>> {
        let context = &self.context;
        let name = name.as_ref();

        let mut matrices = vec![];
        for lora in self.lora.iter() {
            let Some(blend) = lora
                .blend
                .iter()
                .filter(|blend| blend.pattern.is_match(name))
                .last()
            else {
                continue;
            };

            let name = name.split('.').filter(|x| !x.contains("weight")).join(".");
            let Ok(x) = lora.data.tensor(&format!("{name}.lora.0")).await else {
                continue;
            };
            let Ok(y) = lora.data.tensor(&format!("{name}.lora.1")).await else {
                continue;
            };

            let rank = x.1[1];
            let alpha = blend.alpha;
            let x = TensorCpu::from_reader(x)?.transfer_into(context);
            let y = TensorCpu::from_reader(y)?.transfer_into(context);
            matrices.push(LoraMatrix { x, y, rank, alpha });

            log::info!("matrix (LoRA) {name}, alpha: {alpha}, rank: {rank}");
        }
        Ok(matrices)
    }

    pub fn tensor_shape(&self, name: impl AsRef<str>) -> Result<Shape> {
        let shape = self.model.shape(name.as_ref())?;
        Ok(Shape::from_slice_rev(&shape)?)
    }

    pub async fn load_vector_f32(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f32, ReadWrite>> {
        use TensorDimension::{Auto, Dimension};
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref()).await?;
        let tensor: TensorGpu<_, _> = TensorCpu::<f16>::from_reader(tensor)?
            .map(|x| x.to_f32())
            .reshape(Auto, Dimension(1), Dimension(1), Dimension(1))?
            .transfer_into(context);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        for lora in self.lora_vectors(name).await? {
            let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

            let shape = lora.tensor.shape();
            let tensor = tensor.reshape(
                Dimension(shape[0]),
                Dimension(shape[1]),
                Dimension(shape[2]),
                Dimension(shape[3]),
            )?;

            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }
        self.context.queue.submit(Some(encoder.finish()));
        Ok(tensor)
    }

    pub async fn load_vector_exp_f32(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f32, ReadWrite>> {
        use TensorDimension::{Auto, Dimension};
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref()).await?;
        let tensor: TensorGpu<_, _> = TensorCpu::<f16>::from_reader(tensor)?
            // .map(|x| -x.to_f32().exp())
            .map(|x| x.to_f32())
            .reshape(Auto, Dimension(1), Dimension(1), Dimension(1))?
            .transfer_into(context);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        for lora in self.lora_vectors(name).await? {
            let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

            let shape = lora.tensor.shape();
            let tensor = tensor.reshape(
                Dimension(shape[0]),
                Dimension(shape[1]),
                Dimension(shape[2]),
                Dimension(shape[3]),
            )?;

            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }

        let op = TensorOp::opposite_exp(&tensor)?;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        self.context.queue.submit(Some(encoder.finish()));
        Ok(tensor)
    }

    pub async fn load_vector_exp_exp_f32(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f32, ReadWrite>> {
        use TensorDimension::{Auto, Dimension};
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref()).await?;
        let tensor: TensorGpu<_, _> = TensorCpu::<f16>::from_reader(tensor)?
            // .map(|x| -x.to_f32().exp())
            // .map(|x| x.exp())
            .map(|x| x.to_f32())
            .reshape(Auto, Dimension(1), Dimension(1), Dimension(1))?
            .transfer_into(context);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        for lora in self.lora_vectors(name).await? {
            let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

            let shape = lora.tensor.shape();
            let tensor = tensor.reshape(
                Dimension(shape[0]),
                Dimension(shape[1]),
                Dimension(shape[2]),
                Dimension(shape[3]),
            )?;

            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }

        let op = TensorOp::stable_exp(&tensor)?;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        self.context.queue.submit(Some(encoder.finish()));
        Ok(tensor)
    }

    pub async fn load_vector_f16(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f16, ReadWrite>> {
        use TensorDimension::{Auto, Dimension};
        let context = &self.context;
        let lora = self.lora_vectors(name.as_ref()).await?;
        let tensor = self.model.tensor(name.as_ref()).await?;
        let tensor = if lora.is_empty() {
            TensorCpu::from_reader(tensor)?
                .reshape(Auto, Dimension(1), Dimension(1), Dimension(1))?
                .transfer_into(context)
        } else {
            let tensor_f32: TensorGpu<f32, _> = TensorCpu::<f16>::from_reader(tensor)?
                .map(|x| x.to_f32())
                .reshape(Auto, Dimension(1), Dimension(1), Dimension(1))?
                .transfer_into(context);
            let tensor_f16: TensorGpu<f16, _> = context.tensor_init(tensor_f32.shape());

            let mut encoder = context.device.create_command_encoder(&Default::default());
            for lora in lora {
                let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
                let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

                let shape = lora.tensor.shape();
                let tensor = tensor_f32.reshape(
                    Dimension(shape[0]),
                    Dimension(shape[1]),
                    Dimension(shape[2]),
                    Dimension(shape[3]),
                )?;

                let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.execute_tensor_op(&op);
            }

            let op = TensorOp::blit(
                tensor_f32.view(.., .., .., ..)?,
                tensor_f16.view(.., .., .., ..)?,
            )?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
            drop(pass);

            context.queue.submit(Some(encoder.finish()));
            tensor_f16
        };
        Ok(tensor)
    }

    pub async fn load_matrix_f16(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f16, ReadWrite>> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref()).await?;
        let tensor: TensorGpu<_, _> = TensorCpu::from_reader(tensor)?.transfer_into(context);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        for lora in self.lora_matrices(name.as_ref()).await? {
            let factor = vec![lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(
                &factor,
                lora.x.view(.., .., .., ..)?,
                lora.y.view(.., .., .., ..)?,
                tensor.view(.., .., .., ..)?,
            )?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }
        for lora in self.lora_vectors(name.as_ref()).await? {
            let factor = vec![lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }
        context.queue.submit(Some(encoder.finish()));

        Ok(tensor)
    }

    pub async fn load_matrix_f16_discount(
        &self,
        name: impl AsRef<str>,
        discount: f32,
    ) -> Result<TensorGpu<f16, ReadWrite>> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref()).await?;
        let tensor: TensorGpu<_, _> = TensorCpu::<f16>::from_reader(tensor)?
            .map(|x| f16::from_f32(discount * x.to_f32()))
            .transfer_into(context);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        for lora in self.lora_matrices(name.as_ref()).await? {
            let factor = vec![discount * lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(
                &factor,
                lora.x.view(.., .., .., ..)?,
                lora.y.view(.., .., .., ..)?,
                tensor.view(.., .., .., ..)?,
            )?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }
        for lora in self.lora_vectors(name.as_ref()).await? {
            let factor = vec![discount * lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }
        context.queue.submit(Some(encoder.finish()));

        Ok(tensor)
    }

    pub async fn load_in_place_matrix_f16(
        &self,
        matrix: &TensorGpu<f16, ReadWrite>,
        name: impl AsRef<str>,
    ) -> Result<()> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref()).await?;
        let tensor = TensorCpu::from_reader(tensor)?;
        matrix.load(&tensor)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        for lora in self.lora_matrices(name.as_ref()).await? {
            let factor = vec![lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(
                &factor,
                lora.x.view(.., .., .., ..)?,
                lora.y.view(.., .., .., ..)?,
                matrix.view(.., .., .., ..)?,
            )?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }
        for lora in self.lora_vectors(name.as_ref()).await? {
            let factor = vec![lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, matrix)?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }
        context.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    pub async fn load_in_place_matrix_f16_discount(
        &self,
        matrix: &TensorGpu<f16, ReadWrite>,
        name: impl AsRef<str>,
        discount: f32,
    ) -> Result<()> {
        use TensorDimension::{Dimension, Full};
        let context = &self.context;

        let tensor = self.model.tensor(name.as_ref()).await?;
        let tensor = TensorCpu::<f16>::from_reader(tensor)?
            .map(|x| f16::from_f32(discount * x.to_f32()))
            .reshape(Full, Full, Dimension(1), Dimension(1))?;
        matrix.load(&tensor)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        for lora in self.lora_matrices(name.as_ref()).await? {
            let factor = vec![discount * lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(
                &factor,
                lora.x.view(.., .., .., ..)?,
                lora.y.view(.., .., .., ..)?,
                matrix.view(.., .., .., ..)?,
            )?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }
        for lora in self.lora_vectors(name.as_ref()).await? {
            let factor = vec![discount * lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, matrix)?;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.execute_tensor_op(&op);
        }
        context.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    pub async fn load_embed(&self) -> Result<TensorCpu<f16>> {
        let context = &self.context;
        let name = "emb.weight";

        let (dt, shape, tensor) = self.model.tensor(name).await?;
        let lora = self.lora_vectors(name).await?;

        if lora.is_empty() {
            let tensor = TensorCpu::from_reader((dt, shape, tensor))?;
            Ok(tensor)
        } else {
            let tensor = TensorCpu::from_reader((dt, shape, tensor))?.transfer_into(context);
            let mut encoder = context.device.create_command_encoder(&Default::default());
            for lora in lora {
                let factor = vec![lora.alpha, 1.0, 0.0, 0.0];
                let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
                let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.execute_tensor_op(&op);
            }

            let map = context.tensor_init(tensor.shape());
            encoder.copy_tensor(&tensor, &map)?;

            context.queue.submit(Some(encoder.finish()));
            Ok(map.back().await)
        }
    }

    pub async fn load_head(&self, chunk_size: usize) -> Result<Vec<TensorGpu<f16, ReadWrite>>> {
        let context = &self.context;
        let (_, shape, tensor) = self.model.tensor("head.weight").await?;
        let shape = Shape::new(shape[1], shape[0], 1, 1);
        let chunks = (shape[1] + chunk_size - 1) / chunk_size;
        let data = bytemuck::cast_slice(&tensor);

        let head = (0..chunks)
            .map(|chunk| {
                let real_chunk_size = ((chunk + 1) * chunk_size).min(shape[1]) - chunk * chunk_size;
                let start = (chunk * chunk_size) * shape[0];
                let end = start + real_chunk_size * shape[0];
                context.tensor_from_data([shape[0], real_chunk_size, 1, 1], &data[start..end])
            })
            .try_collect()?;
        Ok(head)
    }

    pub async fn load_matrix(&self, name: String, quant: Quant) -> Result<Matrix> {
        let context = &self.context;
        match quant {
            Quant::None => Ok(Matrix::Fp16(self.load_matrix_f16(name).await?)),
            Quant::Int8 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16(&buffer, &name).await?;
                Ok(Matrix::quant_u8(&buffer)?)
            }
            Quant::NF4 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16(&buffer, &name).await?;
                Ok(Matrix::quant_nf4(&buffer)?)
            }
        }
    }

    pub async fn load_matrix_discount(
        &self,
        name: String,
        quant: Quant,
        discount: f32,
    ) -> Result<Matrix> {
        let context = &self.context;
        match quant {
            Quant::None => Ok(Matrix::Fp16(
                self.load_matrix_f16_discount(name, discount).await?,
            )),
            Quant::Int8 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16_discount(&buffer, &name, discount)
                    .await?;
                Ok(Matrix::quant_u8(&buffer)?)
            }
            Quant::NF4 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16_discount(&buffer, &name, discount)
                    .await?;
                Ok(Matrix::quant_nf4(&buffer)?)
            }
        }
    }
}
