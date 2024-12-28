use std::{hash::Hash, sync::Arc};

use embed_doc_image::embed_doc_image;
use half::f16;
use serde::{Deserialize, Serialize};
use wgpu::{BindGroup, CommandBuffer, CommandEncoder, ComputePass};

use super::{
    kind::{Kind, ReadWrite, Uniform},
    Shape, TensorError, TensorGpu, TensorGpuView, TensorScalar, TensorShape,
};
use crate::{
    context::{BindGroupBuilder, CachedPipeline, Macros, PipelineKey},
    num::{Float, Scalar},
    tensor::{shape::TensorDimension, TensorReshape},
};

pub trait TensorCommand<T: Scalar, K: Kind> {
    fn copy_tensor(
        &mut self,
        source: &TensorGpu<T, K>,
        destination: &TensorGpu<T, K>,
    ) -> Result<(), TensorError>;

    fn copy_tensor_batch(
        &mut self,
        source: &TensorGpu<T, K>,
        destination: &TensorGpu<T, K>,
        from: usize,
        to: usize,
    ) -> Result<(), TensorError>;
}

impl<T: Scalar, K: Kind> TensorCommand<T, K> for CommandEncoder {
    fn copy_tensor(
        &mut self,
        source: &TensorGpu<T, K>,
        destination: &TensorGpu<T, K>,
    ) -> Result<(), TensorError> {
        destination.check_shape(source.shape())?;
        let size = destination.size() as u64;
        self.copy_buffer_to_buffer(&source.buffer, 0, &destination.buffer, 0, size);
        Ok(())
    }

    fn copy_tensor_batch(
        &mut self,
        source: &TensorGpu<T, K>,
        destination: &TensorGpu<T, K>,
        from: usize,
        to: usize,
    ) -> Result<(), TensorError> {
        source.check_shape([source.shape[0], source.shape[1], source.shape[2], 1])?;
        destination.check_shape([source.shape[0], source.shape[1], destination.shape[2], 1])?;
        if from >= source.shape[2] {
            return Err(TensorError::BatchOutOfRange {
                batch: from,
                max: source.shape[2],
            });
        }
        if to > destination.shape[2] {
            return Err(TensorError::BatchOutOfRange {
                batch: to,
                max: destination.shape[2],
            });
        }
        self.copy_buffer_to_buffer(
            &source.buffer,
            (T::size() * source.shape[0] * source.shape[1] * from) as u64,
            &destination.buffer,
            (T::size() * destination.shape[0] * destination.shape[1] * to) as u64,
            (T::size() * source.shape[0] * source.shape[1]) as u64,
        );
        Ok(())
    }
}

impl crate::context::Context {
    pub fn encode(&self, op: &TensorOp) -> Vec<CommandBuffer> {
        struct Atom<'a> {
            pipeline: &'a CachedPipeline,
            bindings: &'a [Arc<BindGroup>],
            dispatch: &'a [u32; 3],
        }

        fn dispatch<'b, 'a: 'b>(
            pass: &'b mut ComputePass<'a>,
            Atom {
                pipeline,
                bindings,
                dispatch,
            }: Atom<'a>,
        ) {
            pass.set_pipeline(&pipeline.pipeline);
            for (index, bind) in bindings.iter().enumerate() {
                pass.set_bind_group(index as u32, &**bind, &[]);
            }
            pass.dispatch_workgroups(dispatch[0], dispatch[1], dispatch[2]);
        }

        fn flatten<'b, 'a: 'b>(
            commands: &'b mut Vec<Vec<Atom<'a>>>,
            passes: &'b mut Vec<Atom<'a>>,
            op: &'a TensorOp,
        ) {
            match op {
                TensorOp::Atom {
                    pipeline,
                    bindings,
                    dispatch,
                } => passes.push(Atom {
                    pipeline,
                    bindings,
                    dispatch,
                }),
                TensorOp::List(ops) => ops.iter().for_each(|op| flatten(commands, passes, op)),
                TensorOp::Sep => {
                    let mut temp = vec![];
                    std::mem::swap(&mut temp, passes);
                    commands.push(temp);
                }
            }
        }

        let mut commands = vec![];
        let mut passes = vec![];
        flatten(&mut commands, &mut passes, op);
        commands.push(passes);

        commands
            .into_iter()
            .filter(|atoms| !atoms.is_empty())
            .map(|atoms| {
                let mut encoder = self.device.create_command_encoder(&Default::default());
                let mut pass = encoder.begin_compute_pass(&Default::default());
                for atom in atoms {
                    dispatch(&mut pass, atom);
                }
                drop(pass);
                encoder.finish()
            })
            .collect()
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    #[default]
    #[serde(rename = "")]
    None,
    SquaredRelu,
    Tanh,
    StableExp,
    OppositeExp,
    Softplus,
    Sigmoid,
    Silu,
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(serde_variant::to_variant_name(self).unwrap())
    }
}

impl Macros {
    /// Define a `u32` macro `NF4_BLOCK_SIZE`.
    pub fn nf4(mut self, block_size: u32) -> Self {
        self.insert("NF4_BLOCK_SIZE".into(), format!("{}u", block_size));
        self
    }

    /// Define a `u32` macro `NF4_BLOCK_SIZE`.
    pub fn int8(mut self, block_size: u32) -> Self {
        self.insert("INT8_BLOCK_SIZE".into(), format!("{}u", block_size));
        self
    }

    /// Define a `f32` macro with a given name.
    pub fn f32(mut self, name: impl Into<String>, value: f32) -> Self {
        self.insert(name.into(), format!("{}", value));
        self
    }

    /// Define a `usize` macro with a given name.
    pub fn u32(mut self, name: impl Into<String>, value: u32) -> Self {
        self.insert(name.into(), format!("{}u", value));
        self
    }

    /// Define a `bool` macro with a given name.
    pub fn bool(mut self, name: impl Into<String>, value: bool) -> Self {
        match value {
            true => {
                self.insert(name.into(), Default::default());
                self
            }
            false => self,
        }
    }

    pub fn activate(mut self, name: impl Into<String>, value: Activation) -> Self {
        const ACTIVATION_DEFINE: &str = "
fn squared_relu(x: vec4<f32>) -> vec4<f32> {
    let p = max(x, vec4<f32>(0.0));
    return p * p;
}

fn stable_exp(x: vec4<f32>) -> vec4<f32> {
    return exp(-exp(x));
}

fn opposite_exp(x: vec4<f32>) -> vec4<f32> {
    return -exp(x);
}

fn softplus(x: vec4<f32>) -> vec4<f32> {
    return log(1.0 + exp(x));
}

fn sigmoid(x: vec4<f32>) -> vec4<f32> {
    return 1.0 / (1.0 + exp(-x));
}

fn silu(x: vec4<f32>) -> vec4<f32> {
    return x / (1.0 + exp(-x));
}
";
        self.insert("ACTIVATION_DEFINE".into(), ACTIVATION_DEFINE.to_string());
        self.insert(name.into(), value.to_string());
        self
    }

    /// Define the macro specifies input/output tensor data type.
    pub fn tensor<T: Float>(
        mut self,
        _tensor: &impl TensorScalar<T = T>,
        prefix: Option<&'_ str>,
    ) -> Self {
        match prefix {
            None => self.insert(T::DEF.into(), Default::default()),
            Some(prefix) => self.insert(format!("{}_{}", prefix, T::DEF), Default::default()),
        };
        self
    }

    /// Define a macro with custom display name and prefix.
    pub fn custom(mut self, value: impl std::fmt::Display, prefix: Option<&'_ str>) -> Self {
        match prefix {
            None => self.insert(format!("{}", value), Default::default()),
            Some(prefix) => self.insert(format!("{}_{}", prefix, value), Default::default()),
        };
        self
    }

    /// Add a define when `condition` is true.
    pub fn define(mut self, name: impl Into<String>, condition: bool) -> Self {
        if condition {
            self.insert(name.into(), Default::default());
        }
        self
    }

    /// Add subgroup defines.
    #[cfg(feature = "subgroup-ops")]
    pub fn subgroup(self, min: u32, max: u32) -> Self {
        self.u32("MIN_SUBGROUP_SIZE", min)
            .u32("MAX_SUBGROUP_SIZE", max)
            .define(format!("SUBGROUP_SIZE_{}_{}", min, max), true)
    }
}

pub enum TensorOp {
    Atom {
        pipeline: Arc<CachedPipeline>,
        bindings: Vec<Arc<BindGroup>>,
        dispatch: [u32; 3],
    },
    List(Vec<TensorOp>),
    Sep,
}

impl TensorOp {
    pub const NF4_BLOCK_SIZE: u32 = 64;
    pub const INT8_BLOCK_SIZE: u32 = 128;

    #[inline]
    pub fn empty() -> Self {
        Self::List(vec![])
    }

    /// Softmax operator applied on `x`.
    pub fn softmax(x: &TensorGpu<impl Float, ReadWrite>) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let context = x.context();
        let shape = x.shape();

        #[cfg(not(feature = "subgroup-ops"))]
        let key = PipelineKey::new(
            "softmax",
            "softmax",
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        #[cfg(feature = "subgroup-ops")]
        let key = PipelineKey::new(
            "softmax",
            "softmax",
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None),
        );

        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/softmax.wgsl"),
            &[x.meta_layout(0), x.layout(1, false)],
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/subgroup/softmax.wgsl"),
            &[x.meta_layout(0), x.layout(1, false)],
        );
        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(1, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Embedding on GPU.
    /// - `tokens` shape: `[T, B]`.
    /// - `input` shape: `[C, V]`.
    /// - `output` shape: `[C, T, B]`.
    pub fn embed(
        tokens: &TensorGpu<u32, ReadWrite>,
        input: &TensorGpu<f16, ReadWrite>,
        output: &TensorGpu<impl Float, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let context = output.context();
        let shape = {
            let [index, token, batch, _] = output.shape().into();
            let [_, vocab, _, _] = input.shape().into();
            tokens.check_shape([token, batch, 1, 1])?;
            input.check_shape([index, vocab, 1, 1])?;
            output.check_shape([index, token, batch, 1])?;
            output.shape()
        };

        let key = PipelineKey::new(
            "embed",
            "embed",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(output, None),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/embed.wgsl"),
            &[
                output.meta_layout(0),
                tokens.layout(1, true),
                input.layout(2, true),
                output.layout(3, false),
            ],
        );
        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(1, tokens.resource_key())
            .touch(2, input.resource_key())
            .touch(3, output.resource_key())
            .bind(0, output.meta_binding())
            .bind(1, tokens.binding())
            .bind(2, input.binding())
            .bind(3, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    /// Layer normalization applied on `x`, with weight `w` and bias `b`.
    /// - `x` shape: `[C, T, B]`.
    /// - `w` shape: `[C, 1, 1]`.
    /// - `b` shape: `[C, 1, 1]`.
    /// - `s` shape: `[4, T, B]`, mean and inverse std of `x`.
    pub fn layer_norm(
        w: &TensorGpu<f16, ReadWrite>,
        b: &TensorGpu<f16, ReadWrite>,
        x: &TensorGpu<impl Float, ReadWrite>,
        eps: f32,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let context = x.context();
        let shape = {
            let [index, token, batch, _] = x.shape().into();
            x.check_shape([index, token, batch, 1])?;
            w.check_shape([index, 1, 1, 1])?;
            b.check_shape([index, 1, 1, 1])?;
            x.shape()
        };

        let key = PipelineKey::new(
            "layer_norm",
            "layer_norm",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", eps),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/layer_norm.wgsl"),
            &[
                x.meta_layout(0),
                w.layout(1, true),
                b.layout(2, true),
                x.layout(3, false),
            ],
        );
        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(1, w.resource_key())
            .touch(2, b.resource_key())
            .touch(3, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, w.binding())
            .bind(2, b.binding())
            .bind(3, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Group normalization applied on `x`, with weight `w` and bias `b`.
    /// - `x` shape: `[S, H, A]`.
    /// - `w` shape: `[S, H, 1]`.
    /// - `b` shape: `[S, H, 1]`.
    pub fn group_norm(
        w: &TensorGpu<f16, ReadWrite>,
        b: &TensorGpu<f16, ReadWrite>,
        x: &TensorGpu<impl Float, ReadWrite>,
        eps: f32,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 32;

        let context = x.context();
        let shape = {
            let [index, head, token, _] = x.shape().into();
            x.check_shape([index, head, token, 1])?;
            w.check_shape([index, head, 1, 1])?;
            b.check_shape([index, head, 1, 1])?;
            x.shape()
        };

        let key = PipelineKey::new(
            "group_norm",
            "layer_norm",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .bool("GROUP_NORM", true)
                .tensor(x, None)
                .f32("EPS", eps),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/layer_norm.wgsl"),
            &[
                x.meta_layout(0),
                w.layout(1, true),
                b.layout(2, true),
                x.layout(3, false),
            ],
        );
        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(1, w.resource_key())
            .touch(2, b.resource_key())
            .touch(3, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, w.binding())
            .bind(2, b.binding())
            .bind(3, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Recenter `x` to be zero-mean.
    pub fn recenter(x: &TensorGpu<impl Float, ReadWrite>) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let context = x.context();
        let shape = x.shape();

        #[cfg(not(feature = "subgroup-ops"))]
        let key = PipelineKey::new(
            "recenter",
            "recenter",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", 0.0),
        );
        #[cfg(feature = "subgroup-ops")]
        let key = PipelineKey::new(
            "recenter",
            "recenter",
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", 0.0),
        );

        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/normalize.wgsl"),
            &[x.meta_layout(0), x.layout(3, false)],
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/subgroup/normalize.wgsl"),
            &[x.meta_layout(0), x.layout(3, false)],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(3, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Root-mean-square normalization applied on `x`, with weight `w` and bias `b`.
    /// - `x` shape: `[C, T, B]`.
    /// - `w` shape: `[C, 1, 1]`.
    /// - `b` shape: `[C, 1, 1]`.
    pub fn rms_norm(
        w: &TensorGpu<f16, ReadWrite>,
        b: &TensorGpu<f16, ReadWrite>,
        x: &TensorGpu<impl Float, ReadWrite>,
        eps: f32,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let context = x.context();
        let shape = {
            let [index, token, batch, _] = x.shape().into();
            x.check_shape([index, token, batch, 1])?;
            w.check_shape([index, 1, 1, 1])?;
            b.check_shape([index, 1, 1, 1])?;
            x.shape()
        };

        #[cfg(not(feature = "subgroup-ops"))]
        let key = PipelineKey::new(
            "rms_norm",
            "rms_norm",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", eps),
        );
        #[cfg(feature = "subgroup-ops")]
        let key = PipelineKey::new(
            "rms_norm",
            "rms_norm",
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", eps),
        );

        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/normalize.wgsl"),
            &[
                x.meta_layout(0),
                w.layout(1, true),
                b.layout(2, true),
                x.layout(3, false),
            ],
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/subgroup/normalize.wgsl"),
            &[
                x.meta_layout(0),
                w.layout(1, true),
                b.layout(2, true),
                x.layout(3, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(1, w.resource_key())
            .touch(2, b.resource_key())
            .touch(3, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, w.binding())
            .bind(2, b.binding())
            .bind(3, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    /// L2 normalization applied on `x`.
    /// - `x` shape: `[C, T, B]`.
    pub fn l2_norm(x: &TensorGpu<impl Float, ReadWrite>, eps: f32) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let context = x.context();
        let shape = x.shape();

        #[cfg(not(feature = "subgroup-ops"))]
        let key = PipelineKey::new(
            "l2_norm",
            "l2_norm",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", eps),
        );
        #[cfg(feature = "subgroup-ops")]
        let key = PipelineKey::new(
            "l2_norm",
            "l2_norm",
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", eps),
        );

        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/normalize.wgsl"),
            &[x.meta_layout(0), x.layout(3, false)],
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/subgroup/normalize.wgsl"),
            &[x.meta_layout(0), x.layout(3, false)],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(3, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Fp32 matrix-vector multiplication.
    /// - `matrix` shape: `[C, R, B]`.
    /// - `input` shape: `[C, T, B]`.
    /// - `output` shape: `[R, T, B]`.
    pub fn matmul_vec_fp16<'a, 'b, F0: Float, F1: Float>(
        matrix: &TensorGpu<f16, ReadWrite>,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = output.context();
        let shape = {
            let [m, n, b, _] = output.shape().into();
            let [k, _, _, _] = input.shape().into();
            matrix.check_shape([k, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        #[cfg(not(feature = "subgroup-ops"))]
        let key = PipelineKey::new(
            "matmul_vec_fp16",
            "matmul",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT", act),
        );
        #[cfg(feature = "subgroup-ops")]
        let key = PipelineKey::new(
            "matmul_vec_fp16",
            "matmul",
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT", act),
        );

        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/matmul_vec_fp16.wgsl"),
            &[
                matrix.meta_layout(0),
                input.meta_layout(1),
                output.meta_layout(2),
                matrix.layout(3, true),
                input.layout(4, true),
                output.layout(5, false),
            ],
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/subgroup/matmul_vec_fp16.wgsl"),
            &[
                matrix.meta_layout(0),
                input.meta_layout(1),
                output.meta_layout(2),
                matrix.layout(3, true),
                input.layout(4, true),
                output.layout(5, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, matrix.resource_key())
            .touch(4, input.resource_key())
            .touch(5, output.resource_key())
            .bind(0, matrix.meta_binding())
            .bind(1, input.meta_binding())
            .bind(2, output.meta_binding())
            .bind(3, matrix.binding())
            .bind(4, input.binding())
            .bind(5, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [matrix.shape[1] as u32 / 4, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Int8 matrix-vector multiplication.
    /// - `matrix` shape: `[C, R, B]`.
    /// - `input` shape: `[C, T, B]`.
    /// - `output` shape: `[R, T, B]`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_vec_int8<'a, 'b, F0: Float, F1: Float>(
        matrix: &TensorGpu<u8, ReadWrite>,
        minmax: &TensorGpu<f16, ReadWrite>,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = matrix.context();
        let shape = {
            let [m, n, b, _] = output.shape().into();
            let [k, _, _, _] = input.shape().into();
            let len = matrix.shape().len();
            minmax.check_shape([(len << 1).div_ceil(Self::INT8_BLOCK_SIZE as usize), 1, 1, 1])?;
            matrix.check_shape([k, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        #[cfg(not(feature = "subgroup-ops"))]
        let key = PipelineKey::new(
            "matmul_vec_int8",
            "matmul",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .int8(Self::INT8_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT", act),
        );
        #[cfg(feature = "subgroup-ops")]
        let key = PipelineKey::new(
            "matmul_vec_int8",
            "matmul",
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .int8(Self::INT8_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT", act),
        );

        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/matmul_vec_int8.wgsl"),
            &[
                matrix.meta_layout(0),
                input.meta_layout(1),
                output.meta_layout(2),
                matrix.layout(3, true),
                minmax.layout(4, true),
                input.layout(5, true),
                output.layout(6, false),
            ],
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/subgroup/matmul_vec_int8.wgsl"),
            &[
                matrix.meta_layout(0),
                input.meta_layout(1),
                output.meta_layout(2),
                matrix.layout(3, true),
                minmax.layout(4, true),
                input.layout(5, true),
                output.layout(6, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, matrix.resource_key())
            .touch(4, minmax.resource_key())
            .touch(5, input.resource_key())
            .touch(6, output.resource_key())
            .bind(0, matrix.meta_binding())
            .bind(1, input.meta_binding())
            .bind(2, output.meta_binding())
            .bind(3, matrix.binding())
            .bind(4, minmax.binding())
            .bind(5, input.binding())
            .bind(6, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [matrix.shape[1] as u32 / 4, shape[1] as u32, shape[2] as u32],
        })
    }

    /// NFloat4 matrix-vector multiplication.
    /// - `matrix` shape: `[C, R, B]`.
    /// - `input` shape: `[C, T, B]`.
    /// - `output` shape: `[R, T, B]`.
    pub fn matmul_vec_nf4<'a, 'b, F0: Float, F1: Float>(
        matrix: &TensorGpu<u8, ReadWrite>,
        quant: &TensorGpu<f32, Uniform>,
        absmax: &TensorGpu<f16, ReadWrite>,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = matrix.context();
        let shape = {
            let [m, n, b, _] = output.shape().into();
            let [k, _, _, _] = input.shape().into();
            let len = matrix.shape().len() << 1;
            absmax.check_shape([len.div_ceil(Self::NF4_BLOCK_SIZE as usize), 1, 1, 1])?;
            matrix.check_shape([k >> 1, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        #[cfg(not(feature = "subgroup-ops"))]
        let key = PipelineKey::new(
            "matmul_vec_nf4",
            "matmul",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT", act),
        );
        #[cfg(feature = "subgroup-ops")]
        let key = PipelineKey::new(
            "matmul_vec_nf4",
            "matmul",
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT", act),
        );

        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/matmul_vec_nf4.wgsl"),
            &[
                matrix.meta_layout(0),
                input.meta_layout(1),
                output.meta_layout(2),
                quant.layout(3),
                matrix.layout(4, true),
                absmax.layout(5, true),
                input.layout(6, true),
                output.layout(7, false),
            ],
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/subgroup/matmul_vec_nf4.wgsl"),
            &[
                matrix.meta_layout(0),
                input.meta_layout(1),
                output.meta_layout(2),
                quant.layout(3),
                matrix.layout(4, true),
                absmax.layout(5, true),
                input.layout(6, true),
                output.layout(7, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, quant.resource_key())
            .touch(4, matrix.resource_key())
            .touch(5, absmax.resource_key())
            .touch(6, input.resource_key())
            .touch(7, output.resource_key())
            .bind(0, matrix.meta_binding())
            .bind(1, input.meta_binding())
            .bind(2, output.meta_binding())
            .bind(3, quant.binding())
            .bind(4, matrix.binding())
            .bind(5, absmax.binding())
            .bind(6, input.binding())
            .bind(7, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [matrix.shape[1] as u32 / 4, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Fp16 matrix-matrix multiplication.
    /// - `matrix` shape: `[K, M, B]`.
    /// - `input` shape: `[K, N, B]`.
    /// - `output` shape: `[M, N, B]`.
    ///
    /// Note: `K` must be multiples of 4; `M` and `N` must be multiples of 4.
    pub fn matmul_mat_fp16<'a, 'b, 'c, F0: Float, F1: Float>(
        matrix: impl Into<TensorGpuView<'c, f16>>,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let matrix: TensorGpuView<_> = matrix.into();
        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = output.context();
        let shape = {
            let [m, n, b, _] = output.shape().into();
            let [k, _, _, _] = input.shape().into();
            matrix.check_shape([k, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        let key = PipelineKey::new(
            "matmul_mat_fp16",
            "matmul",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT", act),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/matmul_mat_fp16.wgsl"),
            &[
                matrix.meta_layout(0),
                input.meta_layout(1),
                output.meta_layout(2),
                matrix.layout(3, true),
                input.layout(4, true),
                output.layout(5, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, matrix.resource_key())
            .touch(4, input.resource_key())
            .touch(5, output.resource_key())
            .bind(0, matrix.meta_binding())
            .bind(1, input.meta_binding())
            .bind(2, output.meta_binding())
            .bind(3, matrix.binding())
            .bind(4, input.binding())
            .bind(5, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(u32::div_ceil(shape[0] as u32, 4), BLOCK_SIZE),
                u32::div_ceil(u32::div_ceil(shape[1] as u32, 4), BLOCK_SIZE),
                shape[2] as u32,
            ],
        })
    }

    /// Int8 matrix-matrix multiplication.
    /// - `matrix` shape: `[K, M, B]`.
    /// - `input` shape: `[K, N, B]`.
    /// - `output` shape: `[M, N, B]`.
    ///
    /// Notes:
    /// 1. `K` must be multiples of 4; `M` and `N` must be multiples of 4.
    /// 2. The total size of `matrix` must be multiples of 128.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mat_int8<'a, 'b, 'c, F0: Float, F1: Float>(
        matrix: impl Into<TensorGpuView<'c, u8>>,
        minmax: &TensorGpu<f16, ReadWrite>,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let matrix: TensorGpuView<_> = matrix.into();
        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = output.context();
        let shape = {
            let [m, n, b, _] = output.shape().into();
            let [k, _, _, _] = input.shape().into();
            let len = matrix.shape().len();
            minmax.check_shape([(len << 1).div_ceil(Self::INT8_BLOCK_SIZE as usize), 1, 1, 1])?;
            matrix.check_shape([k, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        let key = PipelineKey::new(
            "matmul_mat_int8",
            "matmul",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .int8(Self::INT8_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT", act),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/matmul_mat_int8.wgsl"),
            &[
                matrix.meta_layout(0),
                input.meta_layout(1),
                output.meta_layout(2),
                minmax.layout(3, true),
                matrix.layout(4, true),
                input.layout(5, true),
                output.layout(6, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, matrix.resource_key())
            .touch(4, minmax.resource_key())
            .touch(5, input.resource_key())
            .touch(6, output.resource_key())
            .bind(0, matrix.meta_binding())
            .bind(1, input.meta_binding())
            .bind(2, output.meta_binding())
            .bind(3, minmax.binding())
            .bind(4, matrix.binding())
            .bind(5, input.binding())
            .bind(6, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(u32::div_ceil(shape[0] as u32, 4), BLOCK_SIZE),
                u32::div_ceil(u32::div_ceil(shape[1] as u32, 4), BLOCK_SIZE),
                shape[2] as u32,
            ],
        })
    }

    /// NFloat4 matrix-matrix multiplication.
    /// - `matrix` shape: `[K, M, B]`.
    /// - `input` shape: `[K, N, B]`.
    /// - `output` shape: `[M, N, B]`.
    ///
    /// Notes:
    /// 1. `K` must be multiples of 8; `M` and `N` must be multiples of 8.
    /// 2. The total size of `matrix` must be multiples of 256.
    pub fn matmul_mat_nf4<'a, 'b, 'c, F0: Float, F1: Float>(
        matrix: impl Into<TensorGpuView<'c, u8>>,
        quant: &TensorGpu<f32, Uniform>,
        absmax: &TensorGpu<f16, ReadWrite>,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let matrix: TensorGpuView<_> = matrix.into();
        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = output.context();
        let shape = {
            let [m, n, b, _] = output.shape().into();
            let [k, _, _, _] = input.shape().into();
            let len = matrix.shape().len() << 1;
            absmax.check_shape([len.div_ceil(Self::NF4_BLOCK_SIZE as usize), 1, 1, 1])?;
            matrix.check_shape([k >> 1, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        let key = PipelineKey::new(
            "matmul_mat_nf4",
            "matmul",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT", act),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/matmul_mat_nf4.wgsl"),
            &[
                matrix.meta_layout(0),
                input.meta_layout(1),
                output.meta_layout(2),
                quant.layout(3),
                absmax.layout(4, true),
                matrix.layout(5, true),
                input.layout(6, true),
                output.layout(7, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, quant.resource_key())
            .touch(4, absmax.resource_key())
            .touch(5, matrix.resource_key())
            .touch(6, input.resource_key())
            .touch(7, output.resource_key())
            .bind(0, matrix.meta_binding())
            .bind(1, input.meta_binding())
            .bind(2, output.meta_binding())
            .bind(3, quant.binding())
            .bind(4, absmax.binding())
            .bind(5, matrix.binding())
            .bind(6, input.binding())
            .bind(7, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(u32::div_ceil(shape[0] as u32, 4), BLOCK_SIZE),
                u32::div_ceil(u32::div_ceil(shape[1] as u32, 4), BLOCK_SIZE),
                shape[2] as u32,
            ],
        })
    }

    /// Add `input` to `output`.
    /// - `input` shape: `[C, 1, B]` or `[C, T, B]`.
    /// - `output` shape: `[C, T, B]`.
    /// - Activations may be applied to `input`, `output` and the final result.
    pub fn add_activate<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act_x: Activation,
        act_y: Activation,
        act_out: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = output.context();
        let shape = {
            let [index, token, batch, _] = output.shape().into();
            input.check_shape_any(&[
                [index, token, batch, 1],
                [index, token, 1, batch],
                [index, 1, batch, 1],
                [index, 1, 1, 1],
            ])?;
            output.check_shape([index, token, batch, 1])?;
            output.shape()
        };

        let key = PipelineKey::new(
            "add",
            "add",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT_X", act_x)
                .activate("ACT_Y", act_y)
                .activate("ACT_OUT", act_out),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/binary.wgsl"),
            &[
                input.meta_layout(0),
                output.meta_layout(1),
                input.layout(2, true),
                output.layout(3, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, input.resource_key())
            .touch(3, output.resource_key())
            .bind(0, input.meta_binding())
            .bind(1, output.meta_binding())
            .bind(2, input.binding())
            .bind(3, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    /// Add `input` to `output`.
    /// - `input` shape: `[C, 1, B]` or `[C, T, B]`.
    /// - `output` shape: `[C, T, B]`.
    pub fn add<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
    ) -> Result<Self, TensorError> {
        Self::add_activate(
            input,
            output,
            Activation::None,
            Activation::None,
            Activation::None,
        )
    }

    /// Multiply `input` to `output`.
    /// - `input` shape: `[C, 1, B]` or `[C, T, B]`.
    /// - `output` shape: `[C, T, B]`.
    /// - Activations may be applied to `input`, `output` and the final result.
    pub fn mul_activate<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act_x: Activation,
        act_y: Activation,
        act_out: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = output.context();
        let shape = {
            let [index, token, batch, _] = output.shape().into();
            input.check_shape_any(&[
                [index, token, batch, 1],
                [index, token, 1, batch],
                [index, 1, batch, 1],
                [index, 1, 1, 1],
            ])?;
            output.check_shape([index, token, batch, 1])?;
            output.shape()
        };

        let key = PipelineKey::new(
            "mul",
            "mul",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .activate("ACT_X", act_x)
                .activate("ACT_Y", act_y)
                .activate("ACT_OUT", act_out),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/binary.wgsl"),
            &[
                input.meta_layout(0),
                output.meta_layout(1),
                input.layout(2, true),
                output.layout(3, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, input.resource_key())
            .touch(3, output.resource_key())
            .bind(0, input.meta_binding())
            .bind(1, output.meta_binding())
            .bind(2, input.binding())
            .bind(3, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    /// Multiply `input` to `output`.
    /// - `input` shape: `[C, 1, B]` or `[C, T, B]`.
    /// - `output` shape: `[C, T, B]`.
    pub fn mul<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
    ) -> Result<Self, TensorError> {
        Self::mul_activate(
            input,
            output,
            Activation::None,
            Activation::None,
            Activation::None,
        )
    }

    pub fn token_shift<'a, 'b, F: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        time_mix: impl Into<TensorGpuView<'a, F>>,
        state: impl Into<TensorGpuView<'b, f32>>,
        input: &TensorGpu<impl Float, ReadWrite>,
        output: &TensorGpu<impl Float, ReadWrite>,
        reversed: bool,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let time_mix: TensorGpuView<_> = time_mix.into();
        let state: TensorGpuView<_> = state.into();

        let context = output.context();
        let shape = {
            let [index, token, count, _] = output.shape().into();
            let [_, head, batch, _] = state.shape().into();
            input.check_shape_any(&[[index, token, count, 1], [index, token, 1, 1]])?;
            time_mix.check_shape_any(&[[index, token, count, 1], [index, 1, 1, 1]])?;
            state.check_shape([index, head, batch, 1])?;
            output.shape()
        };

        let key = PipelineKey::new(
            "token_shift",
            "token_shift",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&time_mix, Some("TIME_MIX"))
                .tensor(input, Some("IN"))
                .tensor(output, Some("OUT"))
                .bool("REVERSED", reversed),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/token_shift.wgsl"),
            &[
                output.meta_layout(0),
                time_mix.meta_layout(1),
                state.meta_layout(2),
                cursors.layout(3, true),
                time_mix.layout(4, true),
                state.layout(5, true),
                input.layout(6, true),
                output.layout(7, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, cursors.resource_key())
            .touch(4, time_mix.resource_key())
            .touch(5, state.resource_key())
            .touch(6, input.resource_key())
            .touch(7, output.resource_key())
            .bind(0, output.meta_binding())
            .bind(1, time_mix.meta_binding())
            .bind(2, state.meta_binding())
            .bind(3, cursors.binding())
            .bind(4, time_mix.binding())
            .bind(5, state.binding())
            .bind(6, input.binding())
            .bind(7, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn time_mix_v4<'a, T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        time_decay: &TensorGpu<f32, ReadWrite>,
        time_first: &TensorGpu<f32, ReadWrite>,
        state: impl Into<TensorGpuView<'a, f32>>,
        k: &TensorGpu<T, ReadWrite>,
        v: &TensorGpu<T, ReadWrite>,
        r: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let state: TensorGpuView<_> = state.into();

        let context = x.context();
        let shape = x.shape();
        k.check_shape(shape)?;
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        time_decay.check_shape([shape[0], 1, 1, 1])?;
        time_first.check_shape([shape[0], 1, 1, 1])?;
        state.check_shape([shape[0], 4, state.shape()[2], 1])?;

        let key = PipelineKey::new(
            "time_mix_v4",
            "time_mix",
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/time_mix_v4.wgsl"),
            &[
                x.meta_layout(0),
                state.meta_layout(1),
                cursors.layout(2, true),
                time_decay.layout(3, true),
                time_first.layout(4, true),
                state.layout(5, false),
                k.layout(6, true),
                v.layout(7, true),
                r.layout(8, true),
                x.layout(9, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, cursors.resource_key())
            .touch(3, time_decay.resource_key())
            .touch(4, time_first.resource_key())
            .touch(5, state.resource_key())
            .touch(6, k.resource_key())
            .touch(7, v.resource_key())
            .touch(8, r.resource_key())
            .touch(9, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, state.meta_binding())
            .bind(2, cursors.binding())
            .bind(3, time_decay.binding())
            .bind(4, time_first.binding())
            .bind(5, state.binding())
            .bind(6, k.binding())
            .bind(7, v.binding())
            .bind(8, r.binding())
            .bind(9, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE), 1, 1],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn time_mix_v5<'a, T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        time_decay: &TensorGpu<f32, ReadWrite>,
        time_first: &TensorGpu<f32, ReadWrite>,
        state: impl Into<TensorGpuView<'a, f32>>,
        k: &TensorGpu<T, ReadWrite>,
        v: &TensorGpu<T, ReadWrite>,
        r: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 32;

        let state: TensorGpuView<_> = state.into();

        let context = x.context();
        let shape = x.shape();
        let stride = shape[0] * shape[1];

        k.check_shape(shape)?;
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        time_decay.check_shape([shape[0], shape[1], 1, 1])?;
        time_first.check_shape([shape[0], shape[1], 1, 1])?;
        state.check_shape([stride, shape[0] + 1, state.shape()[2], 1])?;

        let key = PipelineKey::new(
            "time_mix_v5",
            "time_mix",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .u32("HEAD_SIZE", shape[0] as u32 / 4)
                .tensor(x, None),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/time_mix_v5.wgsl"),
            &[
                x.meta_layout(0),
                state.meta_layout(1),
                cursors.layout(2, true),
                time_decay.layout(3, true),
                time_first.layout(4, true),
                state.layout(5, false),
                k.layout(6, true),
                v.layout(7, true),
                r.layout(8, true),
                x.layout(9, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, cursors.resource_key())
            .touch(3, time_decay.resource_key())
            .touch(4, time_first.resource_key())
            .touch(5, state.resource_key())
            .touch(6, k.resource_key())
            .touch(7, v.resource_key())
            .touch(8, r.resource_key())
            .touch(9, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, state.meta_binding())
            .bind(2, cursors.binding())
            .bind(3, time_decay.binding())
            .bind(4, time_first.binding())
            .bind(5, state.binding())
            .bind(6, k.binding())
            .bind(7, v.binding())
            .bind(8, r.binding())
            .bind(9, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [u32::div_ceil(stride as u32 / 4, BLOCK_SIZE), 1, 1],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn time_mix_v6<'a, T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        time_decay: &TensorGpu<f32, ReadWrite>,
        time_first: &TensorGpu<f32, ReadWrite>,
        state: impl Into<TensorGpuView<'a, f32>>,
        k: &TensorGpu<T, ReadWrite>,
        v: &TensorGpu<T, ReadWrite>,
        r: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 32;

        let state: TensorGpuView<_> = state.into();

        let context = x.context();
        let shape = x.shape();
        let stride = shape[0] * shape[1];

        k.check_shape(shape)?;
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        time_decay.check_shape(shape)?;
        time_first.check_shape([shape[0], shape[1], 1, 1])?;
        state.check_shape([stride, shape[0] + 1, state.shape()[2], 1])?;

        let key = PipelineKey::new(
            "time_mix_v6",
            "time_mix",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .u32("HEAD_SIZE", shape[0] as u32 / 4)
                .tensor(x, None),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/time_mix_v6.wgsl"),
            &[
                x.meta_layout(0),
                state.meta_layout(1),
                cursors.layout(2, true),
                time_decay.layout(3, true),
                time_first.layout(4, true),
                state.layout(5, false),
                k.layout(6, true),
                v.layout(7, true),
                r.layout(8, true),
                x.layout(9, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, cursors.resource_key())
            .touch(3, time_decay.resource_key())
            .touch(4, time_first.resource_key())
            .touch(5, state.resource_key())
            .touch(6, k.resource_key())
            .touch(7, v.resource_key())
            .touch(8, r.resource_key())
            .touch(9, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, state.meta_binding())
            .bind(2, cursors.binding())
            .bind(3, time_decay.binding())
            .bind(4, time_first.binding())
            .bind(5, state.binding())
            .bind(6, k.binding())
            .bind(7, v.binding())
            .bind(8, r.binding())
            .bind(9, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [u32::div_ceil(stride as u32 / 4, BLOCK_SIZE), 1, 1],
        })
    }

    /// The V7 WKV kernel.
    /// - `n`: Stack of `k`, `v`, `a`, `kk`.
    ///
    /// Note that the state layout is different from the official implementation.
    /// Here is an illustration of each head's layout:
    ///
    /// ![time-mix-v7][time-mix-v7]
    #[embed_doc_image("time-mix-v7", "src/tensor/time-mix-v7.png")]
    pub fn time_mix_v7<'a, T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        state: impl Into<TensorGpuView<'a, f32>>,
        r: &TensorGpu<T, ReadWrite>,
        w: &TensorGpu<T, ReadWrite>,
        n: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 32;

        let state: TensorGpuView<_> = state.into();

        let context = x.context();
        let shape = x.shape();
        let stride = shape[0] * shape[1];

        r.check_shape(shape)?;
        w.check_shape(shape)?;
        n.check_shape([shape[0], shape[1], shape[2], 4])?;
        state.check_shape([stride, shape[0] + 1, state.shape()[2], 1])?;

        let key = PipelineKey::new(
            "time_mix_v7",
            "time_mix",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .u32("HEAD_SIZE", shape[0] as u32 / 4)
                .bool("TIME_MIX", true)
                .tensor(x, None)
                .activate("ACT", Activation::None),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/time_mix_v7.wgsl"),
            &[
                x.meta_layout(0),
                state.meta_layout(1),
                cursors.layout(2, true),
                state.layout(3, false),
                r.layout(5, true),
                w.layout(6, true),
                n.layout(7, true),
                x.layout(9, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, cursors.resource_key())
            .touch(3, state.resource_key())
            .touch(5, r.resource_key())
            .touch(6, w.resource_key())
            .touch(7, n.resource_key())
            .touch(9, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, state.meta_binding())
            .bind(2, cursors.binding())
            .bind(3, state.binding())
            .bind(5, r.binding())
            .bind(6, w.binding())
            .bind(7, n.binding())
            .bind(9, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [u32::div_ceil(stride as u32 / 4, BLOCK_SIZE), 1, 1],
        })
    }

    pub fn time_first_v7<T: Float>(
        u: &TensorGpu<f16, ReadWrite>,
        r: &TensorGpu<T, ReadWrite>,
        n: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 32;

        let context = x.context();
        let shape = x.shape();
        let stride = shape[0] * shape[1];

        r.check_shape(shape)?;
        u.check_shape([shape[0], shape[1], 1, 1])?;
        n.check_shape([shape[0], shape[1], shape[2], 4])?;

        let key = PipelineKey::new(
            "time_first_v7",
            "time_first",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .u32("HEAD_SIZE", shape[0] as u32 / 4)
                .bool("TIME_FIRST", true)
                .tensor(x, None)
                .activate("ACT", Activation::None),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/time_mix_v7.wgsl"),
            &[
                x.meta_layout(0),
                u.layout(4, true),
                r.layout(5, true),
                n.layout(7, true),
                x.layout(9, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(4, u.resource_key())
            .touch(5, r.resource_key())
            .touch(7, n.resource_key())
            .touch(9, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(4, u.binding())
            .bind(5, r.binding())
            .bind(7, n.binding())
            .bind(9, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(stride as u32 / 4, BLOCK_SIZE),
                shape[2] as u32,
                1,
            ],
        })
    }

    pub fn control_k_v7<'a, 'b, F0: Float, F1: Float>(
        p: &TensorGpu<f16, ReadWrite>,
        a: impl Into<TensorGpuView<'a, F0>>,
        k: impl Into<TensorGpuView<'b, F1>>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let p: TensorGpuView<_> = p.into();
        let a: TensorGpuView<_> = a.into();
        let k: TensorGpuView<_> = k.into();

        let context = k.context();
        let shape = {
            let [index, token, batch, _] = k.shape().into();
            a.check_shape([index, token, batch, 1])?;
            p.check_shape([index, 1, 1, 1])?;
            k.shape()
        };

        let key = PipelineKey::new(
            "control_k_v7",
            "main",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&a, Some("A"))
                .tensor(&k, Some("K")),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/control_k_v7.wgsl"),
            &[
                p.meta_layout(0),
                a.meta_layout(1),
                k.meta_layout(2),
                p.layout(3, true),
                a.layout(4, true),
                k.layout(5, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, p.resource_key())
            .touch(4, a.resource_key())
            .touch(5, k.resource_key())
            .bind(0, p.meta_binding())
            .bind(1, a.meta_binding())
            .bind(2, k.meta_binding())
            .bind(3, p.binding())
            .bind(4, a.binding())
            .bind(5, k.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn channel_mix<'a, T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        state: impl Into<TensorGpuView<'a, f32>>,
        r: &TensorGpu<T, ReadWrite>,
        v: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let state: TensorGpuView<_> = state.into();

        let context = x.context();
        let shape = x.shape();
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        state.check_shape([shape[0], 1, state.shape()[2], 1])?;

        let key = PipelineKey::new(
            "channel_mix",
            "channel_mix",
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/channel_mix.wgsl"),
            &[
                x.meta_layout(0),
                state.meta_layout(1),
                cursors.layout(2, true),
                state.layout(3, false),
                r.layout(4, true),
                v.layout(5, true),
                x.layout(6, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, cursors.resource_key())
            .touch(3, state.resource_key())
            .touch(4, r.resource_key())
            .touch(5, v.resource_key())
            .touch(6, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, state.meta_binding())
            .bind(2, cursors.binding())
            .bind(3, state.binding())
            .bind(4, r.binding())
            .bind(5, v.binding())
            .bind(6, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                1,
            ],
        })
    }

    pub fn channel_mix_v7<'a, T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        state: impl Into<TensorGpuView<'a, f32>>,
        v: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let state: TensorGpuView<_> = state.into();

        let context = x.context();
        let shape = x.shape();
        v.check_shape(shape)?;
        state.check_shape([shape[0], 1, state.shape()[2], 1])?;

        let key = PipelineKey::new(
            "channel_mix",
            "channel_mix",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .bool("V7", true),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/channel_mix.wgsl"),
            &[
                x.meta_layout(0),
                state.meta_layout(1),
                cursors.layout(2, true),
                state.layout(3, false),
                v.layout(5, true),
                x.layout(6, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, cursors.resource_key())
            .touch(3, state.resource_key())
            .touch(5, v.resource_key())
            .touch(6, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, state.meta_binding())
            .bind(2, cursors.binding())
            .bind(3, state.binding())
            .bind(5, v.binding())
            .bind(6, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                1,
            ],
        })
    }

    pub fn activate<'a, F: Float>(
        x: impl Into<TensorGpuView<'a, F>>,
        act: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let x: TensorGpuView<_> = x.into();

        let context = x.context();
        let shape = x.shape();

        let key = PipelineKey::new(
            "activate",
            "act",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&x, None)
                .activate("ACT", act),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/activation.wgsl"),
            &[x.meta_layout(0), x.layout(1, false)],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(1, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    /// Copy the content of `input` into `output` of the same shape.
    pub fn blit<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
    ) -> Result<Self, TensorError> {
        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = input.context();
        let shape = output.shape();
        input.check_shape(shape)?;

        let block_size = match shape[1] {
            x if x < 8 => [128, 1],
            _ => [16, 16],
        };

        let key = PipelineKey::new(
            "blit",
            "blit",
            Macros::new()
                .u32("BLOCK_SIZE_X", block_size[0])
                .u32("BLOCK_SIZE_Y", block_size[1])
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/blit.wgsl"),
            &[
                input.meta_layout(0),
                output.meta_layout(1),
                input.layout(2, true),
                output.layout(3, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, input.resource_key())
            .touch(3, output.resource_key())
            .bind(0, input.meta_binding())
            .bind(1, output.meta_binding())
            .bind(2, input.binding())
            .bind(3, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, block_size[0]),
                u32::div_ceil(shape[1] as u32, block_size[1]),
                shape[2] as u32,
            ],
        })
    }

    /// Repeat the content of `input` into `output` along the token and batch axes.
    pub fn broadcast<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = input.context();
        let shape = output.shape();
        input.check_shape([shape[0], input.shape()[1], input.shape()[2], 1])?;

        let key = PipelineKey::new(
            "broadcast",
            "broadcast",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/reshape.wgsl"),
            &[
                input.meta_layout(0),
                output.meta_layout(1),
                input.layout(2, true),
                output.layout(3, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, input.resource_key())
            .touch(3, output.resource_key())
            .bind(0, input.meta_binding())
            .bind(1, output.meta_binding())
            .bind(2, input.binding())
            .bind(3, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    /// Swap the `token` and `batch` axes.
    pub fn transpose<'a, 'b, F0: Float, F1: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = input.context();
        let shape = input.shape();
        output.check_shape([shape[0], shape[2], shape[1], 1])?;

        let key = PipelineKey::new(
            "transpose",
            "transpose",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/reshape.wgsl"),
            &[
                input.meta_layout(0),
                output.meta_layout(1),
                input.layout(2, true),
                output.layout(3, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, input.resource_key())
            .touch(3, output.resource_key())
            .bind(0, input.meta_binding())
            .bind(1, output.meta_binding())
            .bind(2, input.binding())
            .bind(3, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn blend(
        factor: &TensorGpu<f32, Uniform>,
        input: &TensorGpu<impl Float, ReadWrite>,
        output: &TensorGpu<impl Float, ReadWrite>,
    ) -> Result<Self, TensorError> {
        let context = output.context();
        let shape = output.shape();
        input.check_shape(shape)?;
        factor.check_shape([4, 1, 1, 1])?;

        let block_size = match shape[1] {
            x if x < 8 => [128, 1],
            _ => [16, 16],
        };

        let key = PipelineKey::new(
            "blend",
            "blend",
            Macros::new()
                .u32("BLOCK_SIZE_X", block_size[0])
                .u32("BLOCK_SIZE_Y", block_size[1])
                .tensor(input, Some("IN"))
                .tensor(output, Some("OUT")),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/blend.wgsl"),
            &[
                input.meta_layout(0),
                output.meta_layout(1),
                factor.layout(2),
                input.layout(3, true),
                output.layout(4, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, factor.resource_key())
            .touch(3, input.resource_key())
            .touch(4, output.resource_key())
            .bind(0, input.meta_binding())
            .bind(1, output.meta_binding())
            .bind(2, factor.binding())
            .bind(3, input.binding())
            .bind(4, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, block_size[0]),
                u32::div_ceil(shape[1] as u32, block_size[1]),
                shape[2] as u32,
            ],
        })
    }

    pub fn blend_lora<'a, 'b, 'c>(
        factor: &TensorGpu<f32, Uniform>,
        xa: impl Into<TensorGpuView<'a, f16>>,
        xb: impl Into<TensorGpuView<'b, f16>>,
        output: impl Into<TensorGpuView<'c, f16>>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let xa: TensorGpuView<_> = xa.into();
        let xb: TensorGpuView<_> = xb.into();
        let output: TensorGpuView<_> = output.into();

        let context = output.context();
        let shape = output.shape();
        factor.check_shape([4, 1, 1, 1])?;
        xa.check_shape([xa.shape()[0], shape[0], shape[2], 1])?;
        xb.check_shape([xb.shape()[0], shape[1], shape[2], 1])?;

        let key = PipelineKey::new(
            "blend_lora",
            "blend_lora",
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/blend_lora.wgsl"),
            &[
                xa.meta_layout(0),
                xb.meta_layout(1),
                output.meta_layout(2),
                factor.layout(3),
                xa.layout(4, true),
                xb.layout(5, true),
                output.layout(6, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, factor.resource_key())
            .touch(4, xa.resource_key())
            .touch(5, xb.resource_key())
            .touch(6, output.resource_key())
            .bind(0, xa.meta_binding())
            .bind(1, xb.meta_binding())
            .bind(2, output.meta_binding())
            .bind(3, factor.binding())
            .bind(4, xa.binding())
            .bind(5, xb.binding())
            .bind(6, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(u32::div_ceil(shape[0] as u32, 4), BLOCK_SIZE),
                u32::div_ceil(u32::div_ceil(shape[1] as u32, 4), BLOCK_SIZE),
                shape[2] as u32,
            ],
        })
    }

    pub fn lerp<'a, 'b, 'c, F0: Float, F1: Float, F2: Float>(
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        factor: impl Into<TensorGpuView<'c, F2>>,
        reversed: bool,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let factor: TensorGpuView<_> = factor.into();
        let input: TensorGpuView<_> = input.into();
        let output: TensorGpuView<_> = output.into();

        let context = output.context();
        let shape = {
            let [index, token, batch, _] = output.shape().into();
            factor.check_shape_any(&[
                [index, token, batch, 1],
                [index, token, 1, 1],
                [index, 1, batch, 1],
                [index, 1, 1, 1],
            ])?;
            input.check_shape([index, token, batch, 1])?;
            output.shape()
        };

        let key = PipelineKey::new(
            "lerp",
            "lerp",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&factor, Some("FACTOR"))
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .bool("REVERSED", reversed),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/lerp.wgsl"),
            &[
                factor.meta_layout(0),
                input.meta_layout(1),
                output.meta_layout(2),
                factor.layout(3, true),
                input.layout(4, true),
                output.layout(5, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(3, factor.resource_key())
            .touch(4, input.resource_key())
            .touch(5, output.resource_key())
            .bind(0, factor.meta_binding())
            .bind(1, input.meta_binding())
            .bind(2, output.meta_binding())
            .bind(3, factor.binding())
            .bind(4, input.binding())
            .bind(5, output.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn discount(
        x: &TensorGpu<impl Float, ReadWrite>,
        factor: f32,
        bias: f32,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let context = x.context();
        let shape = x.shape();

        let key = PipelineKey::new(
            "discount",
            "discount",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("FACTOR", factor)
                .f32("BIAS", bias),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/discount.wgsl"),
            &[x.meta_layout(0), x.layout(1, false)],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(1, x.resource_key())
            .bind(0, x.meta_binding())
            .bind(1, x.binding())
            .build()];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn quantize_mat_int8(
        input: &TensorGpu<f16, ReadWrite>,
        minmax: &TensorGpu<f16, ReadWrite>,
        output: &TensorGpu<u8, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let context = output.context();
        let shape = output.shape();
        let minmax_len = shape.len().div_ceil(Self::INT8_BLOCK_SIZE as usize);
        let minmax_shape = Shape::new(minmax_len << 1, 1, 1, 1);

        input.check_shape(shape)?;
        minmax.check_shape(minmax_shape)?;

        let key = PipelineKey::new(
            "quant_mat_int8_minmax",
            "compute_minmax",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .int8(Self::INT8_BLOCK_SIZE),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/quant_mat_int8.wgsl"),
            &[
                minmax.meta_layout(0),
                input.meta_layout(1),
                input.layout(2, true),
                minmax.layout(3, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, input.resource_key())
            .touch(3, minmax.resource_key())
            .bind(0, minmax.meta_binding())
            .bind(1, input.meta_binding())
            .bind(2, input.binding())
            .bind(3, minmax.binding())
            .build()];

        let compute_minmax = Self::Atom {
            pipeline,
            bindings,
            dispatch: [u32::div_ceil(minmax_len as u32, BLOCK_SIZE), 1, 1],
        };

        let output = output.reshape(
            TensorDimension::Auto,
            TensorDimension::Size(1),
            TensorDimension::Size(1),
            TensorDimension::Size(1),
        )?;

        let key = PipelineKey::new(
            "quant_mat_int8",
            "quantize",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .int8(Self::INT8_BLOCK_SIZE),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/quant_mat_int8.wgsl"),
            &[
                output.meta_layout(0),
                input.meta_layout(1),
                input.layout(2, true),
                minmax.layout(3, false),
                output.layout(4, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, input.resource_key())
            .touch(3, minmax.resource_key())
            .touch(4, output.resource_key())
            .bind(0, output.meta_binding())
            .bind(1, input.meta_binding())
            .bind(2, input.binding())
            .bind(3, minmax.binding())
            .bind(4, output.binding())
            .build()];

        let quantize = Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil(shape[0] as u32, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        };

        Ok(Self::List(vec![compute_minmax, quantize]))
    }

    pub fn quantize_mat_nf4(
        input: &TensorGpu<f16, ReadWrite>,
        quant: &TensorGpu<f32, Uniform>,
        absmax: &TensorGpu<f16, ReadWrite>,
        output: &TensorGpu<u8, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let context = output.context();
        let shape = output.shape();
        let input_shape = Shape::new(shape[0] << 1, shape[1], shape[2], shape[3]);
        let absmax_len = input_shape.len().div_ceil(Self::NF4_BLOCK_SIZE as usize);
        let absmax_shape = Shape::new(absmax_len, 1, 1, 1);

        input.check_shape(input_shape)?;
        absmax.check_shape(absmax_shape)?;

        let absmax_f32: TensorGpu<f32, ReadWrite> = context.tensor_init(absmax_shape);

        let key = PipelineKey::new(
            "quant_mat_nf4_absmax",
            "compute_absmax",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/quant_mat_nf4.wgsl"),
            &[
                absmax_f32.meta_layout(0),
                input.layout(2, true),
                absmax_f32.layout(3, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(2, input.resource_key())
            .touch(3, absmax_f32.resource_key())
            .bind(0, absmax_f32.meta_binding())
            .bind(2, input.binding())
            .bind(3, absmax_f32.binding())
            .build()];

        let compute_absmax = Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil((shape[0] << 1) as u32, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        };

        let output = output.reshape(
            TensorDimension::Auto,
            TensorDimension::Size(1),
            TensorDimension::Size(1),
            TensorDimension::Size(1),
        )?;

        let key = PipelineKey::new(
            "quant_mat_nf4",
            "quantize",
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE),
        );
        let pipeline = context.checkout_pipeline(
            &key,
            include_str!("../shaders/quant_mat_nf4.wgsl"),
            &[
                output.meta_layout(0),
                quant.layout(1),
                input.layout(2, true),
                absmax_f32.layout(3, false),
                output.layout(4, false),
            ],
        );

        let bindings = vec![BindGroupBuilder::new(&key, context, &pipeline.layout)
            .touch(1, quant.resource_key())
            .touch(2, input.resource_key())
            .touch(3, absmax_f32.resource_key())
            .bind(0, output.meta_binding())
            .bind(1, quant.binding())
            .bind(2, input.binding())
            .bind(3, absmax_f32.binding())
            .bind(4, output.binding())
            .build()];

        let quantize = Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                u32::div_ceil((shape[0]) as u32, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        };

        let quantize_absmax = Self::blit(&absmax_f32, absmax)?;

        Ok(Self::List(vec![compute_absmax, quantize, quantize_absmax]))
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use anyhow::Result;
    use half::f16;
    use itertools::Itertools;
    use wgpu::{Instance, PowerPreference};
    // use wgpu_profiler::GpuProfiler;

    use super::TensorOp;
    use crate::{
        context::{Context, ContextBuilder, InstanceExt},
        tensor::{ops::Activation, Shape, TensorGpu},
    };

    fn is_approx(a: impl Into<f32>, b: impl Into<f32>) -> bool {
        let a: f32 = a.into();
        let b: f32 = b.into();
        (a - b).abs() <= f32::max(f32::EPSILON, f32::max(a.abs(), b.abs()) * f32::EPSILON)
    }

    fn is_approx_eps(a: impl Into<f32>, b: impl Into<f32>, eps: f32) -> bool {
        let a: f32 = a.into();
        let b: f32 = b.into();
        (a - b).abs() <= f32::max(eps, f32::max(a.abs(), b.abs()) * eps)
    }

    async fn create_context() -> Result<Context> {
        let instance = Instance::default();
        let adapter = instance.adapter(PowerPreference::HighPerformance).await?;
        let context = ContextBuilder::new(adapter)
            // .features(Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES)
            .build()
            .await?;
        Ok(context)
    }

    #[tokio::test]
    async fn test_softmax() -> Result<()> {
        let context = create_context().await?;
        fastrand::seed(42);

        const C: usize = 1000;
        const T: usize = 3;
        const B: usize = 2;

        let x = [(); C * T * B]
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .to_vec();
        let shape = Shape::new(C, T, B, 1);

        let x_dev: TensorGpu<_, _> = context.tensor_from_data(shape, x.clone())?;
        let softmax = TensorOp::softmax(&x_dev)?;

        context.queue.submit(context.encode(&softmax));
        let x_host = x_dev.back().await.to_vec();

        let mut ans = vec![];
        for x in &x.into_iter().chunks(C) {
            let x = x.collect_vec().into_iter();
            let max = x.clone().reduce(f32::max).unwrap_or_default();
            let x = x.map(|x| (x - max).exp());
            let sum: f32 = x.clone().sum();
            let x = x.map(|x| x / sum);
            ans.extend(x);
        }

        for (index, (a, b)) in itertools::zip_eq(x_host, ans).enumerate() {
            assert!(
                is_approx(a, b),
                "Failed at index {index}, computed: {a} vs. answer: {b}"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_layer_norm() -> Result<()> {
        let context = create_context().await?;
        fastrand::seed(42);

        const C: usize = 1000;
        const T: usize = 3;
        const B: usize = 2;
        const EPS: f32 = 1.0e-5;

        let x = [(); C * T * B]
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .to_vec();
        let w = [(); C]
            .map(|_| f16::from_f32(fastrand::f32() - 0.5))
            .repeat(T * B)
            .to_vec();
        let b = [(); C]
            .map(|_| f16::from_f32(fastrand::f32() - 0.5))
            .repeat(T * B)
            .to_vec();

        let shape = Shape::new(C, T, B, 1);
        let x_dev = context.tensor_from_data(shape, x.clone())?;

        let shape = Shape::new(C, 1, 1, 1);
        let w_dev = context.tensor_from_data(shape, &w[..1000])?;
        let b_dev = context.tensor_from_data(shape, &b[..1000])?;

        // let shape = Shape::new(4, T, B, 1);
        // let s_dev = context.tensor_init(shape);

        let layer_norm = TensorOp::layer_norm(&w_dev, &b_dev, &x_dev, EPS)?;
        context.queue.submit(context.encode(&layer_norm));

        let x_host = x_dev.back().await.to_vec();
        // let s_host = s_dev.back().await.to_vec();

        // test recenter and rms norm
        let shape = Shape::new(C, T, B, 1);
        let x_dev = context.tensor_from_data(shape, x.clone())?;
        let ops = TensorOp::List(vec![
            TensorOp::recenter(&x_dev)?,
            TensorOp::rms_norm(&w_dev, &b_dev, &x_dev, EPS)?,
        ]);
        context.queue.submit(context.encode(&ops));

        let x_rms_host = x_dev.back().await.to_vec();

        let mut ans = vec![];
        // let mut ans_stats = vec![];
        for chunk in &x
            .into_iter()
            .zip(w.into_iter())
            .zip(b.into_iter())
            .chunks(C)
        {
            let chunk = chunk.collect_vec();
            let x = chunk.iter().map(|((x, _), _)| x).copied();
            // let sum: f32 = x.clone().sum();
            // let squared_sum: f32 = x.clone().map(|x| x.powi(2)).sum();

            // let mean = sum / C as f32;
            // let deviation = ((squared_sum / C as f32) - mean.powi(2)).sqrt();
            let (mean, m2, count) = x.fold((0.0f32, 0.0f32, 0u32), |(mean, m2, count), x| {
                let count = count + 1;
                let delta = x - mean;
                let mean = mean + delta / count as f32;
                let m2 = m2 + delta * (x - mean);
                (mean, m2, count)
            });
            let variance = m2 / count as f32 + EPS;
            let deviation = 1.0 / variance.sqrt();
            // ans_stats.append(&mut vec![mean, deviation, variance, 0.0]);

            let x = chunk
                .into_iter()
                .map(|((x, w), b)| (x - mean) * deviation * w.to_f32() + b.to_f32());
            ans.extend(x);
        }

        for (index, (a, &b)) in itertools::zip_eq(x_host, ans.iter()).enumerate() {
            assert!(
                is_approx_eps(a, b, 1.0e-3),
                "Failed at index {index}, computed: {a} vs. answer: {b}"
            );
        }

        for (index, (a, &b)) in itertools::zip_eq(x_rms_host, ans.iter()).enumerate() {
            assert!(
                is_approx_eps(a, b, 1.0e-3),
                "Failed at index {index}, computed: {a} vs. answer: {b}"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_l2_norm() -> Result<()> {
        let context = create_context().await?;
        fastrand::seed(42);

        const C: usize = 1000;
        const T: usize = 3;
        const B: usize = 2;
        const EPS: f32 = 1.0e-12;

        let x = [(); C * T * B]
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .to_vec();

        let shape = Shape::new(C, T, B, 1);
        let x_dev = context.tensor_from_data(shape, x.clone())?;

        let l2_norm = TensorOp::l2_norm(&x_dev, EPS)?;
        context.queue.submit(context.encode(&l2_norm));

        let x_host = x_dev.back().await.to_vec();

        let mut ans = vec![];
        for x in &x.into_iter().chunks(C) {
            let x = x.collect_vec().into_iter();
            let norm = x.clone().map(|x| x * x).sum::<f32>().sqrt();
            let x = x.map(|x| x / (norm + EPS));
            ans.extend(x);
        }

        for (index, (a, b)) in itertools::zip_eq(x_host, ans).enumerate() {
            assert!(
                is_approx(a, b),
                "Failed at index {index}, computed: {a} vs. answer: {b}"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_matmul() -> Result<()> {
        let context = create_context().await?;
        fastrand::seed(42);

        async fn test_matmul_inner(
            context: &Context,
            c: usize,
            r: usize,
            t: usize,
            b: usize,
        ) -> Result<()> {
            // let mut profiler = GpuProfiler::new(&context.adapter, &context.device, &context.queue, 1);

            let matrix = vec![(); c * r * b]
                .into_iter()
                .map(|_| 10.0 * (fastrand::f32() - 0.5))
                .map(f16::from_f32)
                .collect_vec();
            let input_f32 = vec![(); c * t * b]
                .into_iter()
                .map(|_| 10.0 * (fastrand::f32() - 0.5))
                .collect_vec();
            let input_f16 = input_f32.iter().copied().map(f16::from_f32).collect_vec();

            let matrix_shape = Shape::new(c, r, b, 1);
            let input_shape = Shape::new(c, t, b, 1);
            let output_shape = Shape::new(r, t, 2 * b, 1);

            let matrix_dev = context.tensor_from_data(matrix_shape, matrix.clone())?;
            let input_f32_dev = context.tensor_from_data(input_shape, input_f32.clone())?;
            let input_f16_dev: TensorGpu<f16, _> = context.tensor_init(input_shape);
            let output_dev: TensorGpu<_, _> = context.tensor_init(output_shape);

            let ops = TensorOp::List(vec![
                TensorOp::blit(&input_f32_dev, &input_f16_dev)?,
                TensorOp::matmul_vec_fp16(
                    &matrix_dev,
                    &input_f32_dev,
                    output_dev.view(.., .., 0..b, ..)?,
                    Activation::None,
                )?,
                TensorOp::matmul_mat_fp16(
                    &matrix_dev,
                    &input_f16_dev,
                    output_dev.view(.., .., b.., ..)?,
                    Activation::None,
                )?,
            ]);

            // profiler.resolve_queries(&mut encoder);
            context.queue.submit(context.encode(&ops));

            let output_host = output_dev.back().await;
            let output_host: Vec<f32> = Vec::from(output_host);

            // profiler.end_frame().unwrap();
            // context.device.poll(wgpu::MaintainBase::Wait);

            // if let Some(results) = profiler.process_finished_frame() {
            //     wgpu_profiler::chrometrace::write_chrometrace(
            //         std::path::Path::new(&format!("./trace/matmul_{T}.json")),
            //         &results,
            //     )
            //     .expect("failed to write trace");
            // }

            let mut ans = vec![0.0; output_host.len()];
            for ((batch, token), line) in (0..b).cartesian_product(0..t).cartesian_product(0..r) {
                let matrix = &matrix[((batch * r + line) * c)..((batch * r + line) + 1) * c];
                let input = &input_f32[(batch * t + token) * c..((batch * t + token) + 1) * c];
                let product = matrix
                    .iter()
                    .zip(input.iter())
                    .fold(0.0f32, |acc, x| acc + x.0.to_f32() * *x.1);
                ans[(batch * t + token) * r + line] = product;

                let input = &input_f16[(batch * t + token) * c..((batch * t + token) + 1) * c];
                let product = matrix
                    .iter()
                    .zip(input.iter())
                    .fold(0.0f32, |acc, x| acc + x.0.to_f32() * x.1.to_f32());
                ans[((b + batch) * t + token) * r + line] = product;
            }

            for (index, (a, b)) in itertools::zip_eq(output_host, ans).enumerate() {
                assert!(
                    is_approx_eps(a, b, 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            }

            Ok(())
        }

        test_matmul_inner(&context, 2560, 2048, 32, 2).await?;
        test_matmul_inner(&context, 320, 64, 320, 2).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_matmul_int8() -> Result<()> {
        let context = create_context().await?;
        fastrand::seed(42);

        const INT8_BLOCK_SIZE: usize = TensorOp::INT8_BLOCK_SIZE as usize;

        async fn test_matmul_int8_inner(
            context: &Context,
            c: usize,
            r: usize,
            t: usize,
        ) -> Result<()> {
            let matrix = vec![(); c * r]
                .into_iter()
                .map(|_| 10.0 * (fastrand::f32() - 0.5))
                .map(f16::from_f32)
                .collect_vec();
            let input_f32 = vec![(); c * t]
                .into_iter()
                .map(|_| 10.0 * (fastrand::f32() - 0.5))
                .collect_vec();
            let input_f16 = input_f32.iter().copied().map(f16::from_f32).collect_vec();

            let (matrix_u8, min, max) = {
                let mut matrix_u8: Vec<u8> = vec![0; matrix.len()];
                let mut min = vec![f16::MAX; matrix.len().div_ceil(INT8_BLOCK_SIZE)];
                let mut max = vec![f16::MIN; matrix.len().div_ceil(INT8_BLOCK_SIZE)];

                for (i, (min, max)) in itertools::zip_eq(&mut min, &mut max).enumerate() {
                    let start = i * INT8_BLOCK_SIZE;
                    let end = start + INT8_BLOCK_SIZE;
                    let chunk = &matrix[start..end];
                    for value in chunk.iter() {
                        *min = min.min(*value);
                        *max = max.max(*value);
                    }
                    for (j, value) in chunk.iter().enumerate() {
                        let value = value.to_f32();
                        let min = min.to_f32();
                        let max = max.to_f32();
                        let value = (value - min) / (max - min);
                        matrix_u8[start + j] = f32::round(value * 255.0) as u8;
                    }
                }

                (matrix_u8, min, max)
            };
            let minmax = itertools::zip_eq(&min, &max)
                .map(|(&min, &max)| [min, max])
                .collect_vec()
                .concat();

            let minmax_shape = Shape::new((c * r).div_ceil(INT8_BLOCK_SIZE) * 2, 1, 1, 1);
            let matrix_shape = Shape::new(c, r, 1, 1);
            let input_shape = Shape::new(c, t, 1, 1);
            let output_shape = Shape::new(r, t, 1, 1);

            let minmax_dev = context.tensor_init(minmax_shape);
            let matrix_f16_dev = context.tensor_from_data(matrix_shape, matrix.clone())?;

            let matrix_u8_dev = context.tensor_init(matrix_shape);
            let input_dev = context.tensor_from_data(input_shape, input_f16.clone())?;
            let output_dev = context.tensor_init(output_shape);

            let ops = TensorOp::List(vec![TensorOp::quantize_mat_int8(
                &matrix_f16_dev,
                &minmax_dev,
                &matrix_u8_dev,
            )?]);
            context.queue.submit(context.encode(&ops));
            let minmax_host = minmax_dev.back().await.to_vec();
            let matrix_u8_host = matrix_u8_dev.back().await.to_vec();

            for (index, (&a, &b)) in itertools::zip_eq(&minmax_host, &minmax).enumerate() {
                assert!(
                    is_approx_eps(a, b, 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            }
            for (index, (&a, &b)) in itertools::zip_eq(&matrix_u8_host, &matrix_u8).enumerate() {
                assert!(
                    a.abs_diff(b) < 2,
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            }

            let mut ans = vec![0.0; t * r];
            for (token, line) in (0..t).cartesian_product(0..r) {
                let matrix = &matrix_u8_host[line * c..(line + 1) * c];
                let input = &input_f16[token * c..(token + 1) * c];
                let product =
                    matrix
                        .iter()
                        .zip_eq(input.iter())
                        .enumerate()
                        .fold(0.0f32, |acc, (i, x)| {
                            let min = min[(line * c + i) / INT8_BLOCK_SIZE].to_f32();
                            let max = max[(line * c + i) / INT8_BLOCK_SIZE].to_f32();
                            let value = (*x.0 as f32) / 255.0;
                            acc + (value * (max - min) + min) * x.1.to_f32()
                        });
                ans[token * r + line] = product;
            }

            let ops = TensorOp::List(vec![TensorOp::matmul_vec_int8(
                &matrix_u8_dev,
                &minmax_dev,
                &input_dev,
                &output_dev,
                Activation::None,
            )?]);
            context.queue.submit(context.encode(&ops));
            let output_host: Vec<f32> = output_dev.back().await.to_vec();

            for (index, (&a, &b)) in itertools::zip_eq(&output_host, &ans).enumerate() {
                assert!(
                    is_approx_eps(a, b, 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            }

            let ops = TensorOp::List(vec![TensorOp::matmul_mat_int8(
                &matrix_u8_dev,
                &minmax_dev,
                &input_dev,
                &output_dev,
                Activation::None,
            )?]);
            context.queue.submit(context.encode(&ops));
            let output_host = output_dev.back().await.to_vec();

            for (index, (&a, &b)) in itertools::zip_eq(&output_host, &ans).enumerate() {
                assert!(
                    is_approx_eps(a, b, 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            }

            Ok(())
        }

        test_matmul_int8_inner(&context, 2560, 2048, 64).await?;
        test_matmul_int8_inner(&context, 320, 64, 320).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_matmul_nf4() -> Result<()> {
        let context = create_context().await?;
        fastrand::seed(42);

        const NF4_BLOCK_SIZE: usize = TensorOp::NF4_BLOCK_SIZE as usize;

        fn normal() -> f32 {
            let u = fastrand::f32();
            let v = fastrand::f32();
            (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
        }

        async fn test_matmul_nf4_inner(
            context: &Context,
            c: usize,
            r: usize,
            t: usize,
        ) -> Result<()> {
            let matrix = vec![(); c * r]
                .into_iter()
                .map(|_| normal())
                .map(f16::from_f32)
                .collect_vec();
            let input_f32 = vec![(); c * t]
                .into_iter()
                .map(|_| 2.0 * fastrand::f32() - 1.0)
                .collect_vec();
            let input_f16 = input_f32.iter().copied().map(f16::from_f32).collect_vec();

            #[allow(clippy::excessive_precision)]
            let quant: [f32; 16] = [
                -1.0,
                -0.6961928009986877,
                -0.5250730514526367,
                -0.39491748809814453,
                -0.28444138169288635,
                -0.18477343022823334,
                -0.09105003625154495,
                0.0,
                0.07958029955625534,
                0.16093020141124725,
                0.24611230194568634,
                0.33791524171829224,
                0.44070982933044434,
                0.5626170039176941,
                0.7229568362236023,
                1.0,
            ];
            let (matrix_u8, matrix_u4, absmax) = {
                let mut matrix_u8: Vec<u8> = vec![0; matrix.len()];
                let mut matrix_u4: Vec<u8> = vec![0; matrix.len() / 2];
                let mut absmax = vec![f16::ZERO; matrix.len().div_ceil(NF4_BLOCK_SIZE)];

                for (i, absmax) in absmax.iter_mut().enumerate() {
                    let start = i * NF4_BLOCK_SIZE;
                    let end = start + NF4_BLOCK_SIZE;
                    let chunk = &matrix[start..end];
                    *absmax = chunk
                        .iter()
                        .map(|&x| if x >= f16::ZERO { x } else { -x })
                        .reduce(f16::max)
                        .unwrap();
                    for (j, value) in chunk.iter().enumerate() {
                        let value = value.to_f32() / absmax.to_f32();
                        matrix_u8[start + j] = quant
                            .iter()
                            .map(|quant| (value - quant).abs())
                            .enumerate()
                            .fold((0, f32::MAX), |acc, x| if x.1 < acc.1 { x } else { acc })
                            .0 as u8;
                    }
                }

                for (i, x) in matrix_u4.iter_mut().enumerate() {
                    *x = matrix_u8[2 * i] | matrix_u8[2 * i + 1] << 4;
                }

                (matrix_u8, matrix_u4, absmax)
            };

            let quant_shape = Shape::new(quant.len(), 1, 1, 1);
            let absmax_shape = Shape::new((c * r).div_ceil(NF4_BLOCK_SIZE), 1, 1, 1);
            let matrix_f16_shape = Shape::new(c, r, 1, 1);
            let matrix_u4_shape = Shape::new(c / 2, r, 1, 1);
            let input_shape = Shape::new(c, t, 1, 1);
            let output_shape = Shape::new(r, t, 1, 1);

            let quant_dev = context.tensor_from_data(quant_shape, quant.to_vec())?;
            let absmax_dev = context.tensor_init(absmax_shape);
            let matrix_f16_dev = context.tensor_from_data(matrix_f16_shape, matrix.clone())?;

            let matrix_u4_dev = context.tensor_init(matrix_u4_shape);
            let input_dev: TensorGpu<_, _> =
                context.tensor_from_data(input_shape, input_f16.clone())?;
            let output_dev: TensorGpu<_, _> = context.tensor_init(output_shape);

            let ops = TensorOp::List(vec![TensorOp::quantize_mat_nf4(
                &matrix_f16_dev,
                &quant_dev,
                &absmax_dev,
                &matrix_u4_dev,
            )?]);
            context.queue.submit(context.encode(&ops));
            let matrix_u4_host = matrix_u4_dev.back().await.to_vec();
            let absmax_host = absmax_dev.back().await.to_vec();

            for (index, (&a, &b)) in itertools::zip_eq(&absmax_host, &absmax).enumerate() {
                assert!(
                    is_approx_eps(a.to_f32(), b.to_f32(), 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            }

            for (index, (a, b)) in itertools::zip_eq(matrix_u4_host, matrix_u4).enumerate() {
                assert!(
                    a == b,
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            }

            let mut truth = vec![0.0; t * r];
            for (token, line) in (0..t).cartesian_product(0..r) {
                let matrix = &matrix[line * c..(line + 1) * c];
                let input = &input_f16[token * c..(token + 1) * c];
                let product = matrix
                    .iter()
                    .zip(input.iter())
                    .fold(0.0f32, |acc, x| acc + x.0.to_f32() * x.1.to_f32());
                truth[token * r + line] = product;
            }

            let mut ans = vec![0.0; t * r];
            for (token, line) in (0..t).cartesian_product(0..r) {
                let matrix = &matrix_u8[line * c..(line + 1) * c];
                let input = &input_f16[token * c..(token + 1) * c];
                let product =
                    matrix
                        .iter()
                        .zip(input.iter())
                        .enumerate()
                        .fold(0.0f32, |acc, (i, x)| {
                            let amp = absmax[(line * c + i) / NF4_BLOCK_SIZE];
                            acc + quant[*x.0 as usize] * amp.to_f32() * x.1.to_f32()
                        });
                ans[token * r + line] = product;
            }

            let ops = TensorOp::List(vec![TensorOp::matmul_vec_nf4(
                &matrix_u4_dev,
                &quant_dev,
                &absmax_dev,
                &input_dev,
                &output_dev,
                Activation::None,
            )?]);
            context.queue.submit(context.encode(&ops));
            let output_host: Vec<f32> = output_dev.back().await.to_vec();

            for (index, (&a, &b)) in itertools::zip_eq(&output_host, &ans).enumerate() {
                assert!(
                    is_approx_eps(a, b, 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            }

            let ops = TensorOp::List(vec![TensorOp::matmul_mat_nf4(
                &matrix_u4_dev,
                &quant_dev,
                &absmax_dev,
                &input_dev,
                &output_dev,
                Activation::None,
            )?]);
            context.queue.submit(context.encode(&ops));
            let output_host = output_dev.back().await.to_vec();

            for (index, (&a, &b)) in itertools::zip_eq(&output_host, &ans).enumerate() {
                assert!(
                    is_approx_eps(a, b, 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            }

            Ok(())
        }

        test_matmul_nf4_inner(&context, 2560, 2048, 64).await?;
        test_matmul_nf4_inner(&context, 320, 64, 320).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_lerp() -> Result<()> {
        let context = create_context().await?;
        fastrand::seed(42);

        const C: usize = 1000;
        const T: usize = 3;
        const B: usize = 2;

        let x = [(); C * T * B].map(|_| fastrand::f32() - 0.5).to_vec();
        let y = [(); C * T * B].map(|_| fastrand::f32() - 0.5).to_vec();
        let f = [(); C * T * B].map(|_| fastrand::f32()).to_vec();

        let shape = Shape::new(C, T, B, 1);
        let x_dev = context.tensor_from_data(shape, x.clone())?;
        let y_dev = context.tensor_from_data(shape, y.clone())?;
        let f_dev = context.tensor_from_data(shape, f.clone())?;

        let lerp = TensorOp::lerp(&x_dev, &y_dev, &f_dev, false)?;
        context.queue.submit(context.encode(&lerp));

        let y_host = y_dev.back().await.to_vec();

        let mut ans = vec![];
        for chunk in &itertools::multizip((&x, &y, &f)).chunks(C) {
            for (x, y, f) in chunk {
                ans.push(x * (1.0 - f) + y * f);
            }
        }

        for (index, (a, b)) in itertools::zip_eq(y_host, ans).enumerate() {
            assert!(
                is_approx(a, b),
                "Failed at index {index}, computed: {a} vs. answer: {b}"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_blit() -> Result<()> {
        let context = create_context().await?;
        fastrand::seed(42);

        let output = vec![0.0; 24];
        let output: TensorGpu<_, _> = context.tensor_from_data([4, 3, 2, 1], output)?;

        let mut ops = vec![];

        let input = (0..8).map(|x| x as f32).collect_vec();
        let input: TensorGpu<_, _> = context.tensor_from_data([4, 1, 2, 1], input)?;
        ops.push(TensorOp::blit(&input, output.view(.., 1, .., ..)?)?);

        let input = (8..12).map(|x| x as f32).collect_vec();
        let input: TensorGpu<_, _> = context.tensor_from_data([4, 1, 1, 1], input)?;
        ops.push(TensorOp::blit(&input, output.view(.., 2.., 1..2, ..)?)?);

        let ops = TensorOp::List(ops);
        context.queue.submit(context.encode(&ops));

        let output_host = output.back().await;
        let output_host = Vec::from(output_host);

        assert_eq!(
            output_host,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_transpose() -> Result<()> {
        let context = create_context().await?;
        fastrand::seed(42);

        let output = vec![0.0; 36];
        let output: TensorGpu<_, _> = context.tensor_from_data([4, 3, 3, 1], output)?;

        let input = (0..24).map(|x| x as f32).collect_vec();
        let input: TensorGpu<_, _> = context.tensor_from_data([4, 3, 2, 1], input)?;

        let ops = TensorOp::transpose(&input, output.view(.., ..2, .., ..)?)?;
        context.queue.submit(context.encode(&ops));

        let output_host = output.back().await;
        let output_host: Vec<f32> = Vec::from(output_host);

        assert_eq!(
            output_host,
            vec![
                0.0, 1.0, 2.0, 3.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0,
                16.0, 17.0, 18.0, 19.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0,
                23.0, 0.0, 0.0, 0.0, 0.0
            ]
        );

        Ok(())
    }
}
