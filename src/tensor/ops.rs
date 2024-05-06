use std::{hash::Hash, sync::Arc};

use half::f16;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, CommandEncoder, ComputePass};

use super::{
    kind::{Kind, ReadWrite, Uniform},
    Shape, TensorError, TensorGpu, TensorGpuView, TensorScalar, TensorShape,
};
use crate::{
    context::{CachedPipeline, Macros},
    num::{Float, Scalar},
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

pub trait TensorPass<'a> {
    fn execute_tensor_op(&mut self, op: &'a TensorOp);
}

impl<'b, 'a: 'b> TensorPass<'a> for ComputePass<'b> {
    fn execute_tensor_op(&mut self, op: &'a TensorOp) {
        match op {
            TensorOp::Atom {
                pipeline,
                bindings,
                dispatch,
            } => {
                self.set_pipeline(&pipeline.pipeline);
                for (index, bind_group) in bindings.iter().enumerate() {
                    self.set_bind_group(index as u32, bind_group, &[])
                }
                self.dispatch_workgroups(dispatch[0], dispatch[1], dispatch[2]);
            }
            TensorOp::List(ops) => {
                ops.iter().for_each(|op| self.execute_tensor_op(op));
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activation {
    #[default]
    None,
    SquaredRelu,
    Tanh,
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::None => write!(f, "NONE"),
            Activation::SquaredRelu => write!(f, "SQUARED_RELU"),
            Activation::Tanh => write!(f, "TANH"),
        }
    }
}

impl Macros {
    /// Define a `u32` macro `NF4_BLOCK_SIZE`.
    pub fn nf4(mut self, block_size: u32) -> Self {
        self.push(("NF4_BLOCK_SIZE".into(), format!("{}u", block_size)));
        self
    }

    /// Define a `u32` macro `NF4_BLOCK_SIZE`.
    pub fn int8(mut self, block_size: u32) -> Self {
        self.push(("INT8_BLOCK_SIZE".into(), format!("{}u", block_size)));
        self
    }

    /// Define a `f32` macro with a given name.
    pub fn f32(mut self, name: impl Into<String>, value: f32) -> Self {
        self.push((name.into(), format!("{}", value)));
        self
    }

    /// Define a `usize` macro with a given name.
    pub fn u32(mut self, name: impl Into<String>, value: u32) -> Self {
        self.push((name.into(), format!("{}u", value)));
        self
    }

    /// Define a `bool` macro with a given name.
    pub fn bool(mut self, name: impl Into<String>, value: bool) -> Self {
        match value {
            true => {
                self.push((name.into(), Default::default()));
                self
            }
            false => self,
        }
    }

    /// Define the macro specifies input/output tensor data type.
    pub fn tensor<T: Float>(
        mut self,
        _tensor: &impl TensorScalar<T = T>,
        prefix: Option<&'_ str>,
    ) -> Self {
        match prefix {
            None => self.push((T::DEF.into(), Default::default())),
            Some(prefix) => self.push((format!("{}_{}", prefix, T::DEF), Default::default())),
        }
        self
    }

    /// Define a macro with custom display name and prefix.
    pub fn custom(mut self, value: impl std::fmt::Display, prefix: Option<&'_ str>) -> Self {
        match prefix {
            None => self.push((format!("{}", value), Default::default())),
            Some(prefix) => self.push((format!("{}_{}", prefix, value), Default::default())),
        }
        self
    }

    /// Add a define when `condition` is true.
    pub fn define(mut self, name: impl Into<String>, condition: bool) -> Self {
        if condition {
            self.push((name.into(), Default::default()))
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
        bindings: Vec<BindGroup>,
        dispatch: [u32; 3],
    },
    List(Vec<TensorOp>),
}

impl TensorOp {
    pub const NF4_BLOCK_SIZE: u32 = 64;
    pub const INT8_BLOCK_SIZE: u32 = 128;

    #[inline]
    fn block_count(count: u32, block_size: u32) -> u32 {
        (count + block_size - 1) / block_size
    }

    #[inline]
    pub fn empty() -> Self {
        Self::List(vec![])
    }

    /// Softmax operator applied on `x`.
    pub fn softmax(x: &TensorGpu<impl Float, ReadWrite>) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        let context = x.context();
        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            "softmax",
            include_str!("../shaders/softmax.wgsl"),
            "softmax",
            None,
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            "softmax",
            include_str!("../shaders/subgroup/softmax.wgsl"),
            "softmax",
            None,
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
            ],
        })];

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

        let shape = {
            let [index, token, batch, _] = *output.shape();
            let [_, vocab, _, _] = *input.shape();
            tokens.check_shape([token, batch, 1, 1])?;
            input.check_shape([index, vocab, 1, 1])?;
            output.check_shape([index, token, batch, 1])?;
            output.shape()
        };

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "embed",
            include_str!("../shaders/embed.wgsl"),
            "embed",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(output, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: tokens.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
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

        let shape = {
            let [index, token, batch, _] = *x.shape();
            x.check_shape([index, token, batch, 1])?;
            w.check_shape([index, 1, 1, 1])?;
            b.check_shape([index, 1, 1, 1])?;
            x.shape()
        };

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "layer_norm",
            include_str!("../shaders/layer_norm.wgsl"),
            "layer_norm",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", eps),
        );

        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: w.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: b.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: x.binding(),
                },
            ],
        })];

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

        let shape = {
            let [index, head, token, _] = *x.shape();
            x.check_shape([index, head, token, 1])?;
            w.check_shape([index, head, 1, 1])?;
            b.check_shape([index, head, 1, 1])?;
            x.shape()
        };

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "group_norm",
            include_str!("../shaders/layer_norm.wgsl"),
            "group_norm",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", eps),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: w.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: b.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [1, shape[1] as u32, shape[2] as u32],
        })
    }

    /// Recenter `x` to be zero-mean.
    pub fn recenter(x: &TensorGpu<impl Float, ReadWrite>) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();

        let context = x.context();
        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            "recenter",
            include_str!("../shaders/rms_norm.wgsl"),
            "recenter",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", 0.0),
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            "recenter",
            include_str!("../shaders/subgroup/rms_norm.wgsl"),
            "recenter",
            None,
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", 0.0),
        );

        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: x.binding(),
                },
            ],
        })];

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

        let shape = {
            let [index, token, batch, _] = *x.shape();
            x.check_shape([index, token, batch, 1])?;
            w.check_shape([index, 1, 1, 1])?;
            b.check_shape([index, 1, 1, 1])?;
            x.shape()
        };

        let context = x.context();
        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            "rms_norm",
            include_str!("../shaders/rms_norm.wgsl"),
            "rms_norm",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", eps),
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            "rms_norm",
            include_str!("../shaders/subgroup/rms_norm.wgsl"),
            "rms_norm",
            None,
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("EPS", eps),
        );

        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: w.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: b.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: x.binding(),
                },
            ],
        })];

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
    pub fn matmul_vec_fp16(
        matrix: &TensorGpu<f16, ReadWrite>,
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = {
            let [m, n, b, _] = *output.shape();
            let [k, _, _, _] = *input.shape();
            matrix.check_shape([k, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        let context = output.context();
        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            "matmul_vec_fp16",
            include_str!("../shaders/matmul_vec_fp16.wgsl"),
            "matmul",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            "matmul_vec_fp16",
            include_str!("../shaders/subgroup/matmul_vec_fp16.wgsl"),
            "matmul",
            None,
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: output.binding(),
                },
            ],
        })];

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
    pub fn matmul_vec_int8(
        matrix: &TensorGpu<u8, ReadWrite>,
        minmax: &TensorGpu<f16, ReadWrite>,
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = {
            let [m, n, b, _] = *output.shape();
            let [k, _, _, _] = *input.shape();
            minmax.check_shape([(k << 1) / Self::INT8_BLOCK_SIZE as usize, m, b, 1])?;
            matrix.check_shape([k, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        let context = matrix.context();
        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            "matmul_vec_int8",
            include_str!("../shaders/matmul_vec_int8.wgsl"),
            "matmul",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            "matmul_vec_int8",
            include_str!("../shaders/matmul_vec_int8.wgsl"),
            "matmul",
            None,
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .int8(Self::INT8_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: minmax.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: output.binding(),
                },
            ],
        })];

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
    pub fn matmul_vec_nf4(
        matrix: &TensorGpu<u8, ReadWrite>,
        quant: &TensorGpu<f32, Uniform>,
        absmax: &TensorGpu<f16, ReadWrite>,
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = {
            let [m, n, b, _] = *output.shape();
            let [k, _, _, _] = *input.shape();
            absmax.check_shape([k / Self::NF4_BLOCK_SIZE as usize, m, b, 1])?;
            matrix.check_shape([k >> 1, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        let context = matrix.context();
        #[cfg(not(feature = "subgroup-ops"))]
        let pipeline = context.checkout_pipeline(
            "matmul_vec_nf4",
            include_str!("../shaders/matmul_vec_nf4.wgsl"),
            "matmul",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        #[cfg(feature = "subgroup-ops")]
        let pipeline = context.checkout_pipeline(
            "matmul_vec_nf4",
            include_str!("../shaders/matmul_vec_nf4.wgsl"),
            "matmul",
            None,
            Macros::new()
                .subgroup(context.min_subgroup_size(), context.max_subgroup_size())
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: quant.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: absmax.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: output.binding(),
                },
            ],
        })];

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
    /// Note: `K` must be multiples of 128; `M` and `N` must be multiples of 4.
    pub fn matmul_mat_fp16(
        matrix: TensorGpuView<f16>,
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let shape = {
            let [m, n, b, _] = *output.shape();
            let [k, _, _, _] = *input.shape();
            matrix.check_shape([k, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "matmul_mat_fp16",
            include_str!("../shaders/matmul_mat_fp16.wgsl"),
            "matmul",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(Self::block_count(shape[0] as u32, 4), BLOCK_SIZE),
                Self::block_count(Self::block_count(shape[1] as u32, 4), BLOCK_SIZE),
                shape[2] as u32,
            ],
        })
    }

    /// Int8 matrix-matrix multiplication.
    /// - `matrix` shape: `[K, M, B]`.
    /// - `input` shape: `[K, N, B]`.
    /// - `output` shape: `[M, N, B]`.
    ///
    /// Note: `K` must be multiples of 128; `M` and `N` must be multiples of 4.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mat_int8(
        matrix: TensorGpuView<u8>,
        minmax: &TensorGpu<f16, ReadWrite>,
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let shape = {
            let [m, n, b, _] = *output.shape();
            let [k, _, _, _] = *input.shape();
            minmax.check_shape([(k << 1) / Self::INT8_BLOCK_SIZE as usize, m, b, 1])?;
            matrix.check_shape([k, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "matmul_mat_int8",
            include_str!("../shaders/matmul_mat_int8.wgsl"),
            "matmul",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .int8(Self::INT8_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: minmax.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(Self::block_count(shape[0] as u32, 4), BLOCK_SIZE),
                Self::block_count(Self::block_count(shape[1] as u32, 4), BLOCK_SIZE),
                shape[2] as u32,
            ],
        })
    }

    /// NFloat4 matrix-matrix multiplication.
    /// - `matrix` shape: `[K, M, B]`.
    /// - `input` shape: `[K, N, B]`.
    /// - `output` shape: `[M, N, B]`.
    ///
    /// Note: `K` must be multiples of 256; `M` and `N` must be multiples of 8.
    pub fn matmul_mat_nf4(
        matrix: TensorGpuView<u8>,
        quant: &TensorGpu<f32, Uniform>,
        absmax: &TensorGpu<f16, ReadWrite>,
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let shape = {
            let [m, n, b, _] = *output.shape();
            let [k, _, _, _] = *input.shape();
            absmax.check_shape([k / Self::NF4_BLOCK_SIZE as usize, m, b, 1])?;
            matrix.check_shape([k >> 1, m, b, 1])?;
            input.check_shape([k, n, b, 1])?;
            output.check_shape([m, n, b, 1])?;
            output.shape()
        };

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "matmul_mat_nf4",
            include_str!("../shaders/matmul_mat_nf4.wgsl"),
            "matmul",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: matrix.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: quant.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: absmax.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(Self::block_count(shape[0] as u32, 4), BLOCK_SIZE),
                Self::block_count(Self::block_count(shape[1] as u32, 4), BLOCK_SIZE),
                shape[2] as u32,
            ],
        })
    }

    /// Add `input` to `output`.
    /// - `input` shape: `[C, 1, B]` or `[C, T, B]`.
    /// - `output` shape: `[C, T, B]`.
    pub fn add(
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = {
            let [index, token, batch, _] = *output.shape();
            input
                .check_shape([index, 1, batch, 1])
                .or(input.check_shape([index, token, batch, 1]))?;
            output.check_shape([index, token, batch, 1])?;
            output.shape()
        };

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "add",
            include_str!("../shaders/binary.wgsl"),
            "add",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    /// Multiply `input` to `output`.
    /// - `input` shape: `[C, 1, B]` or `[C, T, B]`.
    /// - `output` shape: `[C, T, B]`.
    pub fn mul(
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = {
            let [index, token, batch, _] = *output.shape();
            input
                .check_shape([index, 1, batch, 1])
                .or(input.check_shape([index, token, batch, 1]))?;
            output.check_shape([index, token, batch, 1])?;
            output.shape()
        };

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "mul",
            include_str!("../shaders/binary.wgsl"),
            "mul",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn token_shift(
        cursors: &TensorGpu<u32, ReadWrite>,
        time_mix: TensorGpuView<impl Float>,
        state: TensorGpuView<f32>,
        input: &TensorGpu<impl Float, ReadWrite>,
        output: &TensorGpu<impl Float, ReadWrite>,
        reversed: bool,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = {
            let [index, token, count, _] = *output.shape();
            let [_, head, batch, _] = *state.shape();
            input
                .check_shape([index, token, 1, 1])
                .or(input.check_shape([index, token, count, 1]))?;
            time_mix
                .check_shape([index, 1, 1, 1])
                .or(time_mix.check_shape([index, token, count, 1]))?;
            state.check_shape([index, head, batch, 1])?;
            output.shape()
        };

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "token_shift",
            include_str!("../shaders/token_shift.wgsl"),
            "token_shift",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&time_mix, Some("TIME_MIX"))
                .tensor(input, Some("IN"))
                .tensor(output, Some("OUT"))
                .bool("REVERSED", reversed),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: time_mix.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: cursors.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: time_mix.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: state.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn time_mix_v4<T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        time_decay: &TensorGpu<f32, ReadWrite>,
        time_first: &TensorGpu<f32, ReadWrite>,
        state: TensorGpuView<f32>,
        k: &TensorGpu<T, ReadWrite>,
        v: &TensorGpu<T, ReadWrite>,
        r: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        k.check_shape(shape)?;
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        time_decay.check_shape([shape[0], 1, 1, 1])?;
        time_first.check_shape([shape[0], 1, 1, 1])?;
        state.check_shape([shape[0], 4, state.shape()[2], 1])?;

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "time_mix_v4",
            include_str!("../shaders/time_mix_v4.wgsl"),
            "time_mix",
            None,
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: state.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cursors.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: time_decay.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: time_first.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: state.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: k.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: v.binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: r.binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE), 1, 1],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn time_mix_v5<T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        time_decay: &TensorGpu<f32, ReadWrite>,
        time_first: &TensorGpu<f32, ReadWrite>,
        state: TensorGpuView<f32>,
        k: &TensorGpu<T, ReadWrite>,
        v: &TensorGpu<T, ReadWrite>,
        r: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 32;

        let shape = x.shape();
        let dim = shape[0] * shape[1];

        k.check_shape(shape)?;
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        time_decay.check_shape([shape[0], shape[1], 1, 1])?;
        time_first.check_shape([shape[0], shape[1], 1, 1])?;
        state.check_shape([dim, shape[0] + 1, state.shape()[2], 1])?;

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "time_mix_v5",
            include_str!("../shaders/time_mix_v5.wgsl"),
            "time_mix",
            None,
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: state.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cursors.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: time_decay.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: time_first.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: state.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: k.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: v.binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: r.binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [Self::block_count(dim as u32 / 4, BLOCK_SIZE), 1, 1],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn time_mix_v6<T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        time_decay: &TensorGpu<f32, ReadWrite>,
        time_first: &TensorGpu<f32, ReadWrite>,
        state: TensorGpuView<f32>,
        k: &TensorGpu<T, ReadWrite>,
        v: &TensorGpu<T, ReadWrite>,
        r: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 32;

        let shape = x.shape();
        let dim = shape[0] * shape[1];

        k.check_shape(shape)?;
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        time_decay.check_shape(shape)?;
        time_first.check_shape([shape[0], shape[1], 1, 1])?;
        state.check_shape([dim, shape[0] + 1, state.shape()[2], 1])?;

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "time_mix_v6",
            include_str!("../shaders/time_mix_v6.wgsl"),
            "time_mix",
            None,
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: state.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cursors.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: time_decay.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: time_first.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: state.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: k.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: v.binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: r.binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [Self::block_count(dim as u32 / 4, BLOCK_SIZE), 1, 1],
        })
    }

    pub fn silu(
        input: &TensorGpu<impl Float, ReadWrite>,
        output: &TensorGpu<impl Float, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = output.shape();
        input.check_shape(shape)?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "silu",
            include_str!("../shaders/silu.wgsl"),
            "silu",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(input, Some("IN"))
                .tensor(output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn tanh(x: &TensorGpu<impl Float, ReadWrite>) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "tanh",
            include_str!("../shaders/activation.wgsl"),
            "act_tanh",
            None,
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn opposite_exp(x: &TensorGpu<f32, ReadWrite>) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "opposite_exp",
            include_str!("../shaders/activation.wgsl"),
            "opposite_exp",
            None,
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn stable_exp(x: &TensorGpu<f32, ReadWrite>) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "stable_exp",
            include_str!("../shaders/activation.wgsl"),
            "stable_exp",
            None,
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn squared_relu(x: &TensorGpu<f32, ReadWrite>) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "squared_relu",
            include_str!("../shaders/activation.wgsl"),
            "squared_relu",
            None,
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    pub fn channel_mix<T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        state: TensorGpuView<f32>,
        r: &TensorGpu<T, ReadWrite>,
        v: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        state.check_shape([shape[0], 1, state.shape()[2], 1])?;

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "channel_mix",
            include_str!("../shaders/channel_mix.wgsl"),
            "channel_mix",
            None,
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: state.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cursors.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: state.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: r.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: v.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                1,
            ],
        })
    }

    /// Copy the content of `input` into `output` of the same shape.
    pub fn blit(
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
    ) -> Result<Self, TensorError> {
        let shape = output.shape();
        input.check_shape(shape)?;

        let block_size = match shape[1] {
            x if x < 8 => [128, 1],
            _ => [16, 16],
        };

        let context = input.context();
        let pipeline = context.checkout_pipeline(
            "blit",
            include_str!("../shaders/blit.wgsl"),
            "blit",
            None,
            Macros::new()
                .u32("BLOCK_SIZE_X", block_size[0])
                .u32("BLOCK_SIZE_Y", block_size[1])
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, block_size[0]),
                Self::block_count(shape[1] as u32, block_size[1]),
                shape[2] as u32,
            ],
        })
    }

    /// Repeat the content of `input` into `output` along the token and batch axes.
    pub fn broadcast(
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = output.shape();
        input.check_shape([shape[0], input.shape()[1], input.shape()[2], 1])?;

        let context = input.context();
        let pipeline = context.checkout_pipeline(
            "broadcast",
            include_str!("../shaders/reshape.wgsl"),
            "broadcast",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        })
    }

    /// Swap the `token` and `batch` axes.
    pub fn transpose(
        input: TensorGpuView<impl Float>,
        output: TensorGpuView<impl Float>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = input.shape();
        output.check_shape([shape[0], shape[2], shape[1], 1])?;

        let context = input.context();
        let pipeline = context.checkout_pipeline(
            "transpose",
            include_str!("../shaders/reshape.wgsl"),
            "transpose",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
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
        let shape = output.shape();
        input.check_shape(shape)?;
        factor.check_shape([4, 1, 1, 1])?;

        let block_size = match shape[1] {
            x if x < 8 => [128, 1],
            _ => [16, 16],
        };

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "blend",
            include_str!("../shaders/blend.wgsl"),
            "blend",
            None,
            Macros::new()
                .u32("BLOCK_SIZE_X", block_size[0])
                .u32("BLOCK_SIZE_Y", block_size[1])
                .tensor(input, Some("IN"))
                .tensor(output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: factor.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, block_size[0]),
                Self::block_count(shape[1] as u32, block_size[1]),
                shape[2] as u32,
            ],
        })
    }

    pub fn blend_lora(
        factor: &TensorGpu<f32, Uniform>,
        xa: TensorGpuView<f16>,
        xb: TensorGpuView<f16>,
        output: TensorGpuView<f16>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let shape = output.shape();
        factor.check_shape([4, 1, 1, 1])?;
        xa.check_shape([xa.shape()[0], shape[0], shape[2], 1])?;
        xb.check_shape([xb.shape()[0], shape[1], shape[2], 1])?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "blend_lora",
            include_str!("../shaders/blend_lora.wgsl"),
            "blend_lora",
            None,
            Macros::new().u32("BLOCK_SIZE", BLOCK_SIZE),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: xa.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: xb.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.meta_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: factor.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: xa.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: xb.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: output.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(Self::block_count(shape[0] as u32, 4), BLOCK_SIZE),
                Self::block_count(Self::block_count(shape[1] as u32, 4), BLOCK_SIZE),
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

        let shape = x.shape();
        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "discount",
            include_str!("../shaders/discount.wgsl"),
            "discount",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .tensor(x, None)
                .f32("FACTOR", factor)
                .f32("BIAS", bias),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.binding(),
                },
            ],
        })];

        Ok(Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
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
        let minmax_shape = Shape::new(
            (shape[0] << 1) / Self::INT8_BLOCK_SIZE as usize,
            shape[1],
            shape[2],
            shape[3],
        );

        input.check_shape(shape)?;
        minmax.check_shape(minmax_shape)?;

        let pipeline = context.checkout_pipeline(
            "quant_mat_int8_minmax",
            include_str!("../shaders/quant_mat_int8.wgsl"),
            "compute_minmax",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .int8(Self::INT8_BLOCK_SIZE),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 1,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: minmax.binding(),
                },
            ],
        })];
        let compute_minmax = Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(minmax_shape[0] as u32, BLOCK_SIZE),
                minmax_shape[1] as u32,
                minmax_shape[2] as u32,
            ],
        };

        let pipeline = context.checkout_pipeline(
            "quant_mat_int8",
            include_str!("../shaders/quant_mat_int8.wgsl"),
            "quantize",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .int8(Self::INT8_BLOCK_SIZE),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                BindGroupEntry {
                    binding: 1,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: minmax.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.binding(),
                },
            ],
        })];
        let quantize = Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32, BLOCK_SIZE),
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
        let absmax_shape = Shape::new(
            input_shape[0] / Self::NF4_BLOCK_SIZE as usize,
            shape[1],
            shape[2],
            shape[3],
        );

        input.check_shape(input_shape)?;
        absmax.check_shape(absmax_shape)?;

        let absmax_f32: TensorGpu<f32, ReadWrite> = context.tensor_init(absmax_shape);

        let pipeline = context.checkout_pipeline(
            "quant_mat_nf4_absmax",
            include_str!("../shaders/quant_mat_nf4.wgsl"),
            "compute_absmax",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                // BindGroupEntry {
                //     binding: 0,
                //     resource: absmax_f32.meta_binding(),
                // },
                // BindGroupEntry {
                //     binding: 1,
                //     resource: quant.binding(),
                // },
                BindGroupEntry {
                    binding: 2,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: absmax_f32.binding(),
                },
                // BindGroupEntry {
                //     binding: 4,
                //     resource: output.binding(),
                // },
            ],
        })];
        let compute_absmax = Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(absmax_shape[0] as u32, BLOCK_SIZE),
                absmax_shape[1] as u32,
                absmax_shape[2] as u32,
            ],
        };

        let pipeline = context.checkout_pipeline(
            "quant_mat_nf4",
            include_str!("../shaders/quant_mat_nf4.wgsl"),
            "quantize",
            None,
            Macros::new()
                .u32("BLOCK_SIZE", BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &[
                // BindGroupEntry {
                //     binding: 0,
                //     resource: output.meta_binding(),
                // },
                BindGroupEntry {
                    binding: 1,
                    resource: quant.binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: absmax_f32.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: output.binding(),
                },
            ],
        })];
        let quantize = Self::Atom {
            pipeline,
            bindings,
            dispatch: [
                Self::block_count(shape[0] as u32, BLOCK_SIZE),
                shape[1] as u32,
                shape[2] as u32,
            ],
        };

        let quantize_absmax = Self::blit(
            absmax_f32.view(.., .., .., ..)?,
            absmax.view(.., .., .., ..)?,
        )?;

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

    use super::{TensorOp, TensorPass};
    use crate::{
        context::{Context, ContextBuilder, InstanceExt},
        tensor::{ops::Activation, Shape, TensorGpu},
    };

    fn is_approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= f32::max(f32::EPSILON, f32::max(a.abs(), b.abs()) * f32::EPSILON)
    }

    fn is_approx_eps(a: f32, b: f32, eps: f32) -> bool {
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

    #[test]
    fn test_softmax() -> Result<()> {
        let context = match pollster::block_on(create_context()) {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
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

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&softmax);
        drop(pass);
        context.queue.submit(Some(encoder.finish()));

        let x_host = x_dev.back_local().to_vec();

        let mut ans = vec![];
        for x in &x.into_iter().chunks(C) {
            let x = x.collect_vec().into_iter();
            let max = x.clone().reduce(f32::max).unwrap_or_default();
            let x = x.map(|x| (x - max).exp());
            let sum: f32 = x.clone().sum();
            let mut x: Vec<_> = x.map(|x| x / sum).collect();
            ans.append(&mut x);
        }

        itertools::zip_eq(x_host, ans)
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    is_approx(a, b),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        Ok(())
    }

    #[test]
    fn test_layer_norm() -> Result<()> {
        let context = match pollster::block_on(create_context()) {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
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

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&layer_norm);
        drop(pass);
        context.queue.submit(Some(encoder.finish()));

        let x_host = x_dev.back_local().to_vec();
        // let s_host = s_dev.back_local().to_vec();

        // test recenter and rms norm
        let shape = Shape::new(C, T, B, 1);
        let x_dev = context.tensor_from_data(shape, x.clone())?;
        let ops = TensorOp::List(vec![
            TensorOp::recenter(&x_dev)?,
            TensorOp::rms_norm(&w_dev, &b_dev, &x_dev, EPS)?,
        ]);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);
        context.queue.submit(Some(encoder.finish()));

        let x_rms_host = x_dev.back_local().to_vec();

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

            let mut x: Vec<_> = chunk
                .into_iter()
                .map(|((x, w), b)| (x - mean) * deviation * w.to_f32() + b.to_f32())
                .collect();
            ans.append(&mut x);
        }

        // itertools::zip_eq(s_host.into_iter(), ans_stats.into_iter())
        //     .enumerate()
        //     .for_each(|(index, (a, b))| {
        //         assert!(
        //             is_approx_eps(a, b, 1.0e-3),
        //             "Failed at index {index}, computed: {a} vs. answer: {b}"
        //         );
        //     });

        itertools::zip_eq(x_host, ans.iter())
            .enumerate()
            .for_each(|(index, (a, &b))| {
                assert!(
                    is_approx_eps(a, b, 1.0e-3),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        itertools::zip_eq(x_rms_host, ans.iter())
            .enumerate()
            .for_each(|(index, (a, &b))| {
                assert!(
                    is_approx_eps(a, b, 1.0e-3),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        Ok(())
    }

    #[test]
    fn test_matmul() -> Result<()> {
        let context = match pollster::block_on(create_context()) {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
        fastrand::seed(42);
        // let mut profiler = GpuProfiler::new(&context.adapter, &context.device, &context.queue, 1);

        const C: usize = 2560;
        const R: usize = 2048;
        const T: usize = 32;
        const B: usize = 2;

        let matrix = vec![(); C * R * B]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .map(f16::from_f32)
            .collect_vec();
        let input_f32 = vec![(); C * T * B]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .collect_vec();
        let input_f16 = input_f32.iter().copied().map(f16::from_f32).collect_vec();

        let matrix_shape = Shape::new(C, R, B, 1);
        let input_shape = Shape::new(C, T, B, 1);
        let output_shape = Shape::new(R, T, 2 * B, 1);

        let matrix_dev = context.tensor_from_data(matrix_shape, matrix.clone())?;
        let input_f32_dev: TensorGpu<_, _> =
            context.tensor_from_data(input_shape, input_f32.clone())?;
        let input_f16_dev: TensorGpu<f16, _> = context.tensor_init(input_shape);
        let output_dev: TensorGpu<_, _> = context.tensor_init(output_shape);

        let ops = TensorOp::List(vec![
            TensorOp::blit(
                input_f32_dev.view(.., .., .., ..)?,
                input_f16_dev.view(.., .., .., ..)?,
            )?,
            TensorOp::matmul_vec_fp16(
                &matrix_dev,
                input_f32_dev.view(.., .., .., ..)?,
                output_dev.view(.., .., 0..B, ..)?,
                Activation::None,
            )?,
            TensorOp::matmul_mat_fp16(
                matrix_dev.view(.., .., .., ..)?,
                input_f16_dev.view(.., .., .., ..)?,
                output_dev.view(.., .., B.., ..)?,
                Activation::None,
            )?,
        ]);

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);

        // profiler.resolve_queries(&mut encoder);
        context.queue.submit(Some(encoder.finish()));

        let output_host = output_dev.back_local();
        let output_host = Vec::from(output_host);

        // profiler.end_frame().unwrap();
        context.device.poll(wgpu::MaintainBase::Wait);

        // if let Some(results) = profiler.process_finished_frame() {
        //     wgpu_profiler::chrometrace::write_chrometrace(
        //         std::path::Path::new(&format!("./trace/matmul_{T}.json")),
        //         &results,
        //     )
        //     .expect("failed to write trace");
        // }

        let mut ans = vec![0.0; output_host.len()];
        for batch in 0..B {
            for token in 0..T {
                for line in 0..R {
                    let matrix = &matrix[((batch * R + line) * C)..((batch * R + line) + 1) * C];
                    let input = &input_f32[(batch * T + token) * C..((batch * T + token) + 1) * C];
                    let product = matrix
                        .iter()
                        .zip(input.iter())
                        .fold(0.0f32, |acc, x| acc + x.0.to_f32() * *x.1);
                    ans[(batch * T + token) * R + line] = product;

                    let input = &input_f16[(batch * T + token) * C..((batch * T + token) + 1) * C];
                    let product = matrix
                        .iter()
                        .zip(input.iter())
                        .fold(0.0f32, |acc, x| acc + x.0.to_f32() * x.1.to_f32());
                    ans[((B + batch) * T + token) * R + line] = product;
                }
            }
        }

        itertools::zip_eq(output_host, ans)
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    is_approx_eps(a, b, 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        Ok(())
    }

    #[test]
    fn test_matmul_int8() -> Result<()> {
        let context = match pollster::block_on(create_context()) {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
        fastrand::seed(42);

        const C: usize = 2560;
        const R: usize = 2048;
        const T: usize = 64;
        const INT8_BLOCK_SIZE: usize = TensorOp::INT8_BLOCK_SIZE as usize;

        let matrix = vec![(); C * R]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .map(f16::from_f32)
            .collect_vec();
        let input_f32 = vec![(); C * T]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .collect_vec();
        let input_f16 = input_f32.iter().copied().map(f16::from_f32).collect_vec();

        let (matrix_u8, min, max) = {
            let mut matrix_u8: Vec<u8> = vec![0; matrix.len()];
            let mut min = vec![f16::MAX; matrix.len() / INT8_BLOCK_SIZE];
            let mut max = vec![f16::MIN; matrix.len() / INT8_BLOCK_SIZE];

            for (i, (min, max)) in min.iter_mut().zip_eq(max.iter_mut()).enumerate() {
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

        let minmax_shape = Shape::new(C / INT8_BLOCK_SIZE * 2, R, 1, 1);
        let matrix_shape = Shape::new(C, R, 1, 1);
        let input_shape = Shape::new(C, T, 1, 1);
        let output_shape = Shape::new(R, T, 1, 1);

        let minmax_dev = context.tensor_init(minmax_shape);
        let matrix_f16_dev = context.tensor_from_data(matrix_shape, matrix.clone())?;

        let matrix_u8_dev = context.tensor_init(matrix_shape);
        let input_dev: TensorGpu<_, _> =
            context.tensor_from_data(input_shape, input_f16.clone())?;
        let output_dev: TensorGpu<_, _> = context.tensor_init(output_shape);

        let ops = TensorOp::List(vec![
            TensorOp::quantize_mat_int8(&matrix_f16_dev, &minmax_dev, &matrix_u8_dev)?,
            TensorOp::matmul_mat_int8(
                matrix_u8_dev.view(.., .., .., ..)?,
                &minmax_dev,
                input_dev.view(.., .., .., ..)?,
                output_dev.view(.., .., .., ..)?,
                Activation::None,
            )?,
        ]);

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);
        context.queue.submit(Some(encoder.finish()));

        let matrix_u8_host = matrix_u8_dev.back_local().to_vec();
        let output_host = output_dev.back_local().to_vec();

        // let mut truth = vec![0.0; output_host.len()];
        // for token in 0..T {
        //     for line in 0..R {
        //         let matrix = &matrix[line * C..(line + 1) * C];
        //         let input = &input_f16[token * C..(token + 1) * C];
        //         let product = matrix
        //             .iter()
        //             .zip(input.iter())
        //             .fold(0.0f32, |acc, x| acc + x.0.to_f32() * x.1.to_f32());
        //         truth[token * R + line] = product;
        //     }
        // }

        let mut ans = vec![0.0; output_host.len()];
        for token in 0..T {
            for line in 0..R {
                let matrix = &matrix_u8_host[line * C..(line + 1) * C];
                let input = &input_f16[token * C..(token + 1) * C];
                let product =
                    matrix
                        .iter()
                        .zip_eq(input.iter())
                        .enumerate()
                        .fold(0.0f32, |acc, (i, x)| {
                            let min = min[(line * C + i) / INT8_BLOCK_SIZE].to_f32();
                            let max = max[(line * C + i) / INT8_BLOCK_SIZE].to_f32();
                            let value = (*x.0 as f32) / 255.0;
                            acc + (value * (max - min) + min) * x.1.to_f32()
                        });
                ans[token * R + line] = product;
            }
        }

        itertools::zip_eq(matrix_u8_host, matrix_u8)
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    a.abs_diff(b) < 2,
                    // a == b,
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        itertools::zip_eq(output_host, ans)
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    is_approx_eps(a, b, 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        Ok(())
    }

    #[test]
    fn test_matmul_nf4() -> Result<()> {
        let context = match pollster::block_on(create_context()) {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
        fastrand::seed(42);

        const C: usize = 2560;
        const R: usize = 2048;
        const T: usize = 64;
        const NF4_BLOCK_SIZE: usize = TensorOp::NF4_BLOCK_SIZE as usize;

        fn normal() -> f32 {
            let u = fastrand::f32();
            let v = fastrand::f32();
            (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
        }

        let matrix = vec![(); C * R]
            .into_iter()
            .map(|_| normal())
            .map(f16::from_f32)
            .collect_vec();
        let input_f32 = vec![(); C * T]
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
            let mut absmax = vec![f16::ZERO; matrix.len() / NF4_BLOCK_SIZE];

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
        let absmax_shape = Shape::new(C / NF4_BLOCK_SIZE, R, 1, 1);
        let matrix_f16_shape = Shape::new(C, R, 1, 1);
        let matrix_u4_shape = Shape::new(C / 2, R, 1, 1);
        let input_shape = Shape::new(C, T, 1, 1);
        let output_shape = Shape::new(R, T, 1, 1);

        let quant_dev = context.tensor_from_data(quant_shape, quant.to_vec())?;
        let absmax_dev = context.tensor_init(absmax_shape);
        let matrix_f16_dev = context.tensor_from_data(matrix_f16_shape, matrix.clone())?;

        let matrix_u4_dev = context.tensor_init(matrix_u4_shape);
        let input_dev: TensorGpu<_, _> =
            context.tensor_from_data(input_shape, input_f16.clone())?;
        let output_dev: TensorGpu<_, _> = context.tensor_init(output_shape);

        let ops = TensorOp::List(vec![
            TensorOp::quantize_mat_nf4(&matrix_f16_dev, &quant_dev, &absmax_dev, &matrix_u4_dev)?,
            TensorOp::matmul_mat_nf4(
                matrix_u4_dev.view(.., .., .., ..)?,
                &quant_dev,
                &absmax_dev,
                input_dev.view(.., .., .., ..)?,
                output_dev.view(.., .., .., ..)?,
                Activation::None,
            )?,
        ]);

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);
        context.queue.submit(Some(encoder.finish()));

        let matrix_u4_host = matrix_u4_dev.back_local().to_vec();
        let absmax_host = absmax_dev.back_local().to_vec();
        let output_host = output_dev.back_local().to_vec();

        let mut truth = vec![0.0; output_host.len()];
        for token in 0..T {
            for line in 0..R {
                let matrix = &matrix[line * C..(line + 1) * C];
                let input = &input_f16[token * C..(token + 1) * C];
                let product = matrix
                    .iter()
                    .zip(input.iter())
                    .fold(0.0f32, |acc, x| acc + x.0.to_f32() * x.1.to_f32());
                truth[token * R + line] = product;
            }
        }

        let mut ans = vec![0.0; output_host.len()];
        for token in 0..T {
            for line in 0..R {
                let matrix = &matrix_u8[line * C..(line + 1) * C];
                let input = &input_f16[token * C..(token + 1) * C];
                let product =
                    matrix
                        .iter()
                        .zip(input.iter())
                        .enumerate()
                        .fold(0.0f32, |acc, (i, x)| {
                            let amp = absmax[(line * C + i) / NF4_BLOCK_SIZE];
                            acc + quant[*x.0 as usize] * amp.to_f32() * x.1.to_f32()
                        });
                ans[token * R + line] = product;
            }
        }

        itertools::zip_eq(matrix_u4_host, matrix_u4)
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    a == b,
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        itertools::zip_eq(absmax_host, absmax)
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    is_approx_eps(a.to_f32(), b.to_f32(), 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        itertools::zip_eq(output_host, ans)
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    is_approx_eps(a, b, 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        Ok(())
    }

    #[test]
    fn test_blit() -> Result<()> {
        let context = match pollster::block_on(create_context()) {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
        fastrand::seed(42);

        let output = vec![0.0; 24];
        let output: TensorGpu<_, _> = context.tensor_from_data([4, 3, 2, 1], output)?;

        let mut ops = vec![];

        let input = (0..8).map(|x| x as f32).collect_vec();
        let input: TensorGpu<_, _> = context.tensor_from_data([4, 1, 2, 1], input)?;
        ops.push(TensorOp::blit(
            input.view(.., .., .., ..)?,
            output.view(.., 1, .., ..)?,
        )?);

        let input = (8..12).map(|x| x as f32).collect_vec();
        let input: TensorGpu<_, _> = context.tensor_from_data([4, 1, 1, 1], input)?;
        let input = input.view(.., .., .., ..)?;
        ops.push(TensorOp::blit(input, output.view(.., 2.., 1..2, ..)?)?);

        let ops = TensorOp::List(ops);

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);
        context.queue.submit(Some(encoder.finish()));

        let output_host = output.back_local();
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

    #[test]
    fn test_transpose() -> Result<()> {
        let context = match pollster::block_on(create_context()) {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
        fastrand::seed(42);

        let output = vec![0.0; 36];
        let output: TensorGpu<_, _> = context.tensor_from_data([4, 3, 3, 1], output)?;

        let input = (0..24).map(|x| x as f32).collect_vec();
        let input: TensorGpu<_, _> = context.tensor_from_data([4, 3, 2, 1], input)?;

        let ops = TensorOp::transpose(input.view(.., .., .., ..)?, output.view(.., ..2, .., ..)?)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);
        context.queue.submit(Some(encoder.finish()));

        let output_host = output.back_local();
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
