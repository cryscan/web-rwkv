use std::{hash::Hash, sync::Arc};

use half::f16;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutEntry, BindingType,
    BufferBindingType, CommandEncoder, ComputePass, ComputePipeline, ShaderStages,
};

use super::{
    kind::{Kind, ReadWrite, Uniform},
    Shape, TensorError, TensorGpu, TensorScalar, TensorShape, TensorView,
};
use crate::{
    context::Macros,
    num::{Float, Scalar},
};

pub trait TensorCommand<T: Scalar, K: Kind> {
    fn copy_tensor(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
    ) -> Result<(), TensorError>;

    fn copy_tensor_batch(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
        batch: usize,
    ) -> Result<(), TensorError>;
}

impl<T: Scalar, K: Kind> TensorCommand<T, K> for CommandEncoder {
    fn copy_tensor(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
    ) -> Result<(), TensorError> {
        destination.check_shape(source.shape())?;
        let size = destination.size() as u64;
        self.copy_buffer_to_buffer(&source.buffer, 0, &destination.buffer, 0, size);
        Ok(())
    }

    fn copy_tensor_batch(
        &mut self,
        source: &TensorGpu<T, ReadWrite>,
        destination: &TensorGpu<T, K>,
        batch: usize,
    ) -> Result<(), TensorError> {
        destination.check_shape(Shape::new(source.shape[0], source.shape[1], 1, 1))?;
        if batch >= source.shape[2] {
            return Err(TensorError::BatchOutOfRange {
                batch,
                max: source.shape[2],
            });
        }
        let size = destination.size() as u64;
        let offset = (T::size() * source.shape[0] * source.shape[1] * batch) as u64;
        self.copy_buffer_to_buffer(&source.buffer, offset, &destination.buffer, 0, size);
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
                self.set_pipeline(pipeline);
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

    /// Define a `f32` macro with a given name.
    pub fn float(mut self, value: f32, name: impl Into<String>) -> Self {
        self.push((name.into(), format!("{}", value)));
        self
    }

    /// Define a `bool` macro with a given name.
    pub fn bool(mut self, value: bool, name: impl Into<String>) -> Self {
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
}

pub trait TensorOpHook: Hash + Send + Sync {}

pub enum TensorOp {
    Atom {
        pipeline: Arc<ComputePipeline>,
        bindings: Vec<BindGroup>,
        dispatch: [u32; 3],
    },
    List(Vec<TensorOp>),
}

impl TensorOp {
    pub const NF4_BLOCK_SIZE: u32 = 64;

    #[inline]
    fn block_count(count: u32, block_size: u32) -> u32 {
        (count + block_size - 1) / block_size
    }

    /// Softmax operator applied on `x`.
    pub fn softmax(x: &TensorGpu<impl Float, ReadWrite>) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "softmax",
            include_str!("../shaders/softmax.wgsl"),
            "softmax",
            None,
            Macros::new(BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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

        let shape = output.shape();
        tokens.check_shape(Shape::new(shape[1], shape[2], 1, 1))?;
        input.check_shape(Shape::new(shape[0], input.shape[1], 1, 1))?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "embed",
            include_str!("../shaders/embed.wgsl"),
            "embed",
            None,
            Macros::new(BLOCK_SIZE).tensor(output, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        s: Option<&TensorGpu<f32, ReadWrite>>,
        eps: f32,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        w.check_shape(Shape::new(shape[0], 1, 1, 1))?;
        b.check_shape(Shape::new(shape[0], 1, 1, 1))?;
        if let Some(s) = s {
            s.check_shape(Shape::new(4, shape[1], shape[2], 1))?;
        }

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "layer_norm",
            include_str!("../shaders/layer_norm.wgsl"),
            "layer_norm",
            None,
            Macros::new(BLOCK_SIZE)
                .float(eps, "EPS")
                .tensor(x, None)
                .bool(s.is_some(), "STATS"),
        );

        let mut entries = vec![
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
        ];
        if let Some(s) = s {
            entries.push(BindGroupEntry {
                binding: 4,
                resource: s.binding(),
            });
        }

        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
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

        let shape = x.shape();
        w.check_shape(Shape::new(shape[0], shape[1], 1, 1))?;
        b.check_shape(Shape::new(shape[0], shape[1], 1, 1))?;

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "group_norm",
            include_str!("../shaders/layer_norm.wgsl"),
            "group_norm",
            None,
            Macros::new(BLOCK_SIZE).float(eps, "EPS").tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        input: TensorView<impl Float>,
        output: TensorView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = output.shape();
        matrix.check_shape(Shape::new(input.shape()[0], shape[0], shape[2], 1))?;
        input.check_shape(Shape::new(matrix.shape[0], shape[1], shape[2], 1))?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "matmul_vec_fp16",
            include_str!("../shaders/matmul_vec_fp16.wgsl"),
            "matmul",
            None,
            Macros::new(BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
    /// - `mx` and `rx` shape: `[C, 1, B]`.
    /// - `my` and `ry` shape: `[R, 1, B]`.
    /// - `input` shape: `[C, T, B]`.
    /// - `output` shape: `[R, T, B]`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_vec_int8(
        matrix: &TensorGpu<u8, ReadWrite>,
        mx: &TensorGpu<f32, ReadWrite>,
        rx: &TensorGpu<f32, ReadWrite>,
        my: &TensorGpu<f32, ReadWrite>,
        ry: &TensorGpu<f32, ReadWrite>,
        input: TensorView<impl Float>,
        output: TensorView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = output.shape();
        matrix.check_shape(Shape::new(input.shape()[0], shape[0], shape[2], 1))?;
        input.check_shape(Shape::new(matrix.shape[0], shape[1], shape[2], 1))?;
        mx.check_shape(Shape::new(matrix.shape[0], shape[2], 1, 1))?;
        rx.check_shape(Shape::new(matrix.shape[0], shape[2], 1, 1))?;
        my.check_shape(Shape::new(matrix.shape[1], shape[2], 1, 1))?;
        ry.check_shape(Shape::new(matrix.shape[1], shape[2], 1, 1))?;

        let context = matrix.context();
        let pipeline = context.checkout_pipeline(
            "matmul_vec_int8",
            include_str!("../shaders/matmul_vec_int8.wgsl"),
            "matmul",
            None,
            Macros::new(BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
                    resource: mx.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: rx.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: my.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: ry.binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 9,
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
    /// - `absmax` shape: `[C / S, R, B]`.
    /// - `input` shape: `[C, T, B]`.
    /// - `output` shape: `[R, T, B]`.
    pub fn matmul_vec_nf4(
        matrix: &TensorGpu<u8, ReadWrite>,
        quant: &TensorGpu<f32, Uniform>,
        absmax: &TensorGpu<f16, ReadWrite>,
        input: TensorView<f16>,
        output: TensorView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = output.shape();
        matrix.check_shape(Shape::new(input.shape()[0] / 2, shape[0], shape[2], 1))?;
        input.check_shape(Shape::new(input.shape()[0], shape[1], shape[2], 1))?;
        absmax.check_shape(Shape::new(
            input.shape()[0] / Self::NF4_BLOCK_SIZE as usize,
            shape[0],
            shape[2],
            1,
        ))?;

        let context = matrix.context();
        let pipeline = context.checkout_pipeline(
            "matmul_vec_nf4",
            include_str!("../shaders/matmul_vec_nf4.wgsl"),
            "matmul",
            None,
            Macros::new(BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        matrix: TensorView<f16>,
        input: TensorView<f16>,
        output: TensorView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let shape = output.shape();
        matrix.check_shape(Shape::new(matrix.shape()[0], shape[0], shape[2], 1))?;
        input.check_shape(Shape::new(input.shape()[0], shape[1], shape[2], 1))?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "matmul_mat_fp16",
            include_str!("../shaders/matmul_mat_fp16.wgsl"),
            "matmul",
            None,
            Macros::new(BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        matrix: TensorView<u8>,
        mx: &TensorGpu<f32, ReadWrite>,
        rx: &TensorGpu<f32, ReadWrite>,
        my: &TensorGpu<f32, ReadWrite>,
        ry: &TensorGpu<f32, ReadWrite>,
        input: TensorView<f16>,
        output: TensorView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let shape = output.shape();
        matrix.check_shape(Shape::new(matrix.shape()[0], shape[0], shape[2], 1))?;
        input.check_shape(Shape::new(input.shape()[0], shape[1], shape[2], 1))?;
        mx.check_shape(Shape::new(matrix.shape()[0], shape[2], 1, 1))?;
        rx.check_shape(Shape::new(matrix.shape()[0], shape[2], 1, 1))?;
        my.check_shape(Shape::new(matrix.shape()[1], shape[2], 1, 1))?;
        ry.check_shape(Shape::new(matrix.shape()[1], shape[2], 1, 1))?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "matmul_mat_int8",
            include_str!("../shaders/matmul_mat_int8.wgsl"),
            "matmul",
            None,
            Macros::new(BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
                    resource: mx.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: rx.binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: my.binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: ry.binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: matrix.binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: input.binding(),
                },
                BindGroupEntry {
                    binding: 9,
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
        matrix: TensorView<u8>,
        quant: &TensorGpu<f32, Uniform>,
        absmax: &TensorGpu<f16, ReadWrite>,
        input: TensorView<f16>,
        output: TensorView<impl Float>,
        active: Activation,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let shape = output.shape();
        matrix.check_shape(Shape::new(matrix.shape()[0], shape[0], shape[2], 1))?;
        input.check_shape(Shape::new(input.shape()[0], shape[1], shape[2], 1))?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "matmul_mat_nf4",
            include_str!("../shaders/matmul_mat_nf4.wgsl"),
            "matmul",
            None,
            Macros::new(BLOCK_SIZE)
                .nf4(Self::NF4_BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT"))
                .custom(active, Some("ACT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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

    pub fn add(
        input: TensorView<impl Float>,
        output: TensorView<impl Float>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = output.shape();
        input
            .check_shape(Shape::new(shape[0], 1, shape[2], shape[3]))
            .or(input.check_shape(shape))?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "add",
            include_str!("../shaders/add.wgsl"),
            "add",
            None,
            Macros::new(BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        time_mix: TensorView<impl Float>,
        sx: TensorView<f32>,
        input: &TensorGpu<impl Float, ReadWrite>,
        output: &TensorGpu<impl Float, ReadWrite>,
        reversed: bool,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = output.shape();
        input.check_shape(shape)?;
        time_mix
            .check_shape(Shape::new(shape[0], 1, 1, 1))
            .or(time_mix.check_shape(shape))?;
        sx.check_shape(Shape::new(shape[0], sx.shape()[1], sx.shape()[2], 1))?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "token_shift",
            include_str!("../shaders/token_shift.wgsl"),
            "token_shift",
            None,
            Macros::new(BLOCK_SIZE)
                .tensor(&time_mix, Some("TIME_MIX"))
                .tensor(input, Some("IN"))
                .tensor(output, Some("OUT"))
                .bool(reversed, "REVERSED"),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: time_mix.meta_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: sx.meta_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cursors.binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: time_mix.binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: sx.binding(),
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
                Self::block_count(shape[0] as u32 / 4, BLOCK_SIZE),
                shape[1] as u32,
                1,
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn time_mix_v4<T: Float>(
        cursors: &TensorGpu<u32, ReadWrite>,
        time_decay: &TensorGpu<f32, ReadWrite>,
        time_first: &TensorGpu<f32, ReadWrite>,
        state: TensorView<f32>,
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
        time_decay.check_shape(Shape::new(shape[0], 1, 1, 1))?;
        time_first.check_shape(Shape::new(shape[0], 1, 1, 1))?;
        state.check_shape(Shape::new(shape[0], 4, state.shape()[2], 1))?;

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "time_mix_v4",
            include_str!("../shaders/time_mix_v4.wgsl"),
            "time_mix",
            None,
            Macros::new(BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        state: TensorView<f32>,
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
        time_decay.check_shape(Shape::new(shape[0], shape[1], 1, 1))?;
        time_first.check_shape(Shape::new(shape[0], shape[1], 1, 1))?;
        state.check_shape(Shape::new(dim, shape[0] + 1, state.shape()[2], 1))?;

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "time_mix_v5",
            include_str!("../shaders/time_mix_v5.wgsl"),
            "time_mix",
            None,
            Macros::new(BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        state: TensorView<f32>,
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
        time_first.check_shape(Shape::new(shape[0], shape[1], 1, 1))?;
        state.check_shape(Shape::new(dim, shape[0] + 1, state.shape()[2], 1))?;

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "time_mix_v6",
            include_str!("../shaders/time_mix_v6.wgsl"),
            "time_mix",
            None,
            Macros::new(BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
            Macros::new(BLOCK_SIZE)
                .tensor(input, Some("IN"))
                .tensor(output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
            Macros::new(BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
            Macros::new(BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
            Macros::new(BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        state: TensorView<f32>,
        r: &TensorGpu<T, ReadWrite>,
        v: &TensorGpu<T, ReadWrite>,
        x: &TensorGpu<T, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        v.check_shape(shape)?;
        r.check_shape(shape)?;
        state.check_shape(Shape::new(shape[0], 1, state.shape()[2], 1))?;

        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "channel_mix",
            include_str!("../shaders/channel_mix.wgsl"),
            "channel_mix",
            None,
            Macros::new(BLOCK_SIZE).tensor(x, None),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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

    /// Copy the content of `input` into `output`, given an `offset`.
    pub fn blit(
        input: TensorView<impl Float>,
        output: TensorView<impl Float>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = output.shape();
        input.check_shape(shape)?;

        let context = input.context();
        let pipeline = context.checkout_pipeline(
            "blit",
            include_str!("../shaders/blit.wgsl"),
            "blit",
            None,
            Macros::new(BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        input: TensorView<impl Float>,
        output: TensorView<impl Float>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = input.shape();
        output.check_shape(Shape::new(shape[0], shape[2], shape[1], 1))?;

        let context = input.context();
        let pipeline = context.checkout_pipeline(
            "transpose",
            include_str!("../shaders/blit.wgsl"),
            "transpose",
            None,
            Macros::new(BLOCK_SIZE)
                .tensor(&input, Some("IN"))
                .tensor(&output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        const BLOCK_SIZE: u32 = 128;

        let shape = output.shape();
        input.check_shape(shape)?;
        factor.check_shape(Shape::new(4, 1, 1, 1))?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "blend",
            include_str!("../shaders/blend.wgsl"),
            "blend",
            None,
            Macros::new(BLOCK_SIZE)
                .tensor(input, Some("IN"))
                .tensor(output, Some("OUT")),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
                BindGroupEntry {
                    binding: 4,
                    resource: factor.binding(),
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

    pub fn blend_lora(
        factor: &TensorGpu<f32, Uniform>,
        xa: TensorView<f16>,
        xb: TensorView<f16>,
        output: TensorView<f16>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 8;

        let shape = output.shape();
        factor.check_shape(Shape::new(4, 1, 1, 1))?;
        xa.check_shape(Shape::new(xa.shape()[0], shape[0], shape[2], 1))?;
        xb.check_shape(Shape::new(xb.shape()[0], shape[1], shape[2], 1))?;

        let context = output.context();
        let pipeline = context.checkout_pipeline(
            "blend_lora",
            include_str!("../shaders/blend_lora.wgsl"),
            "blend_lora",
            None,
            Macros::new(BLOCK_SIZE),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = x.shape();
        let context = x.context();
        let pipeline = context.checkout_pipeline(
            "discount",
            include_str!("../shaders/discount.wgsl"),
            "discount",
            None,
            Macros::new(BLOCK_SIZE)
                .tensor(x, None)
                .float(factor, "FACTOR"),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
        mx: &TensorGpu<f32, ReadWrite>,
        rx: &TensorGpu<f32, ReadWrite>,
        my: &TensorGpu<f32, ReadWrite>,
        ry: &TensorGpu<f32, ReadWrite>,
        output: &TensorGpu<u8, ReadWrite>,
    ) -> Result<Self, TensorError> {
        const BLOCK_SIZE: u32 = 128;

        let shape = output.shape();
        input.check_shape(shape)?;
        mx.check_shape(Shape::new(shape[0], 1, 1, 1))?;
        rx.check_shape(Shape::new(shape[0], 1, 1, 1))?;
        my.check_shape(Shape::new(shape[1], 1, 1, 1))?;
        ry.check_shape(Shape::new(shape[1], 1, 1, 1))?;

        let context = output.context();

        let layout: Option<&[BindGroupLayoutEntry]> = Some(&[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]);

        let entries: &[BindGroupEntry] = &[
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
                resource: mx.binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: rx.binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: my.binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: ry.binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: output.binding(),
            },
        ];

        let op = |entry_point: &'static str, dispatch| -> Result<Self, TensorError> {
            let pipeline = context.checkout_pipeline(
                format!("quant_mat_int8_{}", entry_point),
                include_str!("../shaders/quant_mat_int8.wgsl"),
                entry_point,
                layout,
                Macros::new(BLOCK_SIZE),
            );
            let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            })];
            Ok(Self::Atom {
                pipeline,
                bindings,
                dispatch,
            })
        };

        let my = op("compute_my", [1, shape[1] as u32, 1])?;
        let ry = op("compute_ry", [1, shape[1] as u32, 1])?;
        let mx = op("compute_mx", [1, shape[0] as u32 / 4, 1])?;
        let rx = op("compute_rx", [1, shape[0] as u32 / 4, 1])?;
        let quantize = op("quantize", [shape[0] as u32 / 4, shape[1] as u32, 1])?;

        if shape[1] > shape[0] {
            Ok(Self::List(vec![my, mx, rx, ry, quantize]))
        } else {
            Ok(Self::List(vec![mx, my, rx, ry, quantize]))
        }
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
            Macros::new(BLOCK_SIZE).nf4(Self::NF4_BLOCK_SIZE),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
            Macros::new(BLOCK_SIZE).nf4(Self::NF4_BLOCK_SIZE),
        );
        let bindings = vec![context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
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
    use wgpu::PowerPreference;
    // use wgpu_profiler::GpuProfiler;

    use super::{TensorOp, TensorPass};
    use crate::{
        context::{Context, ContextBuilder, Instance},
        // model::matrix::Matrix,
        tensor::{
            matrix::Matrix,
            ops::{Activation, TensorCommand},
            Shape, TensorGpu, TensorInit, TensorShape,
        },
    };

    fn is_approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= f32::max(f32::EPSILON, f32::max(a.abs(), b.abs()) * f32::EPSILON)
    }

    fn is_approx_eps(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= f32::max(eps, f32::max(a.abs(), b.abs()) * eps)
    }

    #[tokio::main]
    async fn create_context() -> Result<Context> {
        let instance = Instance::new();
        let adapter = instance.adapter(PowerPreference::HighPerformance).await?;
        let context = ContextBuilder::new(adapter)
            // .with_features(Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES)
            .build()
            .await?;
        Ok(context)
    }

    #[test]
    fn test_copy() -> Result<()> {
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
        fastrand::seed(42);

        let x = vec![0.0, 1.5, 2.0, -1.0];
        let shape = Shape::new(x.len(), 1, 1, 1);

        let x_device: TensorGpu<_, _> = context.tensor_from_data(shape, x.clone())?;
        let x_map = context.tensor_init(x_device.shape());

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_tensor(&x_device, &x_map)?;
        context.queue.submit(Some(encoder.finish()));

        let x_host = x_map.back();
        let x_host = Vec::from(x_host);

        assert_eq!(x, x_host);
        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<()> {
        let context = match create_context() {
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
        let x_map = context.tensor_init(x_dev.shape());

        let softmax = TensorOp::softmax(&x_dev)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&softmax);
        drop(pass);

        encoder.copy_tensor(&x_dev, &x_map)?;
        context.queue.submit(Some(encoder.finish()));

        let x_host = x_map.back();
        let x_host = Vec::from(x_host);

        let mut ans = vec![];
        for x in &x.into_iter().chunks(C) {
            let x = x.collect_vec().into_iter();
            let max = x.clone().reduce(f32::max).unwrap_or_default();
            let x = x.map(|x| (x - max).exp());
            let sum: f32 = x.clone().sum();
            let mut x: Vec<_> = x.map(|x| x / sum).collect();
            ans.append(&mut x);
        }

        itertools::zip_eq(x_host.into_iter(), ans.into_iter())
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
        let context = match create_context() {
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
        let x_dev = TensorGpu::from_data(&context, shape, &x)?;
        let x_map = context.tensor_init(shape);

        let shape = Shape::new(C, 1, 1, 1);
        let w_dev = TensorGpu::from_data(&context, shape, &w[..1000])?;
        let b_dev = TensorGpu::from_data(&context, shape, &b[..1000])?;

        let shape = Shape::new(4, T, B, 1);
        let s_dev = context.tensor_init(shape);
        let s_map = context.tensor_init(shape);

        let layer_norm = TensorOp::layer_norm(&w_dev, &b_dev, &x_dev, Some(&s_dev), EPS)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&layer_norm);
        drop(pass);

        encoder.copy_tensor(&x_dev, &x_map)?;
        encoder.copy_tensor(&s_dev, &s_map)?;
        context.queue.submit(Some(encoder.finish()));

        let x_host = x_map.back().to_vec();
        let s_host = s_map.back().to_vec();

        let mut ans = vec![];
        let mut ans_stats = vec![];
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
            let deviation = (m2 / count as f32 + EPS).sqrt();
            ans_stats.append(&mut vec![mean, 1.0 / deviation, 0.0, 0.0]);

            let mut x: Vec<_> = chunk
                .into_iter()
                .map(|((x, w), b)| (x - mean) / deviation * w.to_f32() + b.to_f32())
                .collect();
            ans.append(&mut x);
        }

        itertools::zip_eq(s_host.into_iter(), ans_stats.into_iter())
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    is_approx_eps(a, b, 1.0e-3),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        itertools::zip_eq(x_host.into_iter(), ans.into_iter())
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    is_approx_eps(a, b, 1.0e-3),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        Ok(())
    }

    #[test]
    fn test_matmul() -> Result<()> {
        let context = match create_context() {
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
        let input_f32_dev = TensorGpu::from_data(&context, input_shape, input_f32.clone())?;
        let input_f16_dev: TensorGpu<f16, _> = context.tensor_init(input_shape);
        let output_dev = TensorGpu::init(&context, output_shape);
        let output_map = TensorGpu::init(&context, output_shape);

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

        encoder.copy_tensor(&output_dev, &output_map)?;
        context.queue.submit(Some(encoder.finish()));

        let output_host = output_map.back();
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

        itertools::zip_eq(output_host.into_iter(), ans.into_iter())
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
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
        fastrand::seed(42);

        const C: usize = 14336;
        const R: usize = 4096;
        const T: usize = 32;

        // let matrix_shape = Shape::new(C, R, 1, 1);
        let input_shape = Shape::new(C, T, 1, 1);
        let output_shape = Shape::new(R, T, 2, 1);

        let matrix_f16 = vec![(); R * C]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .map(f16::from_f32)
            .collect_vec();

        let mut matrix_u8 = matrix_f16
            .clone()
            .into_iter()
            .map(f16::to_f32)
            .collect_vec();

        let mut mx = vec![f32::MAX; C];
        let mut my = vec![f32::MAX; R];
        let mut rx = vec![f32::MIN; C];
        let mut ry = vec![f32::MIN; R];

        if R > C {
            for i in 0..R {
                (0..C).for_each(|j| my[i] = my[i].min(matrix_u8[C * i + j]));
                (0..C).for_each(|j| matrix_u8[C * i + j] -= my[i]);
            }
            for j in 0..C {
                (0..R).for_each(|i| mx[j] = mx[j].min(matrix_u8[C * i + j]));
                (0..R).for_each(|i| matrix_u8[C * i + j] -= mx[j]);
            }
        } else {
            for j in 0..C {
                (0..R).for_each(|i| mx[j] = mx[j].min(matrix_u8[C * i + j]));
                (0..R).for_each(|i| matrix_u8[C * i + j] -= mx[j]);
            }
            for i in 0..R {
                (0..C).for_each(|j| my[i] = my[i].min(matrix_u8[C * i + j]));
                (0..C).for_each(|j| matrix_u8[C * i + j] -= my[i]);
            }
        }
        for j in 0..C {
            (0..R).for_each(|i| rx[j] = rx[j].max(matrix_u8[C * i + j]));
            (0..R).for_each(|i| matrix_u8[C * i + j] /= rx[j]);
        }
        for i in 0..R {
            (0..C).for_each(|j| ry[i] = ry[i].max(matrix_u8[C * i + j]));
            (0..C).for_each(|j| matrix_u8[C * i + j] /= ry[i]);
        }

        let matrix_f16_dev = context.tensor_from_data(Shape::new(C, R, 1, 1), &matrix_f16)?;
        let matrix_quant = Matrix::quant_u8(&matrix_f16_dev)?;
        let (matrix_u8_dev, mx_dev, my_dev, rx_dev, ry_dev) = match matrix_quant {
            Matrix::Int8 { w, mx, rx, my, ry } => (w, mx, my, rx, ry),
            _ => unreachable!(),
        };

        let matrix_u8_map = context.tensor_init(Shape::new(C, R, 1, 1));
        let mx_map = context.tensor_init(Shape::new(C, 1, 1, 1));
        let my_map = context.tensor_init(Shape::new(R, 1, 1, 1));
        let rx_map = context.tensor_init(Shape::new(C, 1, 1, 1));
        let ry_map = context.tensor_init(Shape::new(R, 1, 1, 1));

        let mut encoder = context.device.create_command_encoder(&Default::default());

        encoder.copy_tensor(&matrix_u8_dev, &matrix_u8_map)?;
        encoder.copy_tensor(&mx_dev, &mx_map)?;
        encoder.copy_tensor(&my_dev, &my_map)?;
        encoder.copy_tensor(&rx_dev, &rx_map)?;
        encoder.copy_tensor(&ry_dev, &ry_map)?;

        context.queue.submit(Some(encoder.finish()));

        let matrix_u8_host = matrix_u8_map.back().to_vec();
        let mx_host = mx_map.back().to_vec();
        let my_host = my_map.back().to_vec();
        let rx_host = rx_map.back().to_vec();
        let ry_host = ry_map.back().to_vec();

        let matrix_u8_host = matrix_u8_host
            .into_iter()
            .map(|x| (x as f32) / 255.0)
            .collect_vec();

        let output = [
            matrix_u8_host.clone(),
            mx_host.clone(),
            my_host.clone(),
            rx_host.clone(),
            ry_host.clone(),
        ]
        .concat();
        let ans = [matrix_u8, mx, my, rx, ry].concat();

        itertools::zip_eq(output.into_iter(), ans.into_iter())
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    is_approx_eps(a, b, 0.005),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        let input_f32 = vec![(); C * T]
            .into_iter()
            .map(|_| 10.0 * (fastrand::f32() - 0.5))
            .collect_vec();
        let input_f16 = input_f32.iter().copied().map(f16::from_f32).collect_vec();

        let input_f32_dev = TensorGpu::from_data(&context, input_shape, input_f32.clone())?;
        let input_f16_dev: TensorGpu<f16, _> = context.tensor_init(input_shape);
        let output_dev = TensorGpu::init(&context, output_shape);
        let output_map = TensorGpu::init(&context, output_shape);

        let ops = TensorOp::List(vec![
            TensorOp::blit(
                input_f32_dev.view(.., .., .., ..)?,
                input_f16_dev.view(.., .., .., ..)?,
            )?,
            TensorOp::matmul_vec_int8(
                &matrix_u8_dev,
                &mx_dev,
                &rx_dev,
                &my_dev,
                &ry_dev,
                input_f32_dev.view(.., .., .., ..)?,
                output_dev.view(.., .., 0..1, ..)?,
                Activation::None,
            )?,
            TensorOp::matmul_mat_int8(
                matrix_u8_dev.view(.., .., .., ..)?,
                &mx_dev,
                &rx_dev,
                &my_dev,
                &ry_dev,
                input_f16_dev.view(.., .., .., ..)?,
                output_dev.view(.., .., 1.., ..)?,
                Activation::None,
            )?,
        ]);

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);

        encoder.copy_tensor(&output_dev, &output_map)?;
        context.queue.submit(Some(encoder.finish()));

        let output_host = output_map.back();
        let output_host = Vec::from(output_host);

        context.device.poll(wgpu::MaintainBase::Wait);

        let mut ans = vec![0.0; output_host.len()];
        for token in 0..T {
            for line in 0..R {
                let matrix = &matrix_u8_host[(line * C)..(line + 1) * C];
                let input = &input_f32[token * C..(token + 1) * C];
                let product = itertools::multizip((matrix, &mx_host, &rx_host, input)).fold(
                    0.0f32,
                    |acc, (m, mx, rx, x)| {
                        let my = my_host[line];
                        let ry = ry_host[line];
                        let m = m * rx * ry + mx + my;
                        acc + m * x
                    },
                );
                ans[token * R + line] = product;

                let input = &input_f16[token * C..(token + 1) * C];
                let product = itertools::multizip((matrix, &mx_host, &rx_host, input)).fold(
                    0.0f32,
                    |acc, (m, mx, rx, x)| {
                        let my = my_host[line];
                        let ry = ry_host[line];
                        let m = m * rx * ry + mx + my;
                        acc + m * x.to_f32()
                    },
                );
                ans[(T + token) * R + line] = product;
            }
        }

        itertools::zip_eq(output_host.into_iter(), ans.into_iter())
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
        let context = match create_context() {
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
            let mut matrix_u8: Vec<u8> = vec![];
            let mut matrix_u4: Vec<u8> = vec![];
            let mut absmax = vec![];
            matrix_u8.resize(matrix.len(), 0);
            matrix_u4.resize(matrix.len() / 2, 0);
            absmax.resize(matrix.len() / NF4_BLOCK_SIZE, f16::ZERO);

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
        let input_dev = TensorGpu::from_data(&context, input_shape, input_f16.clone())?;
        let output_dev = TensorGpu::init(&context, output_shape);
        let output_map = TensorGpu::init(&context, output_shape);

        let matrix_u4_map = TensorGpu::init(&context, matrix_u4_shape);
        let absmax_map = TensorGpu::init(&context, absmax_shape);

        // let ops = TensorOp::List(vec![
        //     TensorOp::quantize_mat_nf4(&matrix_f16_dev, &quant_dev, &absmax_dev, &matrix_u4_dev)?,
        //     TensorOp::matmul_vec_nf4(
        //         &matrix_u4_dev,
        //         &quant_dev,
        //         &absmax_dev,
        //         input_dev.view(.., .., .., ..)?,
        //         output_dev.view(.., .., .., ..)?,
        //         Activation::None,
        //     )?,
        // ]);

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

        encoder.copy_tensor(&matrix_u4_dev, &matrix_u4_map)?;
        encoder.copy_tensor(&absmax_dev, &absmax_map)?;
        encoder.copy_tensor(&output_dev, &output_map)?;
        context.queue.submit(Some(encoder.finish()));

        let matrix_u4_host = matrix_u4_map.back().to_vec();
        let absmax_host = absmax_map.back().to_vec();
        let output_host = output_map.back().to_vec();

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

        itertools::zip_eq(matrix_u4_host.into_iter(), matrix_u4.into_iter())
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    a == b,
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        itertools::zip_eq(absmax_host.into_iter(), absmax.into_iter())
            .enumerate()
            .for_each(|(index, (a, b))| {
                assert!(
                    is_approx_eps(a.to_f32(), b.to_f32(), 0.01),
                    "Failed at index {index}, computed: {a} vs. answer: {b}"
                );
            });

        itertools::zip_eq(output_host.into_iter(), ans.into_iter())
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
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
        fastrand::seed(42);

        let output = vec![0.0; 24];
        let output = TensorGpu::from_data(&context, Shape::new(4, 3, 2, 1), output)?;

        let map = TensorGpu::init(&context, output.shape());
        let mut ops = vec![];

        let input = (0..8).map(|x| x as f32).collect_vec();
        let input = TensorGpu::from_data(&context, Shape::new(4, 1, 2, 1), input)?;
        ops.push(TensorOp::blit(
            input.view(.., .., .., ..)?,
            output.view(.., 1, .., ..)?,
        )?);

        let input = (8..12).map(|x| x as f32).collect_vec();
        let input = TensorGpu::from_data(&context, Shape::new(4, 1, 1, 1), input)?;
        let input = input.view(.., .., .., ..)?;
        ops.push(TensorOp::blit(input, output.view(.., 2.., 1..2, ..)?)?);

        let ops = TensorOp::List(ops);

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);

        encoder.copy_tensor(&output, &map)?;
        context.queue.submit(Some(encoder.finish()));

        let output_host = map.back();
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
        let context = match create_context() {
            Ok(context) => context,
            Err(_) => return Ok(()),
        };
        fastrand::seed(42);

        let output = vec![0.0; 36];
        let output = TensorGpu::from_data(&context, Shape::new(4, 3, 3, 1), output)?;

        let map = TensorGpu::init(&context, output.shape());

        let input = (0..24).map(|x| x as f32).collect_vec();
        let input = TensorGpu::from_data(&context, Shape::new(4, 3, 2, 1), input)?;

        let ops = TensorOp::transpose(input.view(.., .., .., ..)?, output.view(.., ..2, .., ..)?)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);

        encoder.copy_tensor(&output, &map)?;
        context.queue.submit(Some(encoder.finish()));

        let output_host = map.back();
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
