use std::{convert::Infallible, sync::Arc};

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor};

use super::{
    loader::Loader, matrix::Matrix, FromBuilder, ModelBuilder, ModelError, ModelInfo, Quantization,
    StateBuilder,
};
use crate::{
    context::Context,
    tensor::{
        cache::ResourceCache,
        ops::{TensorCommand, TensorOp, TensorPass},
        shape::{Shape, TensorDimension},
        IntoPackedCursors, ReadBack, ReadWrite, TensorCpu, TensorError, TensorGpu, TensorInit,
        TensorReshape, TensorShape, TensorStack, TensorView,
    },
};

#[derive(Debug)]
pub struct Model<'a> {
    context: Context,
    info: ModelInfo,

    /// The head matrix is too big for a storage buffer so it's divided into chunks.
    head_chunk_size: usize,
    /// To prevent the GPU device from lost, this limits the maximum batch-token it processes one time.
    token_chunk_size: usize,

    tensor: ModelTensor<'a>,
    runtime_cache: ResourceCache<usize, Runtime>,
    output_cache: ResourceCache<usize, Output>,
    softmax_cache: ResourceCache<usize, Softmax>,
    stack_cache: ResourceCache<usize, TensorGpu<u32, ReadWrite>>,
}

#[derive(Debug)]
struct ModelTensor<'a> {
    embed: Embed<'a>,
    head: Head,
    layers: Vec<Layer>,
}

#[derive(Debug)]
struct LayerNorm {
    w: TensorGpu<f16, ReadWrite>,
    b: TensorGpu<f16, ReadWrite>,
}

#[derive(Debug)]
struct Att {
    time_decay: TensorGpu<f32, ReadWrite>,
    time_first: TensorGpu<f32, ReadWrite>,

    time_mix_k: TensorGpu<f16, ReadWrite>,
    time_mix_v: TensorGpu<f16, ReadWrite>,
    time_mix_r: TensorGpu<f16, ReadWrite>,
    time_mix_g: TensorGpu<f16, ReadWrite>,

    w_k: Matrix,
    w_v: Matrix,
    w_r: Matrix,
    w_g: Matrix,
    w_o: Matrix,

    group_norm: LayerNorm,
}

#[derive(Debug)]
struct Ffn {
    time_mix_k: TensorGpu<f16, ReadWrite>,
    time_mix_r: TensorGpu<f16, ReadWrite>,

    w_k: Matrix,
    w_v: Matrix,
    w_r: Matrix,
}

#[derive(Debug)]
struct Layer {
    att_layer_norm: LayerNorm,
    ffn_layer_norm: LayerNorm,
    att: Att,
    ffn: Ffn,
}

#[derive(Debug)]
struct Embed<'a> {
    layer_norm: LayerNorm,
    w: TensorCpu<'a, f16>,
}

#[derive(Debug)]
struct Head {
    layer_norm: LayerNorm,
    w: Vec<TensorGpu<f16, ReadWrite>>,
}

/// Runtime buffers.
#[derive(Debug)]
struct Runtime {
    cursors: TensorGpu<u32, ReadWrite>,
    input: TensorGpu<f32, ReadWrite>,

    att_x: TensorGpu<f32, ReadWrite>,
    att_kx: TensorGpu<f32, ReadWrite>,
    att_vx: TensorGpu<f32, ReadWrite>,
    att_rx: TensorGpu<f32, ReadWrite>,
    att_gx: TensorGpu<f32, ReadWrite>,
    att_k: TensorGpu<f32, ReadWrite>,
    att_v: TensorGpu<f32, ReadWrite>,
    att_r: TensorGpu<f32, ReadWrite>,
    att_g: TensorGpu<f32, ReadWrite>,
    att_o: TensorGpu<f32, ReadWrite>,

    ffn_x: TensorGpu<f32, ReadWrite>,
    ffn_kx: TensorGpu<f32, ReadWrite>,
    ffn_rx: TensorGpu<f32, ReadWrite>,
    ffn_k: TensorGpu<f32, ReadWrite>,
    ffn_v: TensorGpu<f32, ReadWrite>,
    ffn_r: TensorGpu<f32, ReadWrite>,
}

impl Runtime {
    pub fn new(context: &Context, info: &ModelInfo, num_token: usize) -> Self {
        let shape = Shape::new(info.num_emb, num_token, 1, 1);
        let cursors_shape = Shape::new(num_token, 1, 1, 1);
        let hidden_shape = Shape::new(info.num_hidden, num_token, 1, 1);

        Self {
            cursors: context.tensor_init(cursors_shape),
            input: context.tensor_init(shape),
            att_x: context.tensor_init(shape),
            att_kx: context.tensor_init(shape),
            att_vx: context.tensor_init(shape),
            att_rx: context.tensor_init(shape),
            att_gx: context.tensor_init(shape),
            att_k: context.tensor_init(shape),
            att_v: context.tensor_init(shape),
            att_r: context.tensor_init(shape),
            att_g: context.tensor_init(shape),
            att_o: context.tensor_init(shape),
            ffn_x: context.tensor_init(shape),
            ffn_kx: context.tensor_init(shape),
            ffn_rx: context.tensor_init(shape),
            ffn_k: context.tensor_init(hidden_shape),
            ffn_v: context.tensor_init(shape),
            ffn_r: context.tensor_init(shape),
        }
    }
}

#[derive(Debug)]
struct Output {
    head_x: TensorGpu<f32, ReadWrite>,
    head_o: TensorGpu<f32, ReadWrite>,
    map: TensorGpu<f32, ReadBack>,
}

impl Output {
    pub fn new(context: &Context, info: &ModelInfo, num_batch: usize) -> Self {
        let head_shape = Shape::new(info.num_emb, num_batch, 1, 1);
        let output_shape = Shape::new(info.num_vocab, num_batch, 1, 1);

        Self {
            head_x: context.tensor_init(head_shape),
            head_o: context.tensor_init(output_shape),
            map: context.tensor_init(output_shape),
        }
    }
}

#[derive(Debug)]
struct Softmax {
    buffer: TensorGpu<f32, ReadWrite>,
    map: TensorGpu<f32, ReadBack>,
}

impl Softmax {
    pub fn new(context: &Context, info: &ModelInfo, num_batch: usize) -> Self {
        let shape = Shape::new(info.num_vocab, 1, num_batch, 1);
        Self {
            buffer: context.tensor_init(shape),
            map: context.tensor_init(shape),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelState {
    context: Context,
    info: ModelInfo,
    max_batch: usize,
    chunk_size: usize,
    head_size: usize,
    state: Vec<TensorGpu<f32, ReadWrite>>,
}

impl ModelState {
    fn att(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let chunk = layer / self.chunk_size;
        let offset = layer % self.chunk_size;
        let head_size = self.info.num_emb / self.info.num_head;

        let start = offset * (head_size + 2);
        let end = start + head_size + 1;
        self.state[chunk].view(.., start..end, .., ..)
    }

    fn ffn(&self, layer: usize) -> Result<TensorView<f32>, TensorError> {
        let chunk = layer / self.chunk_size;
        let offset = layer % self.chunk_size;
        let head_size = self.info.num_emb / self.info.num_head;

        let start = offset * (head_size + 2) + head_size + 1;
        self.state[chunk].view(.., start..=start, .., ..)
    }
}

impl FromBuilder for ModelState {
    type Builder<'a> = StateBuilder;
    type Error = Infallible;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error> {
        let StateBuilder {
            context,
            info,
            max_batch,
            chunk_size,
        } = builder;
        let num_chunk = (info.num_layer + chunk_size - 1) / chunk_size;
        let head_size = info.num_emb / info.num_head;
        let state = (0..num_chunk)
            .map(|_| {
                let data = (0..max_batch)
                    .map(|_| vec![0.0; chunk_size * info.num_emb * (head_size + 2)])
                    .collect_vec()
                    .concat();
                context
                    .tensor_from_data(
                        Shape::new(info.num_emb, chunk_size * (head_size + 2), max_batch, 1),
                        data,
                    )
                    .expect("state creation")
            })
            .collect();
        Ok(Self {
            context: context.clone(),
            info: info.clone(),
            max_batch,
            chunk_size,
            head_size,
            state,
        })
    }
}

impl super::ModelState for ModelState {
    type BackedState = BackedState;

    #[inline]
    fn context(&self) -> &Context {
        &self.context
    }

    #[inline]
    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn load(&self, backed: &BackedState) -> Result<()> {
        use super::BackedState;
        if self.max_batch() != backed.max_batch() {
            return Err(ModelError::BatchSize(self.max_batch(), backed.max_batch()).into());
        }
        for (state, (shape, backed)) in self.state.iter().zip(backed.data.iter()) {
            let host = state.context.tensor_from_data(*shape, backed)?;
            state.load(&host)?;
        }
        Ok(())
    }

    fn load_batch(&self, backed: &BackedState, batch: usize) -> Result<()> {
        use super::BackedState;
        if self.max_batch() != backed.max_batch() {
            return Err(ModelError::BatchSize(self.max_batch(), backed.max_batch()).into());
        }
        for (state, (shape, backed)) in self.state.iter().zip(backed.data.iter()) {
            state.check_shape(*shape)?;
            let shape = state.shape();
            let shape = Shape::new(shape[0], shape[1], 1, 1);
            let host = state.context.tensor_from_data(shape, backed)?;
            state.load_batch(&host, batch)?;
        }
        Ok(())
    }

    fn back(&self) -> BackedState {
        let max_batch = self.max_batch;
        let chunk_size = self.chunk_size;
        let head_size = self.head_size;

        let data = self
            .state
            .iter()
            .map(|state| {
                let shape = state.shape();
                let map = state.context.tensor_init(shape);

                let mut encoder = state
                    .context
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                encoder.copy_tensor(state, &map).expect("back entire state");
                state.context.queue.submit(Some(encoder.finish()));

                let host = TensorCpu::from(map);
                (shape, host.to_vec())
            })
            .collect();
        BackedState {
            max_batch,
            chunk_size,
            head_size,
            data,
        }
    }

    fn back_batch(&self, batch: usize) -> Result<BackedState> {
        let max_batch = self.max_batch;
        let chunk_size = self.chunk_size;
        let head_size = self.head_size;

        if batch >= max_batch {
            return Err(ModelError::BatchOutOfRange {
                batch,
                max: max_batch,
            }
            .into());
        }

        let data: Result<Vec<_>, _> = self
            .state
            .iter()
            .map(|state| -> Result<_, TensorError> {
                let shape = state.shape();
                let shape = Shape::new(shape[0], shape[1], 1, 1);
                let map = state.context.tensor_init(shape);

                let mut encoder = state
                    .context
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                encoder.copy_tensor_batch(state, &map, batch)?;
                state.context.queue.submit(Some(encoder.finish()));

                let host = TensorCpu::from(map);
                Ok((shape, host.to_vec()))
            })
            .collect();
        let data = data?;

        Ok(BackedState {
            max_batch,
            chunk_size,
            head_size,
            data,
        })
    }

    fn blit(&self, other: &ModelState) -> Result<(), TensorError> {
        for (state, other) in self.state.iter().zip(other.state.iter()) {
            state.check_shape(other.shape())?;
            let mut encoder = state
                .context
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            encoder.copy_tensor(state, other)?;
            state.context.queue.submit(Some(encoder.finish()));
        }
        Ok(())
    }

    fn blit_batch(
        &self,
        other: &ModelState,
        from_batch: usize,
        to_batch: usize,
    ) -> Result<(), TensorError> {
        for (state, other) in self.state.iter().zip(other.state.iter()) {
            state.check_shape(other.shape())?;
            let op: TensorOp<'_> = TensorOp::blit(
                state.view(.., .., from_batch, ..)?,
                other.view(.., .., to_batch, ..)?,
            )?;
            let mut encoder = state
                .context
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.execute_tensor_op(&op);
            drop(pass);

            state.context.queue.submit(Some(encoder.finish()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BackedState {
    pub max_batch: usize,
    pub chunk_size: usize,
    pub head_size: usize,
    pub data: Vec<(Shape, Vec<f32>)>,
}

impl FromBuilder for BackedState {
    type Builder<'a> = StateBuilder;
    type Error = Infallible;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error> {
        let StateBuilder {
            info,
            max_batch,
            chunk_size,
            ..
        } = builder;
        let head_size = info.num_emb / info.num_head;
        let shape = Shape::new(info.num_emb, chunk_size * (head_size + 2), max_batch, 1);
        let data = (0..info.num_layer)
            .map(|_| {
                (0..max_batch)
                    .map(|_| vec![0.0; chunk_size * info.num_emb * (head_size + 2)])
                    .collect_vec()
                    .concat()
            })
            .map(|x| (shape, x))
            .collect();
        Ok(Self {
            max_batch,
            chunk_size,
            head_size,
            data,
        })
    }
}

impl super::BackedState for BackedState {
    #[inline]
    fn max_batch(&self) -> usize {
        self.max_batch
    }

    #[inline]
    fn num_layer(&self) -> usize {
        self.chunk_size * self.data.len()
    }

    fn embed(&self, batch: usize, layer: usize) -> Vec<f32> {
        let index = layer / self.chunk_size;
        let offset = layer % self.chunk_size;

        let chunk = &self.data[index];
        let num_emb = chunk.0[0];

        let start = ((batch * self.chunk_size + offset) * (self.head_size + 2) + 1) * num_emb;
        let end = start + num_emb;

        chunk.1[start..end].to_vec()
    }
}

impl<'a> Model<'a> {
    #[inline]
    fn request_runtime(&self, num_token: usize) -> Arc<Runtime> {
        self.runtime_cache.request(num_token, || {
            Runtime::new(&self.context, &self.info, num_token)
        })
    }

    #[inline]
    fn request_output(&self, num_batch: usize) -> Arc<Output> {
        self.output_cache.request(num_batch, || {
            Output::new(&self.context, &self.info, num_batch)
        })
    }

    #[inline]
    fn request_softmax(&self, num_batch: usize) -> Arc<Softmax> {
        self.softmax_cache.request(num_batch, || {
            Softmax::new(&self.context, &self.info, num_batch)
        })
    }

    #[inline]
    fn request_stack(&self, num_batch: usize) -> Arc<TensorGpu<u32, ReadWrite>> {
        self.stack_cache.request(num_batch, || {
            self.context.zeros(Shape::new(num_batch, 1, 1, 1))
        })
    }

    #[inline]
    fn head_shape(&self, num_batch: usize) -> Shape {
        Shape::new(self.info.num_vocab, 1, num_batch, 1)
    }

    fn run_internal(
        &self,
        tokens: Vec<Vec<u16>>,
        state: &ModelState,
        last: Option<usize>,
    ) -> Result<(Arc<Output>, Vec<Option<usize>>), TensorError> {
        let context = &self.context;
        let tensor = &self.tensor;

        let input: Vec<_> = tokens
            .into_iter()
            .map(|tokens| -> Result<_, TensorError> {
                let stack = TensorCpu::stack(
                    tokens
                        .into_iter()
                        .map(|token| tensor.embed.w.slice(.., token as usize, .., ..))
                        .try_collect()?,
                )
                .unwrap_or_else(|_| context.zeros(Shape::new(self.info.num_emb, 1, 0, 1)));
                stack.map(|x| x.to_f32()).reshape(
                    TensorDimension::Full,
                    TensorDimension::Auto,
                    TensorDimension::Dimension(1),
                    TensorDimension::Full,
                )
            })
            .try_collect()?;

        let input = TensorStack::try_from(input)?;
        let max_batch = input.max_batch();
        let num_batch = input.num_batch();
        let num_token = input.num_token();
        let head_size = self.info.num_emb / self.info.num_head;
        assert_ne!(num_token, 0);
        assert_ne!(num_batch, 0);

        // collect batch output copy commands for later
        let mut redirect = vec![None; max_batch];
        let headers = input
            .cursors
            .iter()
            .filter(|cursor| cursor.len > 0)
            .filter(|cursor| !last.is_some_and(|index| cursor.batch == index))
            .enumerate()
            .map(|(index, cursor)| {
                redirect[cursor.batch] = Some(index);
                cursor.token + cursor.len - 1
            })
            .collect_vec();
        let num_header = headers.len();

        let buffer = self.request_runtime(num_token);
        let output = self.request_output(num_header.max(1));
        let stack = self.request_stack(num_batch);

        // gather and group copy operations
        let (head_ops, head_x) = if num_token == 1 || num_token == num_header {
            (TensorOp::List(vec![]), &buffer.ffn_x)
        } else {
            let mut start = 0;
            let mut end = 1;
            let mut ops = vec![];
            while end <= headers.len() {
                if end == headers.len() || headers[end - 1] + 1 != headers[end] {
                    let first = headers[start];
                    let last = headers[end - 1];
                    assert_eq!(last - first + 1, end - start);

                    let input = buffer.ffn_x.view(.., first..=last, .., ..)?;
                    let output = output.head_x.view(.., start..end, .., ..)?;
                    ops.push(TensorOp::blit(input, output)?);

                    start = end;
                }
                end += 1;
            }
            (TensorOp::List(ops), &output.head_x)
        };

        // let head_ops: Vec<_> = input
        //     .cursors
        //     .iter()
        //     .filter(|cursor| cursor.len > 0)
        //     .filter(|cursor| !last.is_some_and(|index| cursor.batch == index))
        //     .enumerate()
        //     .map(|(index, cursor)| -> Result<TensorOp<'_>, TensorError> {
        //         redirect[cursor.batch] = Some(index);
        //         let token = cursor.token + cursor.len - 1;
        //         let input = buffer.ffn_x.as_view((.., token, ..))?;
        //         let output = buffer.head_x.as_view((.., .., index))?;
        //         TensorOp::blit(input, output)
        //     })
        //     .try_collect()?;

        let stack_host =
            context.tensor_from_data(stack.shape(), input.cursors.clone().into_stack())?;
        let cursors =
            context.tensor_from_data(buffer.cursors.shape(), input.cursors.into_cursors())?;

        stack.load(&stack_host)?;
        buffer.input.load(&input.tensor)?;
        buffer.cursors.load(&cursors)?;

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let op = TensorOp::layer_norm(
            &tensor.embed.layer_norm.w,
            &tensor.embed.layer_norm.b,
            &buffer.input,
        )?;
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        for (index, layer) in tensor.layers.iter().enumerate() {
            use TensorDimension::{Auto, Dimension};
            let time_first = layer.att.time_first.reshape(
                Dimension(head_size),
                Auto,
                Dimension(1),
                Dimension(1),
            )?;
            let time_decay = layer.att.time_decay.reshape(
                Dimension(head_size),
                Auto,
                Dimension(1),
                Dimension(1),
            )?;
            let att_x = buffer.att_x.reshape(
                Dimension(head_size),
                Auto,
                Dimension(num_token),
                Dimension(1),
            )?;
            let att_k = buffer.att_k.reshape(
                Dimension(head_size),
                Auto,
                Dimension(num_token),
                Dimension(1),
            )?;
            let att_v = buffer.att_v.reshape(
                Dimension(head_size),
                Auto,
                Dimension(num_token),
                Dimension(1),
            )?;
            let att_r = buffer.att_r.reshape(
                Dimension(head_size),
                Auto,
                Dimension(num_token),
                Dimension(1),
            )?;

            encoder.copy_tensor(&buffer.input, &buffer.att_x)?;

            let ops = TensorOp::List(vec![
                TensorOp::layer_norm(
                    &layer.att_layer_norm.w,
                    &layer.att_layer_norm.b,
                    &buffer.att_x,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.att.time_mix_k,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_kx,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.att.time_mix_v,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_vx,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.att.time_mix_r,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_rx,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.att.time_mix_g,
                    &buffer.att_x,
                    state.att(index)?,
                    &buffer.att_gx,
                )?,
                layer.att.w_k.matmul_op(
                    buffer.att_kx.view(.., .., .., ..)?,
                    buffer.att_k.view(.., .., .., ..)?,
                )?,
                layer.att.w_v.matmul_op(
                    buffer.att_vx.view(.., .., .., ..)?,
                    buffer.att_v.view(.., .., .., ..)?,
                )?,
                layer.att.w_r.matmul_op(
                    buffer.att_rx.view(.., .., .., ..)?,
                    buffer.att_r.view(.., .., .., ..)?,
                )?,
                layer.att.w_g.matmul_op(
                    buffer.att_gx.view(.., .., .., ..)?,
                    buffer.att_g.view(.., .., .., ..)?,
                )?,
                TensorOp::time_mix_v5(
                    &stack,
                    &time_decay,
                    &time_first,
                    &att_k,
                    &att_v,
                    &att_r,
                    &att_x,
                    state.att(index)?,
                )?,
                TensorOp::group_norm(&layer.att.group_norm.w, &layer.att.group_norm.b, &att_x)?,
                TensorOp::silu(&buffer.att_g, &buffer.att_x)?,
                layer.att.w_o.matmul_op(
                    buffer.att_x.view(.., .., .., ..)?,
                    buffer.att_o.view(.., .., .., ..)?,
                )?,
                TensorOp::add(&buffer.input, &buffer.att_o)?,
            ]);

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.execute_tensor_op(&ops);
            drop(pass);

            encoder.copy_tensor(&buffer.att_o, &buffer.ffn_x)?;

            let ops = TensorOp::List(vec![
                TensorOp::layer_norm(
                    &layer.ffn_layer_norm.w,
                    &layer.ffn_layer_norm.b,
                    &buffer.ffn_x,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.ffn.time_mix_k,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                    &buffer.ffn_kx,
                )?,
                TensorOp::token_shift(
                    &buffer.cursors,
                    &layer.ffn.time_mix_r,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                    &buffer.ffn_rx,
                )?,
                layer.ffn.w_k.matmul_op(
                    buffer.ffn_kx.view(.., .., .., ..)?,
                    buffer.ffn_k.view(.., .., .., ..)?,
                )?,
                TensorOp::squared_relu(&buffer.ffn_k)?,
                layer.ffn.w_v.matmul_op(
                    buffer.ffn_k.view(.., .., .., ..)?,
                    buffer.ffn_v.view(.., .., .., ..)?,
                )?,
                layer.ffn.w_r.matmul_op(
                    buffer.ffn_rx.view(.., .., .., ..)?,
                    buffer.ffn_r.view(.., .., .., ..)?,
                )?,
                TensorOp::channel_mix(
                    &buffer.cursors,
                    &buffer.ffn_r,
                    &buffer.ffn_v,
                    &buffer.ffn_x,
                    state.ffn(index)?,
                )?,
                TensorOp::add(&buffer.att_o, &buffer.ffn_x)?,
            ]);

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.execute_tensor_op(&ops);
            drop(pass);

            if index != self.info.num_layer - 1 {
                encoder.copy_tensor(&buffer.ffn_x, &buffer.input)?;
            }
        }

        if num_header > 0 {
            let mut ops = vec![TensorOp::layer_norm(
                &tensor.head.layer_norm.w,
                &tensor.head.layer_norm.b,
                head_x,
            )?];

            for (chunk, matrix) in tensor.head.w.iter().enumerate() {
                let start = chunk * self.head_chunk_size;
                let end = start + self.head_chunk_size;
                let input = head_x.view(.., .., .., ..)?;
                let output = output.head_o.view(start..end, .., .., ..)?;
                ops.push(TensorOp::matmul_vec_fp16(matrix, input, output)?);
            }

            let ops = TensorOp::List(ops);

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.execute_tensor_op(&head_ops);
            pass.execute_tensor_op(&ops);
            drop(pass);

            encoder.copy_tensor(&output.head_o, &output.map)?;
        }

        context.queue.submit(Some(encoder.finish()));
        Ok((output, redirect))
    }
}

impl<'a> FromBuilder for Model<'a> {
    type Builder<'b> = ModelBuilder<'b>;
    type Error = anyhow::Error;

    fn from_builder(builder: Self::Builder<'_>) -> Result<Self, Self::Error> {
        let ModelBuilder {
            context,
            data,
            lora,
            quant,
            head_chunk_size,
            token_chunk_size,
        } = builder;

        let loader = Loader::new(&context, data, lora)?;
        let info = Loader::info(data)?;

        let embed = Embed {
            layer_norm: LayerNorm {
                w: loader.load_vector_f16("blocks.0.ln0.weight")?,
                b: loader.load_vector_f16("blocks.0.ln0.bias")?,
            },
            w: loader.load_embed()?,
        };

        let head = Head {
            layer_norm: LayerNorm {
                w: loader.load_vector_f16("ln_out.weight")?,
                b: loader.load_vector_f16("ln_out.bias")?,
            },
            w: loader.load_head(head_chunk_size)?,
        };

        context.queue.submit(None);
        context.device.poll(wgpu::MaintainBase::Wait);

        let layers = (0..info.num_layer)
            .map(|layer| {
                let att_layer_norm = LayerNorm {
                    w: loader.load_vector_f16(format!("blocks.{layer}.ln1.weight"))?,
                    b: loader.load_vector_f16(format!("blocks.{layer}.ln1.bias"))?,
                };

                let att = format!("blocks.{layer}.att");
                let time_decay = loader.load_vector_exp_exp_f32(format!("{att}.time_decay"))?;
                let time_first = loader.load_vector_f32(format!("{att}.time_first"))?;
                let time_mix_k = loader.load_vector_f16(format!("{att}.time_mix_k"))?;
                let time_mix_v = loader.load_vector_f16(format!("{att}.time_mix_v"))?;
                let time_mix_r = loader.load_vector_f16(format!("{att}.time_mix_r"))?;
                let time_mix_g = loader.load_vector_f16(format!("{att}.time_mix_g"))?;

                let w_k = loader.load_matrix_f16(format!("{att}.key.weight"))?;
                let w_v = loader.load_matrix_f16(format!("{att}.value.weight"))?;
                let w_r = loader.load_matrix_f16(format!("{att}.receptance.weight"))?;
                let w_g = loader.load_matrix_f16(format!("{att}.gate.weight"))?;
                let w_o = loader.load_matrix_f16(format!("{att}.output.weight"))?;

                let group_norm = LayerNorm {
                    w: loader
                        .load_vector_f16(format!("{att}.ln_x.weight"))?
                        .reshape(
                            TensorDimension::Auto,
                            TensorDimension::Dimension(info.num_head),
                            TensorDimension::Dimension(1),
                            TensorDimension::Dimension(1),
                        )?,
                    b: loader
                        .load_vector_f16(format!("{att}.ln_x.bias"))?
                        .reshape(
                            TensorDimension::Auto,
                            TensorDimension::Dimension(info.num_head),
                            TensorDimension::Dimension(1),
                            TensorDimension::Dimension(1),
                        )?,
                };

                let att = match quant {
                    Quantization::Int8(x) if x.contains_layer(layer as u64) => Att {
                        time_decay,
                        time_first,
                        time_mix_k,
                        time_mix_v,
                        time_mix_r,
                        time_mix_g,
                        w_k: Matrix::quant_u8(w_k)?,
                        w_v: Matrix::quant_u8(w_v)?,
                        w_r: Matrix::quant_u8(w_r)?,
                        w_g: Matrix::quant_u8(w_g)?,
                        w_o: Matrix::quant_u8(w_o)?,
                        group_norm,
                    },
                    _ => Att {
                        time_decay,
                        time_first,
                        time_mix_k,
                        time_mix_v,
                        time_mix_r,
                        time_mix_g,
                        w_k: Matrix::Fp16(w_k),
                        w_v: Matrix::Fp16(w_v),
                        w_r: Matrix::Fp16(w_r),
                        w_g: Matrix::Fp16(w_g),
                        w_o: Matrix::Fp16(w_o),
                        group_norm,
                    },
                };

                let ffn_layer_norm = LayerNorm {
                    w: loader.load_vector_f16(format!("blocks.{layer}.ln2.weight"))?,
                    b: loader.load_vector_f16(format!("blocks.{layer}.ln2.bias"))?,
                };

                let ffn = format!("blocks.{layer}.ffn");
                let time_mix_k = loader.load_vector_f16(format!("{ffn}.time_mix_k"))?;
                let time_mix_r = loader.load_vector_f16(format!("{ffn}.time_mix_k"))?;

                let w_k = loader.load_matrix_f16(format!("{ffn}.key.weight"))?;
                let w_v = loader.load_matrix_f16(format!("{ffn}.value.weight"))?;
                let w_r = loader.load_matrix_f16(format!("{ffn}.receptance.weight"))?;

                let ffn = match quant {
                    Quantization::Int8(x) if x.contains_layer(layer as u64) => Ffn {
                        time_mix_k,
                        time_mix_r,
                        w_k: Matrix::quant_u8(w_k)?,
                        w_v: Matrix::quant_u8(w_v)?,
                        w_r: Matrix::quant_u8(w_r)?,
                    },
                    _ => Ffn {
                        time_mix_k,
                        time_mix_r,
                        w_k: Matrix::Fp16(w_k),
                        w_v: Matrix::Fp16(w_v),
                        w_r: Matrix::Fp16(w_r),
                    },
                };

                context.queue.submit(None);
                context.device.poll(wgpu::MaintainBase::Wait);

                Ok(Layer {
                    att_layer_norm,
                    ffn_layer_norm,
                    att,
                    ffn,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        context.queue.submit(None);
        context.device.poll(wgpu::MaintainBase::Wait);

        let tensor = ModelTensor {
            embed,
            head,
            layers,
        };
        Ok(Self {
            context,
            info,
            head_chunk_size,
            token_chunk_size,
            tensor,
            runtime_cache: ResourceCache::new(1),
            output_cache: ResourceCache::new(1),
            softmax_cache: ResourceCache::new(1),
            stack_cache: Default::default(),
        })
    }
}

impl super::Model for Model<'_> {
    type ModelState = ModelState;

    #[inline]
    fn context(&self) -> &Context {
        &self.context
    }

    #[inline]
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn softmax(&self, input: Vec<Option<Vec<f32>>>) -> Result<Vec<Option<Vec<f32>>>> {
        let max_batch = input.len();

        let mut redirect = vec![None; max_batch];
        let input: Vec<_> = input
            .into_iter()
            .enumerate()
            .filter_map(|(batch, data)| data.map(|data| (batch, data)))
            .map(|(batch, data)| {
                TensorCpu::from_data(&self.context, self.head_shape(1), data)
                    .map(|tensor| (batch, tensor))
            })
            .try_collect()?;
        let input = TensorCpu::stack(
            input
                .into_iter()
                .enumerate()
                .map(|(index, (batch, tensor))| {
                    redirect[batch] = Some(index);
                    tensor
                })
                .collect_vec(),
        )?;

        let num_batch = input.shape()[2];
        let softmax = self.request_softmax(num_batch);
        softmax.buffer.load(&input)?;

        let op = TensorOp::softmax(&softmax.buffer)?;

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.execute_tensor_op(&op);
        drop(pass);

        encoder.copy_tensor(&softmax.buffer, &softmax.map)?;
        self.context.queue.submit(Some(encoder.finish()));

        let mut output = TensorCpu::from(softmax.map.clone())
            .split(2)
            .expect("split buffer map")
            .into_iter()
            .map(|tensor| Some(tensor.to_vec()))
            .collect_vec();

        let mut probs = vec![None; max_batch];
        for (probs, redirect) in probs.iter_mut().zip_eq(redirect.into_iter()) {
            if let Some(redirect) = redirect {
                std::mem::swap(probs, &mut output[redirect]);
            }
        }

        Ok(probs)
    }

    fn run(
        &self,
        tokens: &mut Vec<Vec<u16>>,
        state: &Self::ModelState,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let num_token: usize = tokens.iter().map(Vec::len).sum();
        let max_batch = state.max_batch;

        if tokens.len() != max_batch {
            return Err(ModelError::BatchSize(tokens.len(), max_batch).into());
        }
        if num_token == 0 {
            return Ok(vec![None; max_batch]);
        }

        // we only infer at most `token_chunk_size` tokens at a time
        let mut num_token = num_token.min(self.token_chunk_size);
        let mut inputs = vec![vec![]; max_batch];
        let mut last = None;

        // take `num_token` tokens out of all the inputs and put into `input`
        for (index, (batch, input)) in tokens.iter_mut().zip(inputs.iter_mut()).enumerate() {
            let mid = batch.len().min(num_token);
            num_token -= mid;

            let (head, tail) = batch.split_at(mid);
            last = (!tail.is_empty()).then_some(index);
            *input = head.to_vec();
            *batch = tail.to_vec();

            if num_token == 0 {
                break;
            }
        }

        let (output, redirect) = self.run_internal(inputs, state, last)?;
        let output = TensorCpu::from(output.map.clone());

        Ok(redirect
            .into_iter()
            .map(|index| {
                index.map(|index| {
                    output
                        .slice(.., index, .., ..)
                        .expect("this never happens")
                        .to_vec()
                })
            })
            .collect())
    }
}
