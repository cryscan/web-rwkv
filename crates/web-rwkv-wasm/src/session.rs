use std::cell::RefCell;

use anyhow::Result;
use half::f16;
use serde::{de::DeserializeSeed, Deserialize};
use wasm_bindgen::prelude::*;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    num::Float,
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::{Loader, Reader},
        model::{Bundle, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant, State},
        softmax::softmax_one,
        v4, v5, v6, v7, Runtime, SimpleRuntime,
    },
    tensor::{ops::TensorOp, serialization::Seed, TensorCpu},
    wgpu::{Instance, InstanceDescriptor, PowerPreference},
};

use crate::{cache::Cache, loader::TensorReader, ops::TensorOpExt};

pub const TOKEN_CHUNK_SIZE: usize = 128;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum SessionType {
    Puzzle,
    Chat,
    Music,
    Othello,
}

/// We need to slightly modify the model structure using hooks.
fn make_puzzle_hooks_v6<F: Float>(info: &ModelInfo) -> Result<v6::HookMap<F>> {
    let mut hooks = v6::HookMap::new();
    for layer in 0..info.num_layer {
        // add a custom operation before time-mix for each layer
        hooks.insert(
            v6::Hook::PreAttTimeDecayActivate(layer),
            Box::new(move |frame: v6::Frame<F>| {
                let op = TensorOp::mul_exp(&frame.buffer.time_decay, &frame.buffer.att_k)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
    }
    Ok(hooks)
}

fn make_puzzle_hooks_v7<F: Float>(info: &ModelInfo) -> Result<v7::HookMap<F>> {
    let mut hooks = v7::HookMap::new();

    for layer in 0..info.num_layer {
        hooks.insert(
            v7::Hook::PostAttAdapt(layer),
            Box::new(move |frame: v7::Frame<F>| {
                let op = TensorOp::affine(&frame.buffer.att_a, 2.0, 0.0)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
        hooks.insert(
            v7::Hook::PostAttControl(layer),
            Box::new(move |frame: v7::Frame<F>| {
                let op = TensorOp::mul_w(&frame.buffer.att_w, &frame.buffer.att_a)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
    }

    Ok(hooks)
}

#[derive(Debug, Deserialize)]
struct Prefab {
    info: ModelInfo,
}

pub struct Session {
    context: Context,
    info: ModelInfo,
    runtime: Box<dyn Runtime<Rnn>>,
    state: Box<dyn State>,
    cache: RefCell<Cache>,
    ty: SessionType,
}

impl Session {
    pub async fn new<R: Reader>(
        model: R,
        quant: usize,
        quant_nf4: usize,
        quant_sf4: usize,
        ty: SessionType,
    ) -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .adapter(PowerPreference::HighPerformance)
            .await
            .expect("failed to request adapter");
        let info = Loader::info(&model)?;

        let context = ContextBuilder::new(adapter)
            .auto_limits(&info)
            .build()
            .await?;

        let quant = (0..quant)
            .map(|layer| (layer, Quant::Int8))
            .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
            .chain((0..quant_sf4).map(|layer| (layer, Quant::SF4)))
            .collect();
        let builder = ModelBuilder::new(&context, model).quant(quant);
        let rescale_layer = match ty {
            SessionType::Chat => 6,
            SessionType::Puzzle | SessionType::Othello | SessionType::Music => 999, // = no rescale
        };
        let (runtime, state): (Box<dyn Runtime<_>>, Box<dyn State>) = match ty {
            SessionType::Puzzle => {
                let hooks = make_puzzle_hooks_v6(&info)?;
                let model = builder.rescale(rescale_layer).build_v6().await?;
                let bundle = v6::Bundle::<f16>::new_with_hooks(model, 1, hooks);
                let state = bundle.state();
                let runtime = SimpleRuntime::new(bundle);
                (Box::new(runtime), Box::new(state))
            }
            SessionType::Othello => {
                let hooks = make_puzzle_hooks_v7(&info)?;
                let model = builder.rescale(rescale_layer).build_v7().await?;
                let bundle = v7::Bundle::<f16>::new_with_hooks(model, 1, hooks);
                let state = bundle.state();
                let runtime = SimpleRuntime::new(bundle);
                (Box::new(runtime), Box::new(state))
            }
            SessionType::Chat | SessionType::Music => match info.version {
                ModelVersion::V4 => {
                    let model = builder.rescale(rescale_layer).build_v4().await?;
                    let bundle = v4::Bundle::<f16>::new(model, 1);
                    let state = bundle.state();
                    let runtime = SimpleRuntime::new(bundle);
                    (Box::new(runtime), Box::new(state))
                }
                ModelVersion::V5 => {
                    let model = builder.rescale(rescale_layer).build_v5().await?;
                    let bundle = v5::Bundle::<f16>::new(model, 1);
                    let state = bundle.state();
                    let runtime = SimpleRuntime::new(bundle);
                    (Box::new(runtime), Box::new(state))
                }
                ModelVersion::V6 => {
                    let model = builder.rescale(rescale_layer).build_v6().await?;
                    let bundle = v6::Bundle::<f16>::new(model, 1);
                    let state = bundle.state();
                    let runtime = SimpleRuntime::new(bundle);
                    (Box::new(runtime), Box::new(state))
                }
                ModelVersion::V7 => {
                    // no rescale needed for v7 models
                    let model = builder.build_v7().await?;
                    let bundle = v7::Bundle::<f16>::new(model, 1);
                    let state = bundle.state();
                    let runtime = SimpleRuntime::new(bundle);
                    (Box::new(runtime), Box::new(state))
                }
            },
        };

        let cache = RefCell::new(Default::default());

        Ok(Self {
            context,
            info,
            runtime,
            state,
            cache,
            ty,
        })
    }

    pub async fn from_prefab(data: &[u8], ty: SessionType) -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .adapter(PowerPreference::HighPerformance)
            .await
            .expect("failed to request adapter");

        let Prefab { info } = cbor4ii::serde::from_slice::<Prefab>(data)?;

        let context = ContextBuilder::new(adapter)
            .auto_limits(&info)
            .build()
            .await?;

        let reader = cbor4ii::core::utils::SliceReader::new(data);
        let mut deserializer = cbor4ii::serde::Deserializer::new(reader);

        let (runtime, state): (Box<dyn Runtime<_>>, Box<dyn State>) = match ty {
            SessionType::Puzzle => {
                let seed: Seed<_, v6::Model> = Seed::new(&context);
                let hooks = make_puzzle_hooks_v6(&info)?;
                let model = seed.deserialize(&mut deserializer)?;
                let bundle = v6::Bundle::<f16>::new_with_hooks(model, 1, hooks);
                let state = bundle.state();
                let runtime = SimpleRuntime::new(bundle);
                (Box::new(runtime), Box::new(state))
            }
            SessionType::Othello => {
                let seed: Seed<_, v7::Model> = Seed::new(&context);
                let hooks = make_puzzle_hooks_v7(&info)?;
                let model = seed.deserialize(&mut deserializer)?;
                let bundle = v7::Bundle::<f16>::new_with_hooks(model, 1, hooks);
                let state = bundle.state();
                let runtime = SimpleRuntime::new(bundle);
                (Box::new(runtime), Box::new(state))
            }
            SessionType::Chat | SessionType::Music => match info.version {
                ModelVersion::V4 => {
                    let seed: Seed<_, v4::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v4::Bundle::<f16>::new(model, 1);
                    let state = bundle.state();
                    let runtime = SimpleRuntime::new(bundle);
                    (Box::new(runtime), Box::new(state))
                }
                ModelVersion::V5 => {
                    let seed: Seed<_, v5::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v5::Bundle::<f16>::new(model, 1);
                    let state = bundle.state();
                    let runtime = SimpleRuntime::new(bundle);
                    (Box::new(runtime), Box::new(state))
                }
                ModelVersion::V6 => {
                    let seed: Seed<_, v6::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v6::Bundle::<f16>::new(model, 1);
                    let state = bundle.state();
                    let runtime = SimpleRuntime::new(bundle);
                    (Box::new(runtime), Box::new(state))
                }
                ModelVersion::V7 => {
                    let seed: Seed<_, v7::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v7::Bundle::<f16>::new(model, 1);
                    let state = bundle.state();
                    let runtime = SimpleRuntime::new(bundle);
                    (Box::new(runtime), Box::new(state))
                }
            },
        };

        let cache = RefCell::new(Default::default());

        Ok(Self {
            context,
            info,
            runtime,
            state,
            cache,
            ty,
        })
    }

    pub async fn back(&self) -> Result<TensorCpu<f32>> {
        Ok(self.state.back(0).await?)
    }

    pub fn load(&self, backed: TensorCpu<f32>) -> Result<()> {
        Ok(self.state.load(backed, 0)?)
    }

    pub async fn run(&self, tokens: &[u32]) -> Result<TensorCpu<f32>> {
        let mut inference = Some(RnnInput::new(
            vec![RnnInputBatch::new(tokens.to_vec(), RnnOption::Last)],
            TOKEN_CHUNK_SIZE,
        ));

        let output = loop {
            let input = inference.take().unwrap();
            let (input, output) = self.runtime.infer(input).await?;
            inference = Some(input);

            let output = output[0].0.clone();
            if !output.is_empty() {
                break output;
            }
        };

        Ok(output)
    }

    pub async fn softmax(&self, logits: TensorCpu<f32>) -> Result<TensorCpu<f32>> {
        Ok(softmax_one(&self.context, logits).await?)
    }
}

fn err(err: impl ToString) -> JsError {
    JsError::new(&err.to_string())
}

#[wasm_bindgen(js_name = Session)]
pub struct SessionExport(Session);

#[wasm_bindgen(js_class = Session)]
impl SessionExport {
    /// Build a session from a model reader (e.g. a [`TensorReader`] over f16
    /// `safetensors` tensors).
    ///
    /// This is a static async factory rather than a `constructor`: `wasm-bindgen`
    /// async constructors generate invalid TypeScript (the `.d.ts` types the result
    /// as `Session` while the runtime returns `Promise<Session>`), so callers use
    /// `Session.from_reader(...)` and `await` the result.
    pub async fn from_reader(
        model: TensorReader,
        quant: usize,
        quant_nf4: usize,
        quant_sf4: usize,
        ty: SessionType,
    ) -> Result<Self, JsError> {
        let session = Session::new(model, quant, quant_nf4, quant_sf4, ty)
            .await
            .map_err(err)?;
        Ok(Self(session))
    }

    pub async fn from_prefab(data: &[u8], ty: SessionType) -> Result<Self, JsError> {
        let session = Session::from_prefab(data, ty).await.map_err(err)?;
        Ok(Self(session))
    }

    pub async fn run(&self, tokens: &[u32], output: &mut [f32]) -> Result<(), JsError> {
        let data = self.0.run(tokens).await.map_err(err)?;
        output.copy_from_slice(&data.data()[..output.len()]);
        Ok(())
    }

    pub async fn softmax(&self, input: &[f32], output: &mut [f32]) -> Result<(), JsError> {
        assert_eq!(input.len(), output.len());
        let input = self
            .0
            .context
            .tensor_from_data([input.len(), 1, 1, 1], input.to_vec())
            .map_err(err)?;
        let data = self.0.softmax(input).await.map_err(err)?;
        output.copy_from_slice(&data.data()[..output.len()]);
        Ok(())
    }

    pub fn info(&self) -> ModelInfo {
        self.0.info.clone()
    }

    pub fn session_type(&self) -> SessionType {
        self.0.ty
    }

    pub fn state_len(&self) -> usize {
        self.0.state.init_shape().len()
    }

    pub async fn back(&self, state: &mut [f32]) -> Result<(), JsError> {
        let data = self.0.back().await.map_err(err)?;
        assert_eq!(data.len(), state.len());
        state.copy_from_slice(&data);
        Ok(())
    }

    pub fn load(&self, state: &[f32]) -> Result<(), JsError> {
        let shape = self.0.state.init_shape();
        let state = self.0.context.tensor_from_data(shape, state.to_vec())?;
        self.0.load(state).map_err(err)
    }

    pub fn checkout(
        &self,
        tokens: &[u32],
        state: &mut [f32],
        output: &mut [f32],
    ) -> Result<usize, JsError> {
        let cache = self.0.cache.borrow();
        let checkout = cache.checkout(tokens);

        let cutoff = match checkout.item {
            Some(item) => {
                assert_eq!(item.state.len(), state.len());
                state.copy_from_slice(&item.state);

                assert_eq!(item.output.len(), output.len());
                output.copy_from_slice(&item.output);

                checkout.prefix.len()
            }
            None => {
                let data = self.0.state.init().to_vec();
                assert_eq!(data.len(), state.len());
                state.copy_from_slice(&data);

                let data = vec![0.0; output.len()];
                output.copy_from_slice(&data);

                0
            }
        };

        Ok(cutoff)
    }

    pub fn cache(&self, tokens: &[u32], state: &[f32], output: &[f32]) -> Result<(), JsError> {
        let mut cache = self.0.cache.borrow_mut();

        let shape = self.0.state.init_shape();
        let state = self
            .0
            .context
            .tensor_from_data(shape, state.to_vec())
            .map_err(err)?;

        let output = self
            .0
            .context
            .tensor_from_data([output.len(), 1, 1, 1], output.to_vec())
            .map_err(err)?;

        cache.insert(tokens, state, output);

        Ok(())
    }

    pub fn clear_cache(&self) {
        let mut cache = self.0.cache.borrow_mut();
        cache.clear();
    }
}
