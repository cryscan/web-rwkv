use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[cfg(not(target_arch = "wasm32"))]
use flume::Sender;
use thiserror::Error;
use wasm_bindgen::prelude::wasm_bindgen;
use web_rwkv_derive::{Deref, DerefMut};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer,
    BufferDescriptor, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device,
    DeviceDescriptor, Features, Limits, PipelineLayoutDescriptor, PowerPreference, Queue,
    RequestAdapterOptions, ShaderModuleDescriptor,
};

use crate::{
    model::ModelInfo,
    tensor::{
        cache::ResourceCache,
        shape::{IntoBytes, Shape},
        View,
    },
};

#[derive(Deref)]
pub struct Instance(wgpu::Instance);

impl Default for Instance {
    fn default() -> Self {
        Self::new()
    }
}

impl Instance {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(Default::default());
        Self(instance)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn select_adapter(
        &self,
        backends: wgpu::Backends,
        selection: usize,
    ) -> Result<Adapter, CreateEnvironmentError> {
        self.enumerate_adapters(backends)
            .nth(selection)
            .ok_or(CreateEnvironmentError::RequestAdapterFailed)
    }

    pub async fn adapter(
        &self,
        power_preference: PowerPreference,
    ) -> Result<Adapter, CreateEnvironmentError> {
        self.request_adapter(&RequestAdapterOptions {
            power_preference,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok_or(CreateEnvironmentError::RequestAdapterFailed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextId;

#[derive(Debug)]
pub struct ContextInternal {
    pub id: uid::Id<ContextId>,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,

    pipeline_cache: Mutex<HashMap<PipelineKey, Arc<CachedPipeline>>>,
    buffer_cache: ResourceCache<(usize, BufferUsages), Buffer>,

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::type_complexity)]
    buffer_reader: Sender<(Arc<Buffer>, Sender<Box<[u8]>>)>,
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct Context(Arc<ContextInternal>);

pub struct ContextBuilder {
    adapter: Adapter,
    features: Features,
    limits: Limits,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CreateEnvironmentError {
    #[error("failed to request adaptor")]
    RequestAdapterFailed,
    #[error("failed to request device")]
    RequestDeviceFailed,
}

impl<'a> ContextBuilder {
    pub fn new(adapter: Adapter) -> Self {
        Self {
            adapter,
            features: Features::empty(),
            limits: Default::default(),
        }
    }

    pub async fn build(self) -> Result<Context, CreateEnvironmentError> {
        let Self {
            adapter,
            features,
            limits,
        } = self;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features,
                    limits,
                },
                None,
            )
            .await
            .map_err(|_| CreateEnvironmentError::RequestDeviceFailed)?;

        #[cfg(not(target_arch = "wasm32"))]
        let (sender, receiver) = flume::unbounded();

        let context = Arc::new(ContextInternal {
            id: uid::Id::new(),
            adapter,
            device,
            queue,
            pipeline_cache: Default::default(),
            buffer_cache: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            buffer_reader: sender,
        });
        let context = Context(context);

        // start a thread for reading back buffers
        #[cfg(not(target_arch = "wasm32"))]
        {
            let context = context.clone();
            std::thread::spawn(move || {
                while let Ok((buffer, sender)) = receiver.recv() {
                    let data = context.read_back_buffer(buffer);
                    let _ = sender.send(data);
                }
            });
        }

        Ok(context)
    }

    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    pub fn modify_limits(mut self, f: impl FnOnce(&mut Limits)) -> Self {
        f(&mut self.limits);
        self
    }

    /// Compute the limits automatically based on given model build info.
    pub fn with_auto_limits(mut self, info: &ModelInfo) -> Self {
        self.limits.max_buffer_size = ModelInfo::BUFFER_SIZE
            .max(info.max_non_head_buffer_size())
            .max(info.head_buffer_size()) as u64;
        self.limits.max_storage_buffer_binding_size = ModelInfo::STORAGE_BUFFER_BINDING_SIZE
            .max(info.max_non_head_buffer_size())
            .max(info.head_buffer_size())
            as u32;
        self
    }

    pub fn with_features(mut self, features: Features) -> Self {
        self.features = features;
        self
    }

    pub fn modify_features(mut self, f: impl FnOnce(&mut Features)) -> Self {
        f(&mut self.features);
        self
    }
}

/// A container of macro definitions in shader.
#[derive(Debug, Default, Clone, Deref, DerefMut, PartialEq, Eq, Hash)]
pub struct Macros(Vec<(String, String)>);

impl Macros {
    pub fn new(block_size: u32) -> Self {
        Self(vec![("BLOCK_SIZE".into(), format!("{}u", block_size))])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    name: String,
    entry_point: String,
    macros: Macros,
}

impl PipelineKey {
    fn new(name: String, entry_point: String, mut macros: Macros) -> Self {
        macros.0.sort();
        Self {
            name,
            entry_point,
            macros,
        }
    }
}

#[derive(Debug)]
pub struct CachedPipeline {
    pub pipeline: ComputePipeline,
    pub layout: BindGroupLayout,
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Context {}

impl Context {
    pub fn checkout_pipeline(
        &self,
        name: impl AsRef<str>,
        source: impl AsRef<str>,
        entry_point: impl AsRef<str>,
        layout: Option<&[BindGroupLayoutEntry]>,
        macros: Macros,
    ) -> Arc<CachedPipeline> {
        let name = name.as_ref();
        let entry_point = entry_point.as_ref();
        let key = PipelineKey::new(name.into(), entry_point.into(), macros.clone());

        use gpp::{process_str, Context};
        let mut context = Context::new();
        context.macros = macros.0.into_iter().collect();

        let mut cache = self.pipeline_cache.lock().unwrap();
        match cache.get(&key) {
            Some(pipeline) => pipeline.clone(),
            None => {
                let shader = process_str(source.as_ref(), &mut context).unwrap();
                let module = &self.device.create_shader_module(ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(Cow::from(shader)),
                });

                let layout = layout.map(|entries| {
                    let layout = self
                        .device
                        .create_bind_group_layout(&BindGroupLayoutDescriptor {
                            label: None,
                            entries,
                        });
                    self.device
                        .create_pipeline_layout(&PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&layout],
                            push_constant_ranges: &[],
                        })
                });

                let pipeline = self
                    .device
                    .create_compute_pipeline(&ComputePipelineDescriptor {
                        label: Some(name),
                        layout: layout.as_ref(),
                        module,
                        entry_point,
                    });
                let layout = pipeline.get_bind_group_layout(0);
                let pipeline = Arc::new(CachedPipeline { pipeline, layout });

                cache.insert(key, pipeline.clone());
                pipeline
            }
        }
    }

    pub fn checkout_shape_uniform(&self, shape: Shape) -> Arc<Buffer> {
        let view = View {
            shape,
            stride: shape,
            offset: Shape::new(0, 0, 0, 0),
        };
        self.buffer_cache.checkout(
            (view.into_bytes().len(), BufferUsages::UNIFORM),
            |buffer| self.queue.write_buffer(buffer, 0, &view.into_bytes()),
            || {
                self.device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: &view.into_bytes(),
                    usage: BufferUsages::UNIFORM,
                })
            },
        )
    }

    pub fn checkout_view_uniform(&self, view: View) -> Arc<Buffer> {
        self.buffer_cache.checkout(
            (view.into_bytes().len(), BufferUsages::UNIFORM),
            |buffer| self.queue.write_buffer(buffer, 0, &view.into_bytes()),
            || {
                self.device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: &view.into_bytes(),
                    usage: BufferUsages::UNIFORM,
                })
            },
        )
    }

    pub fn checkout_buffer_init(&self, contents: &[u8], usage: BufferUsages) -> Arc<Buffer> {
        self.buffer_cache.checkout(
            (contents.len(), usage),
            |buffer| self.queue.write_buffer(buffer, 0, contents),
            || {
                self.device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents,
                    usage,
                })
            },
        )
    }

    pub fn checkout_buffer(&self, size: usize, usage: BufferUsages) -> Arc<Buffer> {
        self.buffer_cache.checkout(
            (size, usage),
            |_| (),
            || {
                self.device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: size as u64,
                    usage,
                    mapped_at_creation: false,
                })
            },
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::type_complexity)]
    pub fn buffer_reader(&self) -> Sender<(Arc<Buffer>, Sender<Box<[u8]>>)> {
        self.buffer_reader.clone()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_back_buffer(&self, buffer: Arc<Buffer>) -> Box<[u8]> {
        assert!(buffer.usage().contains(BufferUsages::MAP_READ));

        let (sender, receiver) = flume::unbounded();
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::MaintainBase::Wait);
        receiver.recv().unwrap().unwrap();

        let data = {
            let map = slice.get_mapped_range();
            map.to_vec().into_boxed_slice()
        };
        buffer.unmap();
        data
    }
}
