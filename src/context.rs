use std::{borrow::Cow, sync::Arc};

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

use crate::tensor::{
    cache::ResourceCache,
    shape::{IntoBytes, Shape},
    View,
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

#[cfg(not(target_arch = "wasm32"))]
pub enum ContextEvent {
    ReadBack {
        buffer: Arc<Buffer>,
        sender: tokio::sync::oneshot::Sender<Box<[u8]>>,
    },
    Drop,
}

#[derive(Debug)]
pub struct ContextInternal {
    pub id: uid::Id<ContextId>,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,

    pipeline_cache: ResourceCache<PipelineKey, CachedPipeline>,
    view_cache: ResourceCache<View, Buffer>,

    #[cfg(not(target_arch = "wasm32"))]
    event: tokio::sync::mpsc::UnboundedSender<ContextEvent>,
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct Context(Arc<ContextInternal>);

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Context {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) <= 1 {
            let _ = self.event.send(ContextEvent::Drop);
        }
    }
}

pub struct ContextBuilder {
    pub adapter: Adapter,
    pub features: Features,
    pub limits: Limits,
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
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();

        let context = Arc::new(ContextInternal {
            id: uid::Id::new(),
            adapter,
            device,
            queue,
            pipeline_cache: Default::default(),
            view_cache: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            event: sender,
        });
        let context = Context(context);

        // start a thread for reading back buffers
        #[cfg(not(target_arch = "wasm32"))]
        {
            let id = context.id;
            let context = Arc::downgrade(&context);
            std::thread::spawn(move || {
                while let Some(event) = receiver.blocking_recv() {
                    match (event, context.upgrade()) {
                        (ContextEvent::ReadBack { buffer, sender }, Some(context)) => {
                            let data = context.read_back_buffer(buffer);
                            let _ = sender.send(data);
                        }
                        _ => break,
                    }
                }
                log::info!("context {} destroyed", id);
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
    // pub fn new(block_size: u32) -> Self {
    //     Self(vec![("BLOCK_SIZE".into(), format!("{}u", block_size))])
    // }
    pub fn new() -> Self {
        Self(vec![])
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

impl ContextInternal {
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

        self.pipeline_cache.checkout(key, || {
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
            CachedPipeline { pipeline, layout }
        })
    }

    pub(crate) fn checkout_shape_uniform(&self, shape: Shape) -> Arc<Buffer> {
        let view = View {
            shape,
            stride: shape,
            offset: Shape::new(0, 0, 0, 0),
        };
        self.view_cache.checkout(view, || {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: &view.into_bytes(),
                usage: BufferUsages::UNIFORM,
            })
        })
    }

    pub(crate) fn checkout_view_uniform(&self, view: View) -> Arc<Buffer> {
        self.view_cache.checkout(view, || {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: &view.into_bytes(),
                usage: BufferUsages::UNIFORM,
            })
        })
    }

    pub(crate) fn checkout_buffer_init(&self, contents: &[u8], usage: BufferUsages) -> Arc<Buffer> {
        self.device
            .create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents,
                usage,
            })
            .into()
    }

    pub(crate) fn checkout_buffer(&self, size: usize, usage: BufferUsages) -> Arc<Buffer> {
        self.device
            .create_buffer(&BufferDescriptor {
                label: None,
                size: size as u64,
                usage,
                mapped_at_creation: false,
            })
            .into()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn event(&self) -> tokio::sync::mpsc::UnboundedSender<ContextEvent> {
        self.event.clone()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_back_buffer(&self, buffer: Arc<Buffer>) -> Box<[u8]> {
        assert!(buffer.usage().contains(BufferUsages::MAP_READ));

        let (sender, receiver) = tokio::sync::oneshot::channel();
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::MaintainBase::Wait);
        receiver.blocking_recv().unwrap().unwrap();

        let data = {
            let map = slice.get_mapped_range();
            map.to_vec().into_boxed_slice()
        };
        buffer.unmap();
        data
    }
}
