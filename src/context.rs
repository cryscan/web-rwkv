use std::{borrow::Cow, collections::BTreeMap, sync::Arc};

use futures::Future;
use thiserror::Error;
use wasm_bindgen::prelude::wasm_bindgen;
use web_rwkv_derive::{Deref, DerefMut};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer,
    BufferDescriptor, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device,
    DeviceDescriptor, Features, Instance, Limits, MemoryHints, PipelineLayoutDescriptor,
    PowerPreference, Queue, RequestAdapterOptions, ShaderModuleDescriptor,
};

use crate::tensor::{
    cache::ResourceCache,
    shape::{IntoBytes, Shape},
    View,
};

pub trait InstanceExt {
    fn adapter(
        &self,
        power_preference: PowerPreference,
    ) -> impl Future<Output = Result<Adapter, CreateEnvironmentError>>;
}

impl InstanceExt for Instance {
    async fn adapter(
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
pub struct ContextEvent {
    pub buffer: Arc<Buffer>,
    pub sender: tokio::sync::oneshot::Sender<Box<[u8]>>,
}

#[derive(Debug)]
pub struct ContextInternal {
    pub id: uid::Id<ContextId>,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,

    pipeline_cache: ResourceCache<PipelineKey, CachedPipeline>,
    shape_cache: ResourceCache<View, Buffer>,
    buffer_cache: ResourceCache<BufferKey, Buffer>,

    #[cfg(not(target_arch = "wasm32"))]
    event: flume::Sender<ContextEvent>,
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct Context(Arc<ContextInternal>);

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Context {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) <= 1 {
            self.clear_buffers();
            self.queue.submit(None);
            self.device.poll(wgpu::Maintain::Wait);
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
        let features = Features::empty();
        #[cfg(feature = "subgroup-ops")]
        let features = features | Features::SUBGROUP;
        Self {
            adapter,
            features,
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
                    required_features: features,
                    required_limits: limits,
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|_| CreateEnvironmentError::RequestDeviceFailed)?;

        #[cfg(not(target_arch = "wasm32"))]
        let (event, receiver) = flume::unbounded();

        let context = Arc::new(ContextInternal {
            id: uid::Id::new(),
            adapter,
            device,
            queue,
            pipeline_cache: Default::default(),
            shape_cache: Default::default(),
            buffer_cache: ResourceCache::new(2),
            #[cfg(not(target_arch = "wasm32"))]
            event,
        });
        let context = Context(context);

        // start a thread for reading back buffers
        #[cfg(not(target_arch = "wasm32"))]
        {
            let id = context.id;
            let context = Arc::downgrade(&context);
            std::thread::spawn(move || {
                while let Ok(ContextEvent { buffer, sender }) = receiver.recv() {
                    match context.upgrade() {
                        Some(context) => {
                            #[cfg(feature = "trace")]
                            let _span = tracing::trace_span!("device").entered();
                            let data = context.read_back_buffer(buffer);
                            let _ = sender.send(data);
                        }
                        None => break,
                    }
                }
                log::info!("context {} destroyed", id);
            });
        }

        Ok(context)
    }

    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    pub fn update_limits(mut self, f: impl FnOnce(&mut Limits)) -> Self {
        f(&mut self.limits);
        self
    }

    pub fn features(mut self, features: Features) -> Self {
        self.features = features;
        self
    }

    pub fn update_features(mut self, f: impl FnOnce(&mut Features)) -> Self {
        f(&mut self.features);
        self
    }
}

/// A container of macro definitions in shader.
#[derive(Debug, Default, Clone, Deref, DerefMut, PartialEq, Eq, Hash)]
pub struct Macros(BTreeMap<String, String>);

impl Macros {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn compile(self) -> Vec<(String, String)> {
        self.0.into_iter().collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    name: String,
    entry_point: String,
    macros: Vec<(String, String)>,
}

impl PipelineKey {
    fn new(name: String, entry_point: String, macros: Macros) -> Self {
        let macros = macros.compile();
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BufferKey {
    size: usize,
    usage: BufferUsages,
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

        self.pipeline_cache.checkout(
            key,
            || {
                use gpp::{process_str, Context};
                let mut context = Context::new();
                context.macros = macros.0.into_iter().collect();

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
                        compilation_options: Default::default(),
                        cache: None,
                    });
                let layout = pipeline.get_bind_group_layout(0);
                CachedPipeline { pipeline, layout }
            },
            |_| {},
        )
    }

    pub(crate) fn checkout_shape_uniform(&self, shape: Shape) -> Arc<Buffer> {
        let view = View {
            shape,
            stride: shape,
            offset: Shape::new(0, 0, 0, 0),
        };
        let desc = BufferInitDescriptor {
            label: None,
            contents: &view.into_bytes(),
            usage: BufferUsages::UNIFORM,
        };
        self.shape_cache
            .checkout(view, || self.device.create_buffer_init(&desc), |_| {})
    }

    pub(crate) fn checkout_view_uniform(&self, view: View) -> Arc<Buffer> {
        let desc = BufferInitDescriptor {
            label: None,
            contents: &view.into_bytes(),
            usage: BufferUsages::UNIFORM,
        };
        self.shape_cache
            .checkout(view, || self.device.create_buffer_init(&desc), |_| {})
    }

    pub(crate) fn checkout_buffer_init(&self, contents: &[u8], usage: BufferUsages) -> Arc<Buffer> {
        let size = std::mem::size_of_val(contents);
        let _key = BufferKey { size, usage };
        let desc = BufferInitDescriptor {
            label: None,
            contents,
            usage,
        };
        // self.buffer_cache.checkout(
        //     key,
        //     || self.device.create_buffer_init(&desc),
        //     |buffer| self.queue.write_buffer(buffer, 0, contents),
        // )
        self.device.create_buffer_init(&desc).into()
    }

    pub(crate) fn checkout_buffer(&self, size: usize, usage: BufferUsages) -> Arc<Buffer> {
        let key = BufferKey { size, usage };
        let desc = BufferDescriptor {
            label: None,
            size: size as u64,
            usage,
            mapped_at_creation: false,
        };
        self.buffer_cache
            .checkout(key, || self.device.create_buffer(&desc), |_| {})
    }

    // pub(crate) fn checkout_buffer_uncached(&self, size: usize, usage: BufferUsages) -> Arc<Buffer> {
    //     self.device
    //         .create_buffer(&BufferDescriptor {
    //             label: None,
    //             size: size as u64,
    //             usage,
    //             mapped_at_creation: false,
    //         })
    //         .into()
    // }

    /// Maintain resource caches.
    #[inline]
    pub fn maintain(&self) {
        self.pipeline_cache.maintain();
        self.shape_cache.maintain();
        self.buffer_cache.maintain();
    }

    /// Clear resource caches.
    #[inline]
    pub fn clear_buffers(&self) {
        self.shape_cache.clear();
        self.buffer_cache.clear();
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn event(&self) -> flume::Sender<ContextEvent> {
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
            let len = map.len();
            let size = std::mem::size_of::<u32>();
            let data = vec![0u32; (len + size - 1) / size].into_boxed_slice();
            unsafe {
                let data = Box::leak(data);
                let data: &mut [u8] = bytemuck::cast_slice_mut(data);
                data.copy_from_slice(&map);
                Box::from_raw(data)
            }
        };
        buffer.unmap();
        data
    }

    #[cfg(feature = "subgroup-ops")]
    pub fn min_subgroup_size(&self) -> u32 {
        self.adapter.limits().min_subgroup_size
    }

    #[cfg(feature = "subgroup-ops")]
    pub fn max_subgroup_size(&self) -> u32 {
        self.adapter.limits().max_subgroup_size
    }
}
