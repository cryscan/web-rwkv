use std::{
    cell::RefCell,
    collections::HashMap,
    hash::Hash,
    sync::{Arc, RwLock},
};

use web_rwkv_derive::Deref;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferUsages,
};

use crate::context::Context;

pub trait IntoBytes {
    fn into_bytes(self) -> Vec<u8>;
}

#[derive(Debug, Default, Deref)]
pub struct UniformCache<K>(RefCell<RwLock<HashMap<K, Arc<Buffer>>>>);

impl<K> UniformCache<K>
where
    K: Clone + PartialEq + Eq + Hash + IntoBytes,
{
    pub fn clear(&self) {
        let map = self.borrow_mut();
        let mut map = map.write().unwrap();
        map.clear();
    }

    pub fn query(&self, key: &K) -> Option<Arc<Buffer>> {
        let map = self.borrow();
        let map = map.read().unwrap();
        map.get(key).cloned()
    }

    pub fn request(&self, context: &Context, key: K) -> Arc<Buffer> {
        match self.query(&key) {
            Some(buffer) => buffer,
            None => {
                let contents = &key.clone().into_bytes();
                let buffer = Arc::new(context.device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents,
                    usage: BufferUsages::UNIFORM,
                }));
                let map = self.borrow_mut();
                let mut map = map.write().unwrap();
                map.insert(key, buffer.clone());
                buffer
            }
        }
    }
}
