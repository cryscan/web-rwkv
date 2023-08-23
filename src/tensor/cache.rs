use std::{
    cell::RefCell,
    collections::HashMap,
    hash::Hash,
    sync::{Arc, RwLock},
};

#[allow(clippy::type_complexity)]
#[derive(Debug)]
pub struct ResourceCache<K, V> {
    max_count: usize,
    map: RefCell<RwLock<HashMap<K, (Arc<V>, usize)>>>,
}

impl<K, V> Default for ResourceCache<K, V> {
    fn default() -> Self {
        Self {
            max_count: 16,
            map: Default::default(),
        }
    }
}

impl<K, V> ResourceCache<K, V>
where
    K: PartialEq + Eq + Hash,
{
    pub fn query(&self, key: &K) -> Option<Arc<V>> {
        let map = self.map.borrow();
        let map = map.read().unwrap();
        map.get(key).cloned().map(|(v, _)| v)
    }

    pub fn request(&self, key: K, op: impl FnOnce() -> V) -> Arc<V> {
        let buffer = self.query(&key).unwrap_or_else(|| Arc::new(op()));

        let map = self.map.borrow_mut();
        let mut map = map.write().unwrap();
        map.insert(key, (buffer.clone(), 0));

        map.retain(|_, (_, count)| {
            *count += 1;
            *count <= self.max_count
        });

        buffer
    }
}
