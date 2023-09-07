use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex},
};

#[allow(clippy::type_complexity)]
#[derive(Debug)]
pub struct ResourceCache<K, V> {
    max_count: usize,
    map: Mutex<HashMap<K, (Arc<V>, usize)>>,
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
    pub fn new(max_count: usize) -> Self {
        Self {
            max_count,
            map: Default::default(),
        }
    }

    pub fn request(&self, key: K, f: impl FnOnce() -> V) -> Arc<V> {
        let mut map = self.map.lock().unwrap();
        let (value, _) = map.remove(&key).unwrap_or_else(|| (Arc::new(f()), 0));
        map.insert(key, (value.clone(), 0));
        map.retain(|_, (_, count)| {
            *count += 1;
            *count <= self.max_count
        });
        value
    }
}
