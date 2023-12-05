use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex},
};

#[derive(Debug)]
struct CacheItem<V> {
    value: Arc<V>,
    count: usize,
}

impl<V> CacheItem<V> {
    fn make(f: impl FnOnce() -> V) -> Self {
        Self {
            value: f().into(),
            count: 0,
        }
    }
}

#[allow(clippy::type_complexity)]
#[derive(Debug, Clone)]
pub struct ResourceCache<K, V> {
    max_count: usize,
    map: Arc<Mutex<HashMap<K, CacheItem<V>>>>,
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
        if self.max_count > 0 {
            map.retain(|_, item| {
                item.count += 1;
                item.count <= self.max_count
            });
        }
        let CacheItem { value, .. } = map.remove(&key).unwrap_or_else(|| CacheItem::make(f));
        map.insert(
            key,
            CacheItem {
                value: value.clone(),
                count: 0,
            },
        );
        value
    }

    pub fn clear(&self) {
        let mut map = self.map.lock().unwrap();
        map.clear();
    }
}
