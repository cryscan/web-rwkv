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

/// An LRU cache.
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
    /// Note: If `max_count` is 0, the cache won't evict any items.
    pub fn new(max_count: usize) -> Self {
        Self {
            max_count,
            map: Default::default(),
        }
    }

    /// Checkout the item with the given key. If the item doesn't exist, `f` is called to construct it.
    pub fn checkout(&self, key: K, f: impl FnOnce() -> V) -> Arc<V> {
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

    /// Empty the cache.
    pub fn clear(&self) {
        let mut map = self.map.lock().unwrap();
        map.clear();
    }
}
