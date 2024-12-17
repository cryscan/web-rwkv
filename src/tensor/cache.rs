use std::{
    hash::Hash,
    sync::{Arc, RwLock},
};

use rustc_hash::FxHashMap as HashMap;

#[derive(Debug, Clone)]
struct CachedItem<V> {
    value: Arc<V>,
    life: usize,
}

impl<V> CachedItem<V> {
    fn ref_count(&self) -> usize {
        Arc::strong_count(&self.value)
    }
}

#[derive(Debug, Clone)]
pub struct ResourceCache<K, V> {
    map: Arc<RwLock<HashMap<K, Vec<CachedItem<V>>>>>,
    #[allow(unused)]
    limit: usize,
}

impl<K, V> Default for ResourceCache<K, V> {
    fn default() -> Self {
        Self {
            map: Default::default(),
            limit: 0,
        }
    }
}

impl<K, V> ResourceCache<K, V>
where
    K: Clone + PartialEq + Eq + Hash,
{
    pub fn new(limit: usize) -> Self {
        Self {
            map: Default::default(),
            limit,
        }
    }

    /// Update cache internal counters and evict old values.
    pub fn maintain(&self) {
        if self.limit == 0 {
            return;
        }

        let mut map = self.map.write().unwrap();
        for items in map.values_mut() {
            items.retain(|item| match item.ref_count() {
                0 | 1 => item.life < self.limit,
                _ => true,
            });
            items.iter_mut().for_each(|item| match item.ref_count() {
                0 | 1 => item.life = 0,
                _ => item.life += 1,
            });
        }
    }

    /// Release all values.
    pub fn clear(&self) {
        let mut map = self.map.write().unwrap();
        map.clear();
    }

    /// Checkout the item with the given key. If the item doesn't exist, `miss` is called to construct it.
    /// If hit, `hit` is called on the cached value.
    pub fn checkout_hit(&self, key: K, miss: impl FnOnce() -> V, hit: impl FnOnce(&V)) -> Arc<V> {
        let map = self.map.read().unwrap();
        let value = match map
            .get(&key)
            .and_then(|items| items.iter().find(|&item| item.ref_count() <= 1))
        {
            Some(CachedItem { value, .. }) => {
                let value = value.clone();
                drop(map);

                #[cfg(feature = "trace")]
                let _span = tracing::trace_span!("cache hit").entered();
                hit(&value);
                value
            }
            None => {
                drop(map);

                #[cfg(feature = "trace")]
                let _span = tracing::trace_span!("cache miss").entered();

                let value = Arc::new(miss());
                let item = CachedItem {
                    value: value.clone(),
                    life: 0,
                };

                let mut map = self.map.write().unwrap();
                match map.get_mut(&key) {
                    Some(items) => items.push(item),
                    None => map.extend(Some((key, vec![item]))),
                }
                value
            }
        };

        value
    }

    /// Checkout the item with the given key. If the item doesn't exist, `miss` is called to construct it.
    pub fn checkout(&self, key: K, miss: impl FnOnce() -> V) -> Arc<V> {
        self.checkout_hit(key, miss, |_| ())
    }
}

#[derive(Debug, Clone)]
pub struct SharedResourceCache<K, V> {
    map: Arc<RwLock<HashMap<K, CachedItem<V>>>>,
    #[allow(unused)]
    limit: usize,
}

impl<K, V> Default for SharedResourceCache<K, V> {
    fn default() -> Self {
        Self {
            map: Default::default(),
            limit: 0,
        }
    }
}

impl<K, V> SharedResourceCache<K, V>
where
    K: Clone + PartialEq + Eq + Hash,
{
    pub fn new(limit: usize) -> Self {
        Self {
            map: Default::default(),
            limit,
        }
    }

    /// Update cache internal counters and evict old values.
    pub fn maintain(&self) {
        if self.limit == 0 {
            return;
        }

        let mut map = self.map.write().unwrap();
        map.retain(|_, item| match item.ref_count() {
            0 | 1 => item.life < self.limit,
            _ => true,
        });
        for item in map.values_mut() {
            match item.ref_count() {
                0 | 1 => item.life = 0,
                _ => item.life += 1,
            }
        }
    }

    /// Release all values.
    pub fn clear(&self) {
        let mut map = self.map.write().unwrap();
        map.clear();
    }

    /// Checkout the item with the given key. If the item doesn't exist, `miss` is called to construct it.
    /// If hit, `hit` is called on the cached value.
    pub fn checkout_hit(&self, key: K, miss: impl FnOnce() -> V, hit: impl FnOnce(&V)) -> Arc<V> {
        let map = self.map.read().unwrap();
        let value = match map.get(&key) {
            Some(CachedItem { value, .. }) => {
                let value = value.clone();
                drop(map);

                #[cfg(feature = "trace")]
                let _span = tracing::trace_span!("cache hit").entered();
                hit(&value);
                value
            }
            None => {
                drop(map);

                #[cfg(feature = "trace")]
                let _span = tracing::trace_span!("cache miss").entered();

                let value = Arc::new(miss());
                let item = CachedItem {
                    value: value.clone(),
                    life: 0,
                };

                let mut map = self.map.write().unwrap();
                match map.get_mut(&key) {
                    Some(value) => *value = item,
                    None => map.extend(Some((key, item))),
                }
                value
            }
        };

        value
    }

    /// Checkout the item with the given key. If the item doesn't exist, `miss` is called to construct it.
    pub fn checkout(&self, key: K, miss: impl FnOnce() -> V) -> Arc<V> {
        self.checkout_hit(key, miss, |_| ())
    }
}
