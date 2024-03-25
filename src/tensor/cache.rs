use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex},
};

use instant::{Duration, Instant};

#[derive(Debug)]
struct CacheItem<V> {
    value: Arc<V>,
    instant: Instant,
}

impl<V> CacheItem<V> {
    fn new(value: Arc<V>) -> Self {
        Self {
            value,
            instant: Instant::now(),
        }
    }

    fn count(&self) -> usize {
        Arc::strong_count(&self.value)
    }
}

#[derive(Debug)]
pub struct ResourceCache<K, V> {
    duration: Duration,
    map: Mutex<HashMap<K, Vec<CacheItem<V>>>>,
}

impl<K, V> Default for ResourceCache<K, V> {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(1),
            map: Default::default(),
        }
    }
}

impl<K, V> ResourceCache<K, V>
where
    K: PartialEq + Eq + Hash,
{
    /// Note: If `duration` is 0, the cache won't evict any items.
    pub fn new(duration: Duration) -> Self {
        Self {
            duration,
            map: Default::default(),
        }
    }

    /// Checkout the item with the given key. If the item doesn't exist, `f` is called to construct it.
    pub fn checkout(&self, key: K, hit: impl FnOnce(&V), miss: impl FnOnce() -> V) -> Arc<V> {
        // let mut map = self.map.lock().unwrap();
        // if !self.duration.is_zero() {
        //     for (_, items) in map.iter_mut() {
        //         items.retain(|item| item.count() > 1 && item.instant.elapsed() < self.duration);
        //     }
        // }

        // match map
        //     .get_mut(&key)
        //     .and_then(|items| items.iter_mut().find(|item| item.count() == 1))
        // {
        //     Some(item) => {
        //         hit(&item.value);
        //         item.instant = Instant::now();
        //         item.value.clone()
        //     }
        //     None => {
        //         let value = Arc::new(miss());
        //         map.insert(key, vec![CacheItem::new(value.clone())]);
        //         value
        //     }
        // }
        miss().into()
    }
}
