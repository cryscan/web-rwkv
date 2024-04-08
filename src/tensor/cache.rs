use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex},
};

use itertools::Itertools;
use uid::Id;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
struct CacheId;

type CachedItem<V> = (Arc<V>, Id<CacheId>);

#[derive(Debug)]
pub struct ResourceCache<K, V> {
    map: Mutex<HashMap<K, CachedItem<V>>>,
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

    /// Checkout the item with the given key. If the item doesn't exist, `f` is called to construct it.
    pub fn checkout(&self, key: K, miss: impl FnOnce() -> V) -> Arc<V> {
        let mut map = self.map.lock().unwrap();

        let value = match map.remove(&key) {
            Some((value, _)) => value,
            None => Arc::new(miss()),
        };

        if self.limit > 0 {
            let remove_count = map.len() - self.limit.min(map.len());
            let remove = map
                .iter()
                .sorted_unstable_by_key(|(_, (_, id))| id.get())
                .map(|(key, _)| key)
                .take(remove_count)
                .cloned()
                .collect_vec();
            for key in remove {
                map.remove(&key);
            }
        }

        map.insert(key, (value.clone(), Id::new()));
        value
    }
}
