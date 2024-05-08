use std::{
    hash::Hash,
    sync::{Arc, RwLock},
};

use rustc_hash::FxHashMap as HashMap;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
struct CacheId;

type CachedItem<V> = (Arc<V>, uid::Id<CacheId>);

#[derive(Debug)]
pub struct ResourceCache<K, V> {
    map: RwLock<HashMap<K, CachedItem<V>>>,
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

    /// Checkout the item with the given key. If the item doesn't exist, `f` is called to construct it.
    pub fn checkout(&self, key: K, miss: impl FnOnce() -> V) -> Arc<V> {
        let map = self.map.read().unwrap();
        let value = match map.get(&key) {
            Some((value, _)) => value.clone(),
            None => {
                let value = Arc::new(miss());
                drop(map);

                let mut map = self.map.write().unwrap();
                map.insert(key, (value.clone(), uid::Id::new()));
                value
            }
        };

        // if self.limit > 0 {
        //     let remove_count = map.len() - self.limit.min(map.len());
        //     let remove = map
        //         .iter()
        //         .sorted_unstable_by_key(|(_, (_, id))| id.get())
        //         .map(|(key, _)| key)
        //         .take(remove_count)
        //         .cloned()
        //         .collect_vec();
        //     for key in remove {
        //         map.remove(&key);
        //     }
        // }

        value
    }
}
