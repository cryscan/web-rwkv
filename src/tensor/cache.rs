use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex},
};

#[derive(Debug)]
pub struct ResourceCache<K, V> {
    map: Mutex<HashMap<K, Arc<V>>>,
}

impl<K, V> Default for ResourceCache<K, V> {
    fn default() -> Self {
        Self {
            map: Default::default(),
        }
    }
}

impl<K, V> ResourceCache<K, V>
where
    K: PartialEq + Eq + Hash,
{
    /// Checkout the item with the given key. If the item doesn't exist, `f` is called to construct it.
    pub fn checkout(&self, key: K, miss: impl FnOnce() -> V) -> Arc<V> {
        let mut map = self.map.lock().unwrap();
        match map.get(&key) {
            Some(value) => value.clone(),
            None => {
                let value = Arc::new(miss());
                map.insert(key, value.clone());
                value
            }
        }
    }
}
