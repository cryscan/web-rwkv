use anyhow::Result;
use gpp::Context;
use itertools::Itertools;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[derive(Debug)]
pub struct Shader {
    source: String,
    cache: Arc<Mutex<HashMap<ShaderKey, String>>>,
}

impl Shader {
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn specialize(&self, macros: HashMap<String, String>) -> Result<String> {
        let key = ShaderKey::new(&macros);
        let mut cache = self.cache.lock().unwrap();
        let shader = cache.remove(&key);
        let shader = match shader {
            Some(shader) => shader,
            None => {
                let mut context = Context::new();
                context.macros = macros;
                gpp::process_str(&self.source, &mut context)?
            }
        };
        Ok(shader)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
struct ShaderKey(Vec<(String, String)>);

impl ShaderKey {
    pub fn new(macros: &HashMap<String, String>) -> Self {
        let macros = macros.clone().into_iter().sorted().collect();
        Self(macros)
    }
}
