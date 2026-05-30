use std::collections::HashMap;

use itertools::Itertools;
use wasm_bindgen::prelude::*;
use web_rwkv::runtime::model::ModelInfo;

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SimpleSampler {
    info: ModelInfo,
}

#[wasm_bindgen]
impl SimpleSampler {
    #[wasm_bindgen(constructor)]
    pub fn new(info: &ModelInfo) -> Self {
        let info = info.clone();
        Self { info }
    }

    pub fn update(&mut self, _tokens: &[u32]) {}

    pub fn transform(&self, _logits: &mut [f32]) {}

    pub fn sample(&self, probs: &[f32]) -> u32 {
        let token = probs
            .iter()
            .take(self.info.num_vocab)
            .copied()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.total_cmp(y))
            .map(|(id, _)| id)
            .unwrap_or_default();
        token as u32
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct NucleusSampler {
    pub temp: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub count_penalty: f32,
    pub penalty_decay: f32,

    info: ModelInfo,
    state: HashMap<u32, f32>,
}

#[wasm_bindgen]
impl NucleusSampler {
    #[wasm_bindgen(constructor)]
    pub fn new(
        info: &ModelInfo,
        temp: f32,
        top_p: f32,
        presence_penalty: f32,
        count_penalty: f32,
        penalty_decay: f32,
    ) -> Self {
        Self {
            temp,
            top_p,
            presence_penalty,
            count_penalty,
            penalty_decay,
            info: info.clone(),
            state: Default::default(),
        }
    }

    pub fn update(&mut self, tokens: &[u32]) {
        for token in tokens {
            match self.state.get_mut(token) {
                Some(count) => *count += 1.0,
                None => {
                    self.state.insert(*token, 1.0);
                }
            }
        }
        for count in self.state.values_mut() {
            *count *= self.penalty_decay;
        }
    }

    pub fn transform(&self, logits: &mut [f32]) {
        for (&token, &count) in self.state.iter() {
            logits[token as usize] -= self.presence_penalty + self.count_penalty * count;
        }
    }

    pub fn sample(&self, probs: &[f32]) -> u32 {
        let sorted: Vec<_> = probs
            .iter()
            .take(self.info.num_vocab)
            .copied()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > self.top_p {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, x))
                }
            })
            .map(|(id, _, x)| (id, x.powf(1.0 / self.temp)))
            .collect();

        let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
        let sorted: Vec<_> = sorted
            .into_iter()
            .map(|(id, x)| (id, x / sum))
            .scan((0, 0.0), |(_, cum), (id, x)| {
                *cum += x;
                Some((id, *cum))
            })
            .collect();

        let rand: f32 = fastrand::f32();
        let token = sorted
            .into_iter()
            .find_or_first(|&(_, cum)| rand <= cum)
            .map(|(id, _)| id)
            .unwrap_or_default();
        token as u32
    }
}
