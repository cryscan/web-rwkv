use std::{collections::HashMap, io::Cursor};

use base64::{engine::general_purpose::STANDARD, Engine};
use image::{ImageFormat, RgbImage};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_rwkv::runtime::model::{ModelInfo, ModelVersion};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateStats {
    layer: usize,
    head: usize,
    bins: [f32; 7],
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVisual {
    num_layer: usize,
    num_head: usize,
    stats: Vec<StateStats>,
    images: Vec<Vec<String>>,
}

#[wasm_bindgen]
impl StateVisual {
    #[wasm_bindgen(constructor)]
    pub fn new(info: &ModelInfo, state: &[f32]) -> Result<Self, JsError> {
        let num_layer = info.num_layer;
        let num_head = info.num_head;
        let head_size = info.num_emb / info.num_head;

        let mut heads: HashMap<(usize, usize), Vec<f32>> = HashMap::new();

        for (index, &value) in state.iter().enumerate() {
            match info.version {
                ModelVersion::V4 => break,
                ModelVersion::V5 | ModelVersion::V6 | ModelVersion::V7 => {
                    let column = index % info.num_emb;
                    let line = index / info.num_emb;

                    let head = column / head_size;
                    let layer = line / (head_size + 2);

                    let x = column % head_size;
                    let y = line % (head_size + 2);
                    if y == 0 || y == head_size + 1 {
                        continue;
                    }
                    let y = y - 1;

                    let key = (layer, head);
                    match heads.get_mut(&key) {
                        Some(values) => values[y * head_size + x] = value,
                        None => {
                            let mut values = vec![0.0; head_size * head_size];
                            values[y * head_size + x] = value;
                            heads.insert(key, values);
                        }
                    }
                }
            }
        }

        let stats: HashMap<_, _> = heads
            .iter()
            .map(|(&(layer, head), values)| {
                let mut values = values.clone();
                values.sort_by(|x, y| x.total_cmp(y));
                let p0 = 0usize;
                let p4 = values.len() - 1;
                let p2 = (p0 + p4) / 2;
                let p1 = (p0 + p2) / 2;
                let p3 = (p2 + p4) / 2;
                let p_005 = ((p4 as f32) * 0.005) as usize;
                let p_995 = ((p4 as f32) * 0.995) as usize;

                let min = values[p0];
                let max = values[p4];
                let q1 = (values[p1] + values[p1 + 1]) / 2.0;
                let q2 = (values[p2] + values[p2 + 1]) / 2.0;
                let q3 = (values[p3] + values[p3 + 1]) / 2.0;
                let q_005 = (values[p_005] + values[p_005 + 1]) / 2.0;
                let q_995 = (values[p_995] + values[p_995 + 1]) / 2.0;
                let bins = [min, q_005, q1, q2, q3, q_995, max];
                ((layer, head), bins)
            })
            .collect();

        let mut images = Vec::with_capacity(num_layer);
        for layer in 0..num_layer {
            let mut line = Vec::with_capacity(num_head);
            for head in 0..num_head {
                let key = (layer, head);
                let data = heads.get(&key).unwrap();
                let bins = stats.get(&key).unwrap();

                assert_eq!(data.len(), head_size * head_size);
                let h = head_size as u32;
                const SCALE: u32 = 4;

                let [_, q0, q1, _, q3, q4, _] = bins;
                let r13 = (q3 - q1).max(f32::EPSILON);
                let r04 = (q4 - q0).max(f32::EPSILON);

                let mut image = RgbImage::new(SCALE * h, SCALE * h);
                for (y, x) in (0..SCALE * h).cartesian_product(0..SCALE * h) {
                    let pixel = image.get_pixel_mut(x, y);
                    let index = (y / SCALE * h + x / SCALE) as usize;
                    let value = data[index];

                    let r = ((value - q1) / r13).clamp(0.0, 1.0) * 255.0;
                    let b = ((value - q0) / r04).clamp(0.0, 1.0) * 255.0;
                    pixel.0 = [r as u8, r as u8, b as u8];
                }

                let mut bytes: Vec<u8> = Vec::new();
                image
                    .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
                    .expect("failed to write image to bytes");
                line.push(STANDARD.encode(bytes));
            }
            images.push(line);
        }

        let stats = stats
            .into_iter()
            .map(|((layer, head), bins)| StateStats { layer, head, bins })
            .sorted_by_key(|x| (x.layer, x.head))
            .collect_vec();

        Ok(Self {
            num_layer,
            num_head,
            stats,
            images,
        })
    }

    pub fn json(&self) -> Result<String, JsError> {
        serde_json::to_string(&self).map_err(err)
    }
}

fn err(err: impl ToString) -> JsError {
    JsError::new(&err.to_string())
}
