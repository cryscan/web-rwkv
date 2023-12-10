use std::{collections::HashMap, path::Path};

use anyhow::Result;
use half::{bf16, f16};
use itertools::Itertools;
use repugnant_pickle::{RepugnantTorchTensors as TorchTensors, TensorType};
use safetensors::{tensor::TensorView, Dtype};

pub const RENAME: [(&str, &str); 4] = [
    ("time_faaaa", "time_first"),
    ("time_maa", "time_mix"),
    ("lora_A", "lora.0"),
    ("lora_B", "lora.1"),
];

pub const TRANSPOSE: [&str; 4] = [
    "time_mix_w1",
    "time_mix_w2",
    "time_decay_w1",
    "time_decay_w2",
];

struct Tensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f16>,
}

fn load_tensors<'a, 'b, 'c>(
    data: &'a [u8],
    torch: TorchTensors,
    rename: impl IntoIterator<Item = (&'b str, &'b str)>,
    transpose: impl IntoIterator<Item = &'c str>,
) -> Vec<Tensor> {
    let mut tensors = vec![];
    let rename = rename.into_iter().collect_vec();
    let transpose = transpose.into_iter().collect_vec();

    for tensor in torch.into_iter() {
        let name = rename
            .iter()
            .fold(tensor.name, |name, (p, to)| name.replace(p, to));
        let shape = tensor.shape;
        let size: usize = shape.iter().product();
        let bytes = size * tensor.tensor_type.size();

        assert!(matches!(tensor.tensor_type, TensorType::BFloat16));
        let start = tensor.absolute_offset as usize;
        let end = start + bytes;
        let data: &[bf16] = bytemuck::cast_slice(&data[start..end]);
        let data: Vec<_> = data.iter().map(|x| f16::from_f32(x.to_f32())).collect();

        if transpose.iter().any(|p| name.contains(p)) {
            let mut transposed = vec![f16::ZERO; data.len()];
            let num_col = *shape.iter().nth_back(0).expect("should be at least 2d");
            let num_row = *shape.iter().nth_back(1).expect("should be at least 2d");
            let num_batch = *shape.iter().nth_back(2).unwrap_or(&1);
            for b in 0..num_batch {
                for i in 0..num_row {
                    for j in 0..num_col {
                        let from = b * num_col * num_row + i * num_col + j;
                        let to = b * num_col * num_row + j * num_row + i;
                        transposed[to] = data[from];
                    }
                }
            }
            let mut shape = shape;
            *shape.iter_mut().nth_back(0).unwrap() = num_row;
            *shape.iter_mut().nth_back(1).unwrap() = num_col;

            println!("{name}\t{:?}\t(Transposed)", shape);
            tensors.push(Tensor {
                name,
                shape,
                data: transposed,
            });
        } else {
            println!("{name}\t{:?}", shape);
            tensors.push(Tensor { name, shape, data });
        }
    }

    tensors
}

pub fn convert_safetensors<'a, 'b, 'c>(
    input: impl AsRef<Path>,
    data: &'a [u8],
    output: impl AsRef<Path>,
    rename: impl IntoIterator<Item = (&'b str, &'b str)>,
    transpose: impl IntoIterator<Item = &'c str>,
) -> Result<()> {
    let torch = TorchTensors::new_from_file(input)?;
    let tensors = load_tensors(data, torch, rename, transpose);
    let views = tensors
        .iter()
        .map(|x| TensorView::new(Dtype::F16, x.shape.clone(), bytemuck::cast_slice(&x.data)))
        .collect::<Result<Vec<_>, _>>()?;
    let data = tensors
        .iter()
        .zip(views)
        .map(|(tensor, view)| (tensor.name.clone(), view))
        .collect::<HashMap<_, _>>();

    safetensors::serialize_to_file(&data, &None, output.as_ref())?;
    Ok(())
}
