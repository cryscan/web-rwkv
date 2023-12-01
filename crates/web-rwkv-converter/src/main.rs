use std::{collections::HashMap, fs::File, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use half::{bf16, f16};
use memmap2::Mmap;
use repugnant_pickle::{RepugnantTorchTensors as TorchTensors, TensorType};
use safetensors::{tensor::TensorView, Dtype};

struct Tensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f16>,
}

fn load_tensors(
    data: &[u8],
    torch: TorchTensors,
    rename: &[(&str, &str)],
    transpose: &[&str],
) -> Vec<Tensor> {
    let mut tensors = vec![];

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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let tensors = TorchTensors::new_from_file(&cli.input)?;
    // print!("{:#?}", tensors);

    let file = File::open(&cli.input)?;
    let map = unsafe { Mmap::map(&file)? };

    let rename = [
        ("time_faaaa", "time_first"),
        ("time_maa", "time_mix"),
        ("lora_A", "lora.0"),
        ("lora_B", "lora.1"),
    ];
    let transpose = [
        "time_mix_w1",
        "time_mix_w2",
        "time_decay_w1",
        "time_decay_w2",
    ];

    let tensors = load_tensors(&map, tensors, &rename, &transpose);
    let views = tensors
        .iter()
        .map(|x| TensorView::new(Dtype::F16, x.shape.clone(), bytemuck::cast_slice(&x.data)))
        .collect::<Result<Vec<_>, _>>()?;
    let metadata: HashMap<String, TensorView> = tensors
        .iter()
        .zip(views)
        .map(|(tensor, view)| (tensor.name.clone(), view))
        .collect();

    let output = cli.output.unwrap_or_else(|| {
        let path = cli
            .input
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        let stem = cli.input.file_stem().expect("please name the file");
        let name: PathBuf = [&stem.to_string_lossy(), "st"].join(".").into();
        path.join(name)
    });
    safetensors::serialize_to_file(&metadata, &None, &output)?;

    Ok(())
}
