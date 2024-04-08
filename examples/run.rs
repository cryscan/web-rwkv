use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::PathBuf,
};

use anyhow::Result;
use clap::{Parser, ValueEnum};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use instant::{Duration, Instant};
#[cfg(not(debug_assertions))]
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption},
        loader::Loader,
        model::{Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        softmax::softmax,
        v4, v5, v6, JobRuntime, Submission,
    },
    tokenizer::Tokenizer,
};

fn sample(probs: &[f32], _top_p: f32) -> u16 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap()
        .0 as u16
}

async fn create_context(info: &ModelInfo, _auto: bool) -> Result<Context> {
    let instance = Instance::new();
    #[cfg(not(debug_assertions))]
    let adapter = if _auto {
        instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await?
    } else {
        let backends = wgpu::Backends::all();
        let adapters = instance
            .enumerate_adapters(backends)
            .map(|adapter| adapter.get_info())
            .map(|info| format!("{} ({:?})", info.name, info.backend))
            .collect_vec();
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Please select an adapter")
            .default(0)
            .items(&adapters)
            .interact()?;
        instance.select_adapter(backends, selection)?
    };
    #[cfg(debug_assertions)]
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .with_auto_limits(info)
        .build()
        .await?;
    println!("{:#?}", context.adapter.get_info());
    Ok(context)
}

fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/rwkv_vocab_v20230424.json")?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(Tokenizer::new(&contents)?)
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EmbedDevice {
    #[default]
    Cpu,
    Gpu,
}

impl From<EmbedDevice> for web_rwkv::runtime::model::EmbedDevice {
    fn from(value: EmbedDevice) -> Self {
        match value {
            EmbedDevice::Cpu => Self::Cpu,
            EmbedDevice::Gpu => Self::Gpu,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: Option<PathBuf>,
    #[arg(short, long, value_name = "FILE")]
    lora: Option<PathBuf>,
    #[arg(short, long, value_name = "LAYERS", default_value_t = 0)]
    quant: usize,
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant_nf4: usize,
    #[arg(short, long, action)]
    turbo: bool,
    #[arg(short, long)]
    embed_device: Option<EmbedDevice>,
    #[arg(long, default_value_t = 32)]
    token_chunk_size: usize,
    #[arg(short, long, action)]
    adapter: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let tokenizer = load_tokenizer()?;
    let model = cli.model.unwrap_or_else(|| {
        std::fs::read_dir("assets/models")
            .unwrap()
            .filter_map(|x| x.ok())
            .find(|x| x.path().extension().is_some_and(|x| x == "st"))
            .unwrap()
            .path()
    });

    let file = File::open(model)?;
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    println!("{:#?}", info);

    let context = create_context(&info, cli.adapter).await?;
    let quant = (0..cli.quant)
        .map(|layer| (layer, Quant::Int8))
        .chain((0..cli.quant_nf4).map(|layer| (layer, Quant::NF4)))
        .collect();
    let embed_device = cli.embed_device.unwrap_or_default().into();

    let builder = ModelBuilder::new(&context, model)
        .with_embed_device(embed_device)
        .with_quant(quant);

    let runtime = match info.version {
        ModelVersion::V4 => {
            let runtime = Build::<v4::ModelRuntime<f16, 1>>::build(builder).await?;
            JobRuntime::new(runtime).await
        }
        ModelVersion::V5 => {
            let runtime = Build::<v5::ModelRuntime<f16, 1>>::build(builder).await?;
            JobRuntime::new(runtime).await
        }
        ModelVersion::V6 => {
            let runtime = Build::<v6::ModelRuntime<f16, 1>>::build(builder).await?;
            JobRuntime::new(runtime).await
        }
    };

    const PROMPT: &str = include_str!("prompt.md");
    let tokens = tokenizer.encode(PROMPT.as_bytes())?;
    let prompt_len = tokens.len();
    let prompt = InferInputBatch {
        tokens,
        option: InferOption::Last,
        ..Default::default()
    };
    let mut prompt = InferInput::new([prompt], cli.token_chunk_size);

    let mut read = false;
    let mut instant = Instant::now();
    let mut prefill = Duration::ZERO;

    let num_token = 500;
    for _ in 0..num_token {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let input = prompt.clone();
        let submission = Submission { input, sender };

        let _ = runtime.send(submission).await;
        let Ok((input, output)) = receiver.await else {
            break;
        };
        prompt = input;

        let output = &output[0].output;
        if output.size() > 0 {
            if !read {
                print!("\n{}", PROMPT);
                prefill = instant.elapsed();
                instant = Instant::now();
                read = true;
            }

            let output = softmax(&context, output).await?;
            let probs = output.map(|x| x.to_f32()).to_vec();
            let token = sample(&probs, 0.0);
            prompt.batches[0].tokens.push(token);

            let decoded = tokenizer.decode(&[token])?;
            let word = String::from_utf8_lossy(&decoded);
            print!("{}", word);
            std::io::stdout().flush().unwrap();
        } else {
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }

    let duration = instant.elapsed();

    println!(
        "\n\nPrefill:\t{} tokens,\t{} mills,\t{} tps",
        prompt_len,
        prefill.as_millis(),
        prompt_len as f64 / prefill.as_secs_f64()
    );
    println!(
        "Generation:\t{} tokens,\t{} mills,\t{} tps",
        num_token,
        duration.as_millis(),
        num_token as f64 / duration.as_secs_f64()
    );
    std::io::stdout().flush()?;

    Ok(())
}
