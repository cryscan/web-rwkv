use std::{io::Write, path::PathBuf};

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
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption},
        loader::{Loader, Lora},
        model::{Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        softmax::softmax_one,
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
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

async fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/rwkv_vocab_v20230424.json").await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
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
    model: PathBuf,
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
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("rt_gen", log::LevelFilter::Info)
        .init()
        .unwrap();
    let cli = Cli::parse();

    let tokenizer = load_tokenizer().await?;

    let file = File::open(cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    log::info!("{:#?}", info);

    let context = create_context(&info, cli.adapter).await?;
    log::info!("{:#?}", context.adapter.get_info());

    let quant = (0..cli.quant)
        .map(|layer| (layer, Quant::Int8))
        .chain((0..cli.quant_nf4).map(|layer| (layer, Quant::NF4)))
        .collect();
    let embed_device = cli.embed_device.unwrap_or(EmbedDevice::Cpu).into();
    let lora = match cli.lora {
        Some(path) => {
            let file = File::open(path).await?;
            let mut reader = BufReader::new(file);
            let mut data = vec![];
            reader.read_to_end(&mut data).await?;
            Some(data)
        }
        None => None,
    };

    let builder = ModelBuilder::new(&context, model)
        .embed_device(embed_device)
        .quant(quant);
    let builder = match &lora {
        Some(data) => {
            let data = SafeTensors::deserialize(data)?;
            let blend = Default::default();
            let lora = Lora { data, blend };
            builder.lora(lora)
        }
        None => builder,
    };

    let runtime = match info.version {
        ModelVersion::V4 => {
            let model = Build::<v4::Model>::build(builder).await?;
            let builder = v4::ModelJobBuilder::<f16>::new(model, 1);
            JobRuntime::new(builder).await
        }
        ModelVersion::V5 => {
            let model = Build::<v5::Model>::build(builder).await?;
            let builder = v5::ModelJobBuilder::<f16>::new(model, 1);
            JobRuntime::new(builder).await
        }
        ModelVersion::V6 => {
            let model = Build::<v6::Model>::build(builder).await?;
            let builder = v6::ModelJobBuilder::<f16>::new(model, 1);
            JobRuntime::new(builder).await
        }
    };

    // const PROMPT: &str = "User: Hi!\n\nAssistant: Hello! I'm your AI assistant. I'm here to help you with various tasks, such as answering questions, brainstorming ideas, drafting emails, writing code, providing advice, and much more.\n\nUser: Hi!\n\nAssistant:";
    const PROMPT: &str = include_str!("prompt.md");
    let tokens = tokenizer.encode(PROMPT.as_bytes())?;
    let prompt_len = tokens.len();
    let prompt = InferInputBatch {
        tokens,
        option: InferOption::Last,
    };
    let mut prompt = InferInput::new(vec![prompt], cli.token_chunk_size);

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

        let output = output[0].0.clone();
        if output.size() > 0 {
            if !read {
                print!("\n{}", PROMPT);
                prefill = instant.elapsed();
                instant = Instant::now();
                read = true;
            }

            let output = softmax_one(&context, output).await?;
            let output = output.map(|x| x.to_f32()).to_vec();
            let token = sample(&output, 0.0);
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
    print!("\n\n");

    let duration = instant.elapsed();
    log::info!(
        "Prefill:\t{} tokens,\t{} mills,\t{} tps",
        prompt_len,
        prefill.as_millis(),
        prompt_len as f64 / prefill.as_secs_f64()
    );
    log::info!(
        "Generation:\t{} tokens,\t{} mills,\t{} tps",
        num_token,
        duration.as_millis(),
        num_token as f64 / duration.as_secs_f64()
    );

    Ok(())
}
