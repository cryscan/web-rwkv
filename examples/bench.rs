//! A simple example to benchmark the inference performance of the model.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
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
#[cfg(feature = "trace")]
use tracing_subscriber::layer::SubscriberExt;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{IntoTokens, Rnn, RnnInput, RnnInputBatch, RnnOption, Token},
        loader::{Loader, Lora},
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        softmax::softmax_one,
        v4, v5, v6, v7, Runtime, TokioRuntime,
    },
};

const REPEAT: usize = 5;

async fn create_context(info: &ModelInfo, _auto: bool) -> Result<Context> {
    let instance = wgpu::Instance::default();
    #[cfg(not(debug_assertions))]
    let adapter = if _auto {
        instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await?
    } else {
        let backends = wgpu::Backends::all();
        let adapters = instance.enumerate_adapters(backends);
        let names = adapters
            .iter()
            .map(|adapter| adapter.get_info())
            .map(|info| format!("{} ({:?})", info.name, info.backend))
            .collect_vec();
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Please select an adapter")
            .default(0)
            .items(&names)
            .interact()?;
        adapters.into_iter().nth(selection).unwrap()
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

fn sample(probs: &[f32], _top_p: f32) -> u16 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap()
        .0 as u16
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
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant_sf4: usize,
    #[arg(long, default_value_t = 128)]
    token_chunk_size: usize,
    #[arg(short, long, action)]
    adapter: bool,
    #[arg(long, default_value_t = 512)]
    prefill_length: usize,
    #[arg(long, default_value_t = 128)]
    generation_length: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("bench", log::LevelFilter::Info)
        .init()?;
    #[cfg(feature = "trace")]
    {
        let registry = tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default());
        tracing::subscriber::set_global_default(registry)?;
    }

    let cli = Cli::parse();
    let model_path = cli.model.clone();

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
        .chain((0..cli.quant_sf4).map(|layer| (layer, Quant::SF4)))
        .collect();
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

    let builder = ModelBuilder::new(&context, model).quant(quant);
    let builder = match &lora {
        Some(data) => {
            let data = SafeTensors::deserialize(data)?;
            let blend = Default::default();
            let lora = Lora { data, blend };
            builder.lora(lora)
        }
        None => builder,
    };

    let runtime: Box<dyn Runtime<Rnn>> = match info.version {
        ModelVersion::V4 => {
            let model = builder.build_v4().await?;
            let bundle = v4::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V5 => {
            let model = builder.build_v5().await?;
            let bundle = v5::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V6 => {
            let model = builder.build_v6().await?;
            let bundle = v6::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
    };

    fastrand::seed(42);
    let prompt_len = cli.prefill_length;

    println!("| model                                                    | quant_int8 | quant_float4 |    test |            t/s |");
    println!("|----------------------------------------------------------|-----------:|-------------:|--------:|---------------:|");

    let mut prefill = Duration::ZERO;
    for _ in 0..REPEAT {
        let tokens: Vec<u16> = (0..prompt_len)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch {
            tokens: tokens.into_tokens(),
            option: RnnOption::Last,
        };
        let mut prompt = RnnInput::new(vec![prompt], cli.token_chunk_size);
        let instant = Instant::now();
        for _ in 0..prompt_len {
            let input = prompt.clone();
            let (input, output) = runtime.infer(input).await?;
            prompt = input;

            let output = output[0].0.clone();
            if output.size() > 0 {
                let output = softmax_one(&context, output).await?;
                let output = output.to_vec();
                let _ = sample(&output, 0.0);

                let duration = instant.elapsed();
                prefill += duration;
                break;
            }
        }
    }
    let tps = prompt_len as f64 / prefill.as_secs_f64() * REPEAT as f64;
    println!(
        "| {:<56} | {:>10} | {:>12} | {:>7} | {:>14} |",
        model_path.file_name().unwrap().to_string_lossy(),
        cli.quant,
        cli.quant_nf4.max(cli.quant_sf4),
        format!("pp{}", cli.prefill_length.to_string().clone()),
        format!("{:.2}", tps)
    );

    let num_token = cli.generation_length;
    let mut generation = Duration::ZERO;
    for _ in 0..REPEAT {
        let tokens: Vec<u16> = vec![0];
        let prompt = RnnInputBatch {
            tokens: tokens.into_tokens(),
            option: RnnOption::Last,
        };
        let mut prompt = RnnInput::new(vec![prompt], cli.token_chunk_size);
        let instant = Instant::now();
        for _ in 0..num_token {
            let input = prompt.clone();
            let (input, output) = runtime.infer(input).await?;
            prompt = input;

            let output = output[0].0.clone();
            if output.size() > 0 {
                let output = softmax_one(&context, output).await?;
                let output = output.to_vec();
                let token = sample(&output, 0.0);
                prompt.batches[0].tokens.push(Token::Token(token));
            }
        }
        generation += instant.elapsed();
    }
    let tps = num_token as f64 / generation.as_secs_f64() * REPEAT as f64;
    println!(
        "| {:<56} | {:>10} | {:>12} | {:>7} | {:>14} |",
        model_path.file_name().unwrap().to_string_lossy(),
        cli.quant,
        cli.quant_nf4.max(cli.quant_sf4),
        format!("tg{}", cli.generation_length.to_string().clone()),
        format!("{:.2}", tps)
    );
    Ok(())
}
