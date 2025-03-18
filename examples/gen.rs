//! A simple example on loading the tokenizer and the model, reading a long prompt,
//! and doing continuation.

use std::{io::Write, path::PathBuf};

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
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::{Loader, Lora},
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        softmax::softmax_one,
        v4, v5, v6, v7, Runtime, TokioRuntime,
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

async fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/vocab/rwkv_vocab_v20230424.json").await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    Ok(Tokenizer::new(&contents)?)
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
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("gen", log::LevelFilter::Info)
        .init()?;
    #[cfg(feature = "trace")]
    {
        let registry = tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default());
        tracing::subscriber::set_global_default(registry)?;
    }

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

    const PROMPT: &str = include_str!("prompt.md");
    let tokens = tokenizer.encode(PROMPT.as_bytes())?;
    let prompt_len = tokens.len();
    let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
    let mut prompt = RnnInput::new(vec![prompt], cli.token_chunk_size);

    let mut read = false;
    let mut instant = Instant::now();
    let mut prefill = Duration::ZERO;

    let num_token = 500;
    for _ in 0..num_token {
        // each time `runtime.infer` is called,
        // it consumes a chunk of the input and returns the remaining back
        let input = prompt.clone();
        let (input, output) = runtime.infer(input).await?;
        prompt = input;

        let output = output[0].0.clone();
        if output.size() > 0 {
            // the runtime is producing output: we have read the prompt and start the inference
            if !read {
                // just read the whole prompt, print the prompt and reset the timer
                print!("\n{}", PROMPT);
                prefill = instant.elapsed();
                instant = Instant::now();
                read = true;
            }

            let output = softmax_one(&context, output).await?;
            let output = output.to_vec();
            let token = sample(&output, 0.0);
            prompt.batches[0].push(token);

            let decoded = tokenizer.decode(&[token])?;
            let word = String::from_utf8_lossy(&decoded);
            print!("{}", word);
            std::io::stdout().flush().unwrap();
        } else {
            // reading the prompt, print "." every `token_chunk_size` tokens
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }
    print!("\n\n");

    let duration = instant.elapsed();
    log::info!(
        "prefill:\t{} tokens,\t{} mills,\t{} tps",
        prompt_len,
        prefill.as_millis(),
        prompt_len as f64 / prefill.as_secs_f64()
    );
    log::info!(
        "inference:\t{} tokens,\t{} mills,\t{} tps",
        num_token,
        duration.as_millis(),
        num_token as f64 / duration.as_secs_f64()
    );

    Ok(())
}
