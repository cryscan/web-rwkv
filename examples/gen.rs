use anyhow::Result;
use clap::{Parser, ValueEnum};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
#[cfg(not(debug_assertions))]
use itertools::Itertools;
use memmap2::Mmap;
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::PathBuf,
    time::{Duration, Instant},
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{
        loader::Loader, v4, v5, v6, Lora, Model, ModelBase, ModelBuilder, ModelInfo, ModelInput,
        ModelOutput, ModelState, ModelVersion, Quant, StateBuilder,
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

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = Instance::new();
    #[cfg(not(debug_assertions))]
    let adapter = {
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

#[allow(clippy::too_many_arguments)]
fn load_model<M: Model>(
    context: &Context,
    data: &[u8],
    lora: Option<PathBuf>,
    quant: usize,
    quant_nf4: usize,
    embed_device: Option<EmbedDevice>,
    turbo: bool,
    token_chunk_size: usize,
) -> Result<M> {
    let quant = (0..quant)
        .map(|layer| (layer, Quant::Int8))
        .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
        .collect();
    let model = ModelBuilder::new(context, data)
        .with_quant(quant)
        .with_turbo(turbo)
        .with_token_chunk_size(token_chunk_size)
        .with_embed_device(embed_device.unwrap_or_default().into());
    match lora {
        Some(lora) => {
            let file = File::open(lora)?;
            let map = unsafe { Mmap::map(&file)? };
            model
                .add_lora(Lora {
                    data: &map,
                    blend: Default::default(),
                })
                .build()
        }
        None => model.build(),
    }
}

async fn run(cli: Cli) -> Result<()> {
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
    let map = unsafe { Mmap::map(&file)? };

    let info = Loader::info(&map)?;
    println!("{:#?}", info);

    let context = create_context(&info).await?;

    match info.version {
        ModelVersion::V4 => {
            let model: v4::Model<f16> = load_model(
                &context,
                &map,
                cli.lora,
                cli.quant,
                cli.quant_nf4,
                cli.embed_device,
                cli.turbo,
                cli.token_chunk_size,
            )?;
            let state: v4::ModelState = StateBuilder::new(&context, model.info()).build();
            run_internal(model, state, tokenizer).await
        }
        ModelVersion::V5 => {
            let model: v5::Model<f16> = load_model(
                &context,
                &map,
                cli.lora,
                cli.quant,
                cli.quant_nf4,
                cli.embed_device,
                cli.turbo,
                cli.token_chunk_size,
            )?;
            let state: v5::ModelState = StateBuilder::new(&context, model.info()).build();
            run_internal(model, state, tokenizer).await
        }
        ModelVersion::V6 => {
            let model: v6::Model<f16> = load_model(
                &context,
                &map,
                cli.lora,
                cli.quant,
                cli.quant_nf4,
                cli.embed_device,
                cli.turbo,
                cli.token_chunk_size,
            )?;
            let state: v6::ModelState = StateBuilder::new(&context, model.info()).build();
            run_internal(model, state, tokenizer).await
        }
    }
}

async fn run_internal<M, S>(model: M, state: S, tokenizer: Tokenizer) -> Result<()>
where
    S: ModelState,
    M: Model<State = S>,
{
    const PROMPT: &str = include_str!("prompt.md");
    let mut tokens = vec![ModelInput {
        tokens: tokenizer.encode(PROMPT.as_bytes())?,
        ..Default::default()
    }];

    let prompt_len = tokens[0].tokens.len();
    println!("Reading {} tokens.", prompt_len);

    let mut read = false;
    let mut count = 0usize;

    let mut instant;
    let mut prefill = Duration::ZERO;
    let mut duration = Duration::ZERO;

    let num_tokens = 500;
    while count < num_tokens {
        instant = Instant::now();
        let logits = model.run(&mut tokens, &state).await?;
        let probs = model.softmax(logits).await?;
        duration += instant.elapsed();

        if let ModelOutput::Last(probs) = &probs[0] {
            if !read {
                print!("\n{}", PROMPT);
                prefill = duration;
                duration = Duration::ZERO;
                read = true;
            }

            let token = sample(probs, 0.5);
            let decoded = tokenizer.decode(&[token])?;
            let word = String::from_utf8_lossy(&decoded);
            print!("{}", word);
            std::io::stdout().flush().unwrap();

            tokens[0].tokens = vec![token];
            count += 1;
        } else {
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }

    println!();
    println!(
        "Prefill: {} tokens, {} mills, {} tps.",
        prompt_len,
        prefill.as_millis(),
        prompt_len as f64 / prefill.as_secs_f64()
    );
    println!(
        "Generation: {} tokens, {} mills, {} tps.",
        num_tokens,
        duration.as_millis(),
        num_tokens as f64 / duration.as_secs_f64()
    );
    std::io::stdout().flush()?;

    Ok(())
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EmbedDevice {
    #[default]
    Cpu,
    Gpu,
}

impl From<EmbedDevice> for web_rwkv::model::EmbedDevice {
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
    #[arg(short, long)]
    embed_device: Option<EmbedDevice>,
    #[arg(short, long, action)]
    turbo: bool,
    #[arg(long, default_value_t = 32)]
    token_chunk_size: usize,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    run(cli).await.unwrap();
}
