use anyhow::Result;
use clap::Parser;
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
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
        loader::Loader, v4, v5, v6, Lora, Model, ModelBase, ModelBuilder, ModelInfo, ModelState,
        ModelVersion, Quant, StateBuilder,
    },
    tokenizer::Tokenizer,
};

fn sample(probs: &[f32], _top_p: f32) -> u16 {
    let sorted = probs
        .iter()
        .copied()
        .enumerate()
        .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
        // .scan((0, 0.0), |(_, cum), (id, x)| {
        //     if *cum > top_p {
        //         None
        //     } else {
        //         *cum += x;
        //         Some((id, *cum))
        //     }
        // })
        .collect_vec();
    // let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
    // let sorted = sorted.into_iter().map(|(id, x)| (id, x / sum));

    // let rand = fastrand::f32();
    // let token = sorted
    //     .into_iter()
    //     .find_or_first(|&(_, cum)| rand <= cum)
    //     .map(|(id, _)| id)
    //     .unwrap_or_default();
    let token = sorted[0].0;
    token as u16
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = Instance::new();
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: info.max_buffer_size() as u32,
        ..Default::default()
    };
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
        .with_default_pipelines()
        .with_limits(limits)
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

fn load_model<M: Model>(
    context: &Context,
    data: &[u8],
    lora: Option<PathBuf>,
    quant: Option<usize>,
    quant_nf4: Option<usize>,
    turbo: bool,
) -> Result<M> {
    let quant = quant
        .map(|layer| (0..layer).map(|layer| (layer, Quant::Int8)).collect_vec())
        .unwrap_or_default();
    let quant_nf4 = quant_nf4
        .map(|layer| (0..layer).map(|layer| (layer, Quant::NF4)).collect_vec())
        .unwrap_or_default();
    let quant = quant.into_iter().chain(quant_nf4).collect();
    let model = ModelBuilder::new(context, data)
        .with_quant(quant)
        .with_turbo(turbo);
    match lora {
        Some(lora) => {
            let file = File::open(lora)?;
            let map = unsafe { Mmap::map(&file)? };
            model
                .add_lora(Lora {
                    data: map.to_vec(),
                    blend: Default::default(),
                })
                .build()
        }
        None => model.build(),
    }
}

async fn run(cli: Cli) -> Result<()> {
    let tokenizer = load_tokenizer()?;
    let model = cli.model.unwrap_or(
        std::fs::read_dir("assets/models")
            .unwrap()
            .filter_map(|x| x.ok())
            .find(|x| x.path().extension().is_some_and(|x| x == "st"))
            .unwrap()
            .path(),
    );

    let file = File::open(model)?;
    let map = unsafe { Mmap::map(&file)? };

    let info = Loader::info(&map)?;
    println!("{:#?}", info);

    let context = create_context(&info).await?;

    match info.version {
        ModelVersion::V4 => {
            let model: v4::Model = load_model(
                &context,
                &map,
                cli.lora,
                cli.quant,
                cli.quant_nf4,
                cli.turbo,
            )?;
            let state: v4::ModelState = StateBuilder::new(&context, model.info()).build();
            run_internal(model, state, tokenizer).await
        }
        ModelVersion::V5 => {
            let model: v5::Model = load_model(
                &context,
                &map,
                cli.lora,
                cli.quant,
                cli.quant_nf4,
                cli.turbo,
            )?;
            let state: v5::ModelState = StateBuilder::new(&context, model.info()).build();
            run_internal(model, state, tokenizer).await
        }
        ModelVersion::V6 => {
            let model: v6::Model = load_model(
                &context,
                &map,
                cli.lora,
                cli.quant,
                cli.quant_nf4,
                cli.turbo,
            )?;
            let state: v6::ModelState = StateBuilder::new(&context, model.info()).build();
            run_internal(model, state, tokenizer).await
        }
    }
}

async fn run_internal<M, S>(model: M, state: S, tokenizer: Tokenizer) -> Result<()>
where
    S: ModelState,
    M: Model<ModelState = S>,
{
    let prompt = "The Space Needle is located in downtown";
    let mut tokens = vec![tokenizer.encode(prompt.as_bytes())?];
    print!("{}", prompt);
    let mut instant;
    let mut duration = Duration::default();

    let num_tokens = 100;
    for index in 0..=num_tokens {
        instant = Instant::now();
        let logits = model.run(&mut tokens, &state).await?;
        let probs = model.softmax(logits).await?;
        duration = match index {
            0 => Duration::default(),
            _ => duration + instant.elapsed(),
        };

        if let Some(probs) = &probs[0] {
            let token = sample(probs, 0.5);
            let decoded = tokenizer.decode(&[token])?;
            let word = String::from_utf8_lossy(&decoded);
            print!("{}", word);
            tokens[0] = vec![token];
        }
    }

    println!("\n{} tokens: {} mills", num_tokens, duration.as_millis());
    std::io::stdout().flush()?;

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: Option<PathBuf>,
    #[arg(short, long, value_name = "FILE")]
    lora: Option<PathBuf>,
    #[arg(short, long, value_name = "LAYERS")]
    quant: Option<usize>,
    #[arg(long, value_name = "LAYERS")]
    quant_nf4: Option<usize>,
    #[arg(short, long, action)]
    turbo: bool,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    run(cli).await.unwrap();
}
