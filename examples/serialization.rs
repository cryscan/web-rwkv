use std::{
    convert::Infallible,
    fs::File,
    io::{BufReader, Read, Write},
    path::PathBuf,
};

use anyhow::Result;
use clap::{Parser, ValueEnum};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
#[cfg(not(debug_assertions))]
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::{de::DeserializeSeed, Serialize};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{
        loader::{Loader, Lora},
        v4, v5, v6, Build, BuildFuture, Model, ModelBuilder, ModelInfo, ModelInput, ModelOutput,
        ModelState, ModelVersion, Quant, StateBuilder,
    },
    tensor::serialization::Seed,
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
async fn load_model<'a, M, S>(
    context: &Context,
    data: &'a [u8],
    lora: Option<&'a [u8]>,
    quant: usize,
    quant_nf4: usize,
    embed_device: Option<EmbedDevice>,
    turbo: bool,
    token_chunk_size: usize,
) -> Result<(M, S)>
where
    M: Model<State = S>,
    S: ModelState,
    ModelBuilder<SafeTensors<'a>>: BuildFuture<M, Error = anyhow::Error>,
    StateBuilder: Build<S, Error = Infallible>,
{
    let quant = (0..quant)
        .map(|layer| (layer, Quant::Int8))
        .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
        .collect();

    let model = SafeTensors::deserialize(data)?;
    let model = ModelBuilder::new(context, model)
        .with_quant(quant)
        .with_turbo(turbo)
        .with_token_chunk_size(token_chunk_size)
        .with_embed_device(embed_device.unwrap_or_default().into());
    let model: M = match lora {
        Some(lora) => {
            let data = SafeTensors::deserialize(lora)?;
            model
                .add_lora(Lora {
                    data,
                    blend: Default::default(),
                })
                .build()
                .await?
        }
        None => model.build().await?,
    };

    let state: S = StateBuilder::new(context, model.info()).build()?;
    Ok((model, state))
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
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    println!("{:#?}", info);

    let lora = match cli.lora {
        Some(lora) => {
            let file = File::open(lora)?;
            let data = unsafe { Mmap::map(&file)? };
            Some(data)
        }
        None => None,
    };
    let lora = lora.as_deref();

    let context = create_context(&info).await?;
    match info.version {
        ModelVersion::V4 => {
            let (model, state) = load_model(
                &context,
                &data,
                lora,
                cli.quant,
                cli.quant_nf4,
                cli.embed_device,
                cli.turbo,
                cli.token_chunk_size,
            )
            .await?;
            run_internal::<v4::Model<f16>, _>(model, state, tokenizer, cli.output).await
        }
        ModelVersion::V5 => {
            let (model, state) = load_model(
                &context,
                &data,
                lora,
                cli.quant,
                cli.quant_nf4,
                cli.embed_device,
                cli.turbo,
                cli.token_chunk_size,
            )
            .await?;
            run_internal::<v5::Model<f16>, _>(model, state, tokenizer, cli.output).await
        }
        ModelVersion::V6 => {
            let (model, state) = load_model(
                &context,
                &data,
                lora,
                cli.quant,
                cli.quant_nf4,
                cli.embed_device,
                cli.turbo,
                cli.token_chunk_size,
            )
            .await?;
            run_internal::<v6::Model<f16>, _>(model, state, tokenizer, cli.output).await
        }
    }
}

async fn run_internal<M, S>(
    model: M,
    state: S,
    tokenizer: Tokenizer,
    output: Option<PathBuf>,
) -> Result<()>
where
    S: ModelState,
    M: Model<State = S> + Serialize,
    for<'de> Seed<'de, Context, M>: DeserializeSeed<'de, Value = M>,
{
    if let Some(output) = output {
        println!("serializing model into {:?}", output);

        struct FileWriter(File);

        impl cbor4ii::core::enc::Write for FileWriter {
            type Error = std::io::Error;

            fn push(&mut self, input: &[u8]) -> Result<(), Self::Error> {
                self.0.write_all(input)
            }
        }

        let file = FileWriter(File::create(output)?);
        let mut serializer = cbor4ii::serde::Serializer::new(file);

        model.serialize(&mut serializer)?;

        return Ok(());
    }

    println!("serializing model...");
    let buf = cbor4ii::serde::to_vec(vec![], &model)?;
    println!(
        "serialized buffer size: {} ({} MB)",
        buf.len(),
        buf.len() / (1 << 20)
    );

    let context = model.context().clone();
    drop(model);

    let reader = cbor4ii::core::utils::SliceReader::new(&buf);
    let mut deserializer = cbor4ii::serde::Deserializer::new(reader);
    let seed = Seed::<Context, M>::new(&context);

    println!("deserializing model...");
    let model: M = seed.deserialize(&mut deserializer)?;

    println!("model reloaded");
    complete(model, state, tokenizer).await
}

async fn complete<M, S>(model: M, state: S, tokenizer: Tokenizer) -> Result<()>
where
    S: ModelState,
    M: Model<State = S>,
{
    const PROMPT: &str = "The eiffel tower is located in the city of";
    print!("{}", PROMPT);

    let mut tokens = vec![ModelInput {
        tokens: tokenizer.encode(PROMPT.as_bytes())?,
        ..Default::default()
    }];

    let mut count = 0usize;
    let num_tokens = 100;
    while count < num_tokens {
        let logits = model.run(&mut tokens, &state).await?;
        let probs = model.softmax(logits).await?;

        if let ModelOutput::Last(probs) = &probs[0] {
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
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    run(cli).await.unwrap();
}
