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
use std::{io::Write, path::PathBuf};
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
#[cfg(feature = "trace")]
use tracing_subscriber::layer::SubscriberExt;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption},
        loader::{Loader, Lora},
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        softmax::softmax_one,
        v4, v5, v6, v7, TokioRuntime,
    },
    tensor::serialization::Seed,
    tokenizer::Tokenizer,
    wgpu,
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
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant_sf4: usize,
    #[arg(short, long)]
    embed_device: Option<EmbedDevice>,
    #[arg(short, long, action)]
    turbo: bool,
    #[arg(long, default_value_t = 32)]
    token_chunk_size: usize,
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,
    #[arg(short, long, action)]
    adapter: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("serialization", log::LevelFilter::Info)
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
            let context = context.clone();
            let model = builder.build_v4().await?;
            let f = move || serde::<v4::Model>(cli.output, &context, model);
            let model = tokio::task::spawn_blocking(f).await??;
            let bundle = v4::Bundle::<f16>::new(model, 1);
            TokioRuntime::new(bundle).await
        }
        ModelVersion::V5 => {
            let context = context.clone();
            let model = builder.build_v5().await?;
            let f = move || serde::<v5::Model>(cli.output, &context, model);
            let model = tokio::task::spawn_blocking(f).await??;
            let bundle = v5::Bundle::<f16>::new(model, 1);
            TokioRuntime::new(bundle).await
        }
        ModelVersion::V6 => {
            let context = context.clone();
            let model = builder.build_v6().await?;
            let f = move || serde::<v6::Model>(cli.output, &context, model);
            let model = tokio::task::spawn_blocking(f).await??;
            let bundle = v6::Bundle::<f16>::new(model, 1);
            TokioRuntime::new(bundle).await
        }
        ModelVersion::V7 => {
            let context = context.clone();
            let model = builder.build_v7().await?;
            let f = move || serde::<v7::Model>(cli.output, &context, model);
            let model = tokio::task::spawn_blocking(f).await??;
            let bundle = v7::Bundle::<f16>::new(model, 1);
            TokioRuntime::new(bundle).await
        }
    };

    const PROMPT: &str = include_str!("prompt.md");
    let tokens = tokenizer.encode(PROMPT.as_bytes())?;
    let prompt = InferInputBatch {
        tokens,
        option: InferOption::Last,
    };
    let mut prompt = InferInput::new(vec![prompt], cli.token_chunk_size);

    let num_token = 500;
    for _ in 0..num_token {
        let input = prompt.clone();
        let (input, output) = runtime.infer(input).await?;
        prompt = input;

        let output = output[0].0.clone();
        if output.size() > 0 {
            let output = softmax_one(&context, output).await?;
            let output = output.to_vec();
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

    Ok(())
}

fn serde<M>(output: Option<PathBuf>, context: &Context, model: M) -> Result<M>
where
    M: Serialize,
    for<'de> Seed<'de, Context, M>: DeserializeSeed<'de, Value = M>,
{
    struct FileWriter(std::fs::File);

    impl cbor4ii::core::enc::Write for FileWriter {
        type Error = std::io::Error;

        fn push(&mut self, input: &[u8]) -> Result<(), Self::Error> {
            self.0.write_all(input)
        }
    }

    if let Some(output) = output {
        let file = FileWriter(std::fs::File::open(output)?);
        let mut serializer = cbor4ii::serde::Serializer::new(file);

        model.serialize(&mut serializer)?;

        return Ok(model);
    }

    log::info!("serializing model...");
    let buf = cbor4ii::serde::to_vec(vec![], &model)?;
    log::info!(
        "serialized buffer size: {} ({} MB)",
        buf.len(),
        buf.len() / (1 << 20)
    );
    drop(model);

    let reader = cbor4ii::core::utils::SliceReader::new(&buf);
    let mut deserializer = cbor4ii::serde::Deserializer::new(reader);
    let seed = Seed::<Context, M>::new(context);

    log::info!("deserializing model...");
    let model: M = seed.deserialize(&mut deserializer)?;

    Ok(model)
}
