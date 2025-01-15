use std::{io::Write, path::PathBuf};

use anyhow::Result;
use clap::{Parser, ValueEnum};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
#[cfg(not(debug_assertions))]
use itertools::Itertools;
use memmap2::Mmap;
use ops::TensorOpExt;
use safetensors::SafeTensors;
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    num::Float,
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo},
        v7, TokioRuntime,
    },
    tensor::ops::TensorOp,
    tokenizer::Tokenizer,
    wgpu,
};

mod ops;

const PROMPT: &str = r"<input>
● ● ● ● ● ● ● ● 
● · ○ ○ ● ● ● ○ 
● ○ ○ ○ ○ ● ● ○ 
● ○ ○ ○ ○ ● ● ● 
● ○ ○ ○ ○ ● ● ● 
● ○ ○ ○ ○ ● ● ● 
● · · · · ● ● ● 
● · · ○ ○ ○ ○ ○ 
NEXT ● 
MAX_WIDTH-2
MAX_DEPTH-2
</input>

";

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
    let file = File::open("assets/vocab/othello_vocab.json").await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    Ok(Tokenizer::new(&contents)?)
}

/// We need to slightly modify the model structure using hooks.
fn make_hooks<F: Float>(info: &ModelInfo) -> Result<v7::HookMap<F>> {
    let mut hooks = v7::HookMap::new();

    for layer in 0..info.num_layer {
        hooks.insert(
            v7::Hook::PostAttAdapt(layer),
            Box::new(move |frame: v7::Frame<F>| {
                let op = TensorOp::discount(&frame.buffer.att_a, 2.0, 0.0)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
        hooks.insert(
            v7::Hook::PostAttControl(layer),
            Box::new(move |frame: v7::Frame<F>| {
                let op = TensorOp::mul_w(&frame.buffer.att_w, &frame.buffer.att_a)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
    }

    Ok(hooks)
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
    #[arg(short, long)]
    embed_device: Option<EmbedDevice>,
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
        .with_module_level("othello", log::LevelFilter::Info)
        .init()?;
    let cli = Cli::parse();

    let tokenizer = load_tokenizer().await?;

    let model = cli
        .model
        .unwrap_or("assets/models/rwkv7_othello_9m_L10_D256_extended.st".into());
    let file = File::open(model).await?;
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    log::info!("{:#?}", info);

    let context = create_context(&info, cli.adapter).await?;
    log::info!("{:#?}", context.adapter.get_info());

    let embed_device = cli.embed_device.unwrap_or(EmbedDevice::Cpu).into();

    let hooks = make_hooks(&info)?;
    let model = ModelBuilder::new(&context, model)
        .embed_device(embed_device)
        .rescale(0)
        .build_v7()
        .await?;
    let bundle = v7::Bundle::<f16>::new_with_hooks(model, 1, hooks);
    let runtime = TokioRuntime::new(bundle).await;

    let tokens = tokenizer.encode(PROMPT.as_bytes())?;
    let prompt = InferInputBatch {
        tokens,
        option: InferOption::Last,
    };
    let mut prompt = InferInput::new(vec![prompt], cli.token_chunk_size);

    print!("{PROMPT}");

    loop {
        let input = prompt.clone();
        let (input, output) = runtime.infer(input).await?;
        prompt = input;

        let output = output[0].0.clone();
        if output.size() > 0 {
            // let output = softmax_one(&context, output).await?;
            let output = output.to_vec();
            let token = sample(&output[..info.num_vocab], 0.0);
            prompt.batches[0].tokens.push(token);

            let decoded = tokenizer.decode(&[token])?;
            let word = String::from_utf8_lossy(&decoded);
            print!("{}", word);
            std::io::stdout().flush().unwrap();

            match token {
                0 | 59 => break,
                _ => {}
            }
        }
    }

    Ok(())
}
