//! This example shows how to read-back and load model states to archive
//! session management (e.g., retrying) in a conversational application.

use std::{io::Write, path::PathBuf};

use anyhow::Result;
use clap::{Args, Parser, ValueEnum};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::{Loader, Lora},
        model::{Bundle, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant, State},
        softmax::softmax_one,
        v4, v5, v6, v7, Runtime, TokioRuntime,
    },
    tensor::{TensorCpu, TensorInit, TensorShape},
    tokenizer::Tokenizer,
};

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

async fn load_prompt(path: Option<PathBuf>) -> Result<Prompt> {
    match path {
        Some(path) => {
            let file = File::open(path).await?;
            let mut reader = BufReader::new(file);
            let mut contents = String::new();
            reader.read_to_string(&mut contents).await?;
            Ok(serde_json::from_str(&contents)?)
        }
        None => Ok(Prompt {
            user: String::from("User"),
            bot: String::from("Assistant"),
            intro: String::new(),
            text: vec![
                [
                    String::from("Hi!"),
                    String::from("Hello! I'm your AI assistant. I'm here to help you with various tasks, such as answering questions, brainstorming ideas, drafting emails, writing code, providing advice, and much more.")
                ]
            ],
        }),
    }
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
    #[arg(short, long, action)]
    turbo: bool,
    #[arg(short, long)]
    embed_device: Option<EmbedDevice>,
    #[arg(long, default_value_t = 128)]
    token_chunk_size: usize,
    #[arg(short, long, action)]
    adapter: bool,
    #[arg(short, long, value_name = "FILE")]
    prompt: Option<PathBuf>,
    #[command(flatten)]
    sampler: Sampler,
}

#[derive(Debug, Serialize, Deserialize)]
struct Prompt {
    user: String,
    bot: String,
    intro: String,
    text: Vec<[String; 2]>,
}

impl Prompt {
    fn build(&self) -> String {
        let user = self.user.trim();
        let bot = self.bot.trim();
        let intro = self.intro.trim();
        let text = self
            .text
            .iter()
            .map(|turn| {
                let user_text = turn[0].trim();
                let bot_text = turn[1].trim();
                format!("{user}: {user_text}\n\n{bot}: {bot_text}\n\n")
            })
            .join("");
        format!("{intro}\n\n{text}")
            .replace("{user}", user)
            .replace("{bot}", bot)
    }
}

#[derive(Debug, Clone, Args)]
struct Sampler {
    #[arg(long, default_value_t = 0.5)]
    top_p: f32,
    #[arg(long, default_value_t = 1.0)]
    temp: f32,
}

impl Sampler {
    pub fn sample(&self, probs: &[f32]) -> u16 {
        let sorted: Vec<_> = probs
            .iter()
            .copied()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > self.top_p {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, x))
                }
            })
            .map(|(id, _, x)| (id, x.powf(1.0 / self.temp)))
            .collect();

        let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
        let sorted: Vec<_> = sorted
            .into_iter()
            .map(|(id, x)| (id, x / sum))
            .scan((0, 0.0), |(_, cum), (id, x)| {
                *cum += x;
                Some((id, *cum))
            })
            .collect();

        let rand = fastrand::f32();
        let token = sorted
            .into_iter()
            .find_or_first(|&(_, cum)| rand <= cum)
            .map(|(id, _)| id)
            .unwrap_or_default();
        token as u16
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("chat", log::LevelFilter::Info)
        .init()?;
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

    let (runtime, state): (Box<dyn Runtime<Rnn>>, Box<dyn State>) = match info.version {
        ModelVersion::V4 => {
            let model = builder.build_v4().await?;
            let bundle = v4::Bundle::<f16>::new(model, 1);
            let state = bundle.state();
            let runtime = TokioRuntime::new(bundle).await;
            (Box::new(runtime), Box::new(state))
        }
        ModelVersion::V5 => {
            let model = builder.build_v5().await?;
            let bundle = v5::Bundle::<f16>::new(model, 1);
            let state = bundle.state();
            let runtime = TokioRuntime::new(bundle).await;
            (Box::new(runtime), Box::new(state))
        }
        ModelVersion::V6 => {
            let model = builder.build_v6().await?;
            let bundle = v6::Bundle::<f16>::new(model, 1);
            let state = bundle.state();
            let runtime = TokioRuntime::new(bundle).await;
            (Box::new(runtime), Box::new(state))
        }
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, 1);
            let state = bundle.state();
            let runtime = TokioRuntime::new(bundle).await;
            (Box::new(runtime), Box::new(state))
        }
    };

    println!("\n\nInstructions:\n\n+: Alternative reply\n-: Exit chatting\n\n------------");

    // run initial prompt
    let prompt = load_prompt(cli.prompt).await?;
    let mut inference = RnnInput::new(
        vec![RnnInputBatch {
            tokens: tokenizer.encode(prompt.build().as_bytes())?,
            option: RnnOption::Last,
        }],
        cli.token_chunk_size,
    );

    loop {
        let input = inference.clone();
        let (input, output) = runtime.infer(input).await?;
        inference = input;

        if output[0].size() > 0 {
            assert_eq!(inference.batches[0].tokens.len(), 0);
            break;
        }
    }

    print!("{}", prompt.build());
    std::io::stdout().flush()?;

    // read back initial state
    let mut backed = state.back(0).await?;
    let mut last_user_text = String::from("Hi!");
    let mut last_tokens = vec![];

    // main conversation loop
    loop {
        let mut model_text = String::new();
        let mut user_text = String::new();

        print!("{}: ", prompt.user);
        std::io::stdout().flush()?;

        while user_text.is_empty() {
            std::io::stdin().read_line(&mut user_text)?;
            user_text = user_text.trim().into();
        }

        match user_text.as_str() {
            "-" => break,
            "+" => {
                // retry: reset the prompt and state to the last turn
                user_text.clone_from(&last_user_text);
                inference.batches[0] = RnnInputBatch {
                    tokens: last_tokens.clone(),
                    option: RnnOption::Last,
                };
                state.load(backed.clone(), 0)?;
            }
            _ => {
                last_user_text.clone_from(&user_text);
                last_tokens.clone_from(&inference.batches[0].tokens);
                backed = state.back(0).await?;
            }
        }

        print!("\n{}:", prompt.bot);
        std::io::stdout().flush()?;

        let prompt = format!("{}: {}\n\n{}:", prompt.user, user_text, prompt.bot);
        inference.batches[0]
            .tokens
            .append(&mut tokenizer.encode(prompt.as_bytes())?);

        // inference loop: read the user prompt and generate until the stop token "\n\n"
        loop {
            let input = inference.clone();
            let (input, output) = runtime.infer(input).await?;
            inference = input;

            let output = output[0].0.clone();
            let shape = output.shape();
            if output.size() == 0 {
                // we are not finishing reading the prompt
                continue;
            }

            let output = output.to_vec();
            assert_eq!(output.len(), info.num_vocab_padded());

            let output = TensorCpu::from_data(shape, output)?;
            let output = softmax_one(&context, output).await?;

            let token = cli.sampler.sample(&output);
            let decoded = tokenizer.decode(&[token])?;
            let word = String::from_utf8_lossy(&decoded);

            model_text += &word;
            print!("{}", word);
            std::io::stdout().flush()?;

            inference.batches[0] = RnnInputBatch {
                tokens: vec![token],
                option: RnnOption::Last,
            };

            if model_text.contains("\n\n") {
                break;
            }
        }
    }

    Ok(())
}
