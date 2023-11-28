use anyhow::Result;
use clap::{Args, Parser};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use itertools::Itertools;
use memmap2::Mmap;
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read, Write},
    path::PathBuf,
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{
        loader::Loader, v4, v5, v6, FromBuilder, Lora, Model, ModelBuilder, ModelState,
        ModelVersion, Quant, StateBuilder,
    },
    tokenizer::Tokenizer,
};

#[derive(Debug, Clone, Args)]
struct Sampler {
    #[arg(long, default_value_t = 0.5)]
    top_p: f32,
    #[arg(long, default_value_t = 1.0)]
    temp: f32,
    #[arg(long, default_value_t = 0.3)]
    presence_penalty: f32,
    #[arg(long, default_value_t = 0.3)]
    frequency_penalty: f32,
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

async fn create_context() -> Result<Context> {
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
        .with_default_pipelines()
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

fn load_model<'a, M>(
    context: &Context,
    data: &'a [u8],
    lora: Option<PathBuf>,
    quant: Option<usize>,
    quant_nf4: Option<usize>,
    turbo: bool,
) -> Result<M>
where
    M: Model + FromBuilder<Builder<'a> = ModelBuilder<'a>, Error = anyhow::Error>,
{
    let quant = quant
        .map(|layer| (0..layer).map(|layer| (layer, Quant::Int8)).collect_vec())
        .unwrap_or_default();
    let quant_nf4 = quant_nf4
        .map(|layer| (0..layer).map(|layer| (layer, Quant::NF4)).collect_vec())
        .unwrap_or_default();
    let quant = quant.into_iter().chain(quant_nf4.into_iter()).collect();
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

fn load_prompt(path: Option<PathBuf>) -> Result<Prompt> {
    match path {
        Some(path) => {
            let file = File::open(path)?;
            let mut reader = BufReader::new(file);
            let mut contents = String::new();
            reader.read_to_string(&mut contents)?;
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

async fn run(cli: Cli) -> Result<()> {
    let context = create_context().await?;
    let tokenizer = load_tokenizer()?;

    let model = cli.model.unwrap_or(
        std::fs::read_dir("assets/models")
            .unwrap()
            .filter_map(|x| x.ok())
            .find(|x| x.path().extension().is_some_and(|x| x == "st"))
            .unwrap()
            .path(),
    );
    let prompt = load_prompt(cli.prompt)?;
    let sampler = cli.sampler;

    let file = File::open(model)?;
    let map = unsafe { Mmap::map(&file)? };

    let info = Loader::info(&map)?;
    println!("{:#?}", info);

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
            run_internal(model, state, tokenizer, prompt, sampler).await
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
            run_internal(model, state, tokenizer, prompt, sampler).await
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
            run_internal(model, state, tokenizer, prompt, sampler).await
        }
    }
}

async fn run_internal<M, S>(
    model: M,
    state: S,
    tokenizer: Tokenizer,
    prompt: Prompt,
    sampler: Sampler,
) -> Result<()>
where
    S: ModelState,
    M: Model<ModelState = S>,
{
    let user = &prompt.user;
    let bot = &prompt.bot;
    let prompt = prompt.build();

    let mut tokens = vec![tokenizer.encode(prompt.as_bytes())?];

    println!("\n\nInstructions:\n\n+: Alternative reply\n-: Exit chatting\n\n------------");
    print!("{}", prompt);
    std::io::stdout().flush()?;

    // run initial prompt
    loop {
        let logits = model.run(&mut tokens, &state).await?;
        if logits.iter().any(Option::is_some) {
            break;
        }
    }
    tokens[0].clear();

    let mut backed = state.back().await;
    let mut last_user_text = String::from("Hi!");
    let mut last_tokens = vec![];

    loop {
        let mut model_text = String::new();
        let mut user_text = String::new();
        let mut occurrences = HashMap::new();

        print!("{}: ", user);
        std::io::stdout().flush()?;

        while user_text.is_empty() {
            std::io::stdin().read_line(&mut user_text)?;
            user_text = user_text.trim().into();
        }

        if &user_text == "-" {
            break;
        } else if &user_text == "+" {
            state.load(&backed)?;
            user_text = last_user_text.clone();
            tokens = last_tokens.clone();
        } else {
            backed = state.back().await;
            last_user_text = user_text.clone();
            last_tokens = tokens.clone();
        }

        print!("\n{}:", bot);
        std::io::stdout().flush()?;

        let prompt = format!("{user}: {user_text}\n\n{bot}:");
        tokens[0].append(&mut tokenizer.encode(prompt.as_bytes())?);

        loop {
            let mut logits = loop {
                let logits = model.run(&mut tokens, &state).await?;
                if logits.iter().any(Option::is_some) {
                    break logits;
                }
            };
            logits.iter_mut().for_each(|logits| {
                if let Some(logits) = logits {
                    logits[0] = f32::NEG_INFINITY;
                    for (&token, &count) in occurrences.iter() {
                        let penalty =
                            sampler.presence_penalty + count as f32 * sampler.frequency_penalty;
                        logits[token as usize] -= penalty;
                    }
                }
            });

            let probs = model.softmax(logits).await?;
            if let Some(probs) = &probs[0] {
                let token = sampler.sample(probs);
                let decoded = tokenizer.decode(&[token])?;
                let word = String::from_utf8_lossy(&decoded);

                model_text += &word;
                print!("{}", word);
                std::io::stdout().flush()?;

                tokens[0] = vec![token];
                let count = occurrences.get(&token).unwrap_or(&1);
                occurrences.insert(token, *count);

                if token == 0 || model_text.contains("\n\n") {
                    break;
                }
            }
        }
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: Option<PathBuf>,
    #[arg(short, long, value_name = "FILE")]
    lora: Option<PathBuf>,
    #[arg(short, long, value_name = "FILE")]
    prompt: Option<PathBuf>,
    #[arg(short, long, value_name = "LAYERS")]
    quant: Option<usize>,
    #[arg(long, value_name = "LAYERS")]
    quant_nf4: Option<usize>,
    #[arg(short, long, action)]
    turbo: bool,
    #[command(flatten)]
    sampler: Sampler,
}

#[derive(Debug, Deserialize)]
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

fn main() {
    let cli = Cli::parse();
    pollster::block_on(run(cli)).unwrap();
}
