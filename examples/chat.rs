use ahash::{HashMap, HashMapExt};
use anyhow::Result;
use clap::{Args, Parser};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use itertools::Itertools;
use memmap2::Mmap;
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::PathBuf,
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{LayerFlags, Model, ModelBuilder, ModelState, Quantization},
    tensor::Shape,
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
    pub fn sample(&self, probs: Vec<f32>) -> u16 {
        let sorted: Vec<_> = probs
            .into_iter()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(&y).reverse())
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
        let adapters = instance.adapters();
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Please select an adapter")
            .default(0)
            .items(&adapters)
            .interact()?;
        instance.select_adapter(selection)?
    };
    #[cfg(debug_assertions)]
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .with_default_pipelines()
        .with_quant_pipelines()
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

fn load_model<'a>(
    context: &'a Context,
    model: PathBuf,
    quant: Option<u64>,
) -> Result<Model<'a, '_>> {
    let file = File::open(model)?;
    let map = unsafe { Mmap::map(&file)? };
    let quant = quant
        .map(|bits| Quantization::Int8(LayerFlags::from_bits_retain(bits)))
        .unwrap_or_default();
    let model = ModelBuilder::new(&context, &map)
        .with_quant(quant)
        .build()?;
    println!("{:#?}", model.info());
    Ok(model)
}

async fn run(cli: Cli) -> Result<()> {
    let context = create_context().await?;
    let tokenizer = load_tokenizer()?;

    let model = cli
        .model
        .unwrap_or("assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st".into());
    let model = load_model(&context, model, cli.quant)?;
    let sampler = cli.sampler;

    let user = "User";
    let bot = "Assistant";
    let prompt = format!("\n\n{user}: Hi!\n\n{bot}: Hello! I'm an AI assistant trained by Peng Bo! I'm here to help you with various tasks, such as answering questions, brainstorming ideas, drafting emails, writing code, providing advice, and much more.\n\n");
    let mut tokens = tokenizer.encode(prompt.as_bytes())?;

    print!("\n\nInstructions:\n\n+: Alternative reply\n-: Exit chatting");
    print!("{}", prompt);
    std::io::stdout().flush()?;

    let mask = context.tensor_from_data(Shape::new(1, 1, 1), vec![u32::MAX])?;

    let state = ModelState::new(&context, model.info(), 1);
    let shape = model.input_shape(tokens.len(), 1);
    let _ = model.run(
        &context.tensor_from_data(shape, tokens.clone())?,
        &mask,
        &state,
    )?;
    tokens.clear();

    let mut backed_state = state.clone().into();
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
            state.load(&backed_state)?;
            user_text = last_user_text.clone();
            tokens = last_tokens.clone();
        } else {
            backed_state = state.clone().into();
            last_user_text = user_text.clone();
            last_tokens = tokens.clone();
        }

        print!("\n{}:", bot);
        std::io::stdout().flush()?;

        let prompt = format!("{user}: {user_text}\n\n{bot}:");
        tokens.append(&mut tokenizer.encode(prompt.as_bytes())?);

        loop {
            let shape = model.input_shape(tokens.len(), 1);
            let mut logits = model
                .run(
                    &context.tensor_from_data(shape, tokens.clone())?,
                    &mask,
                    &state,
                )?
                .to_vec();
            logits[0] = f32::NEG_INFINITY;
            for (&token, &count) in occurrences.iter() {
                let penalty = sampler.presence_penalty + count as f32 * sampler.frequency_penalty;
                logits[token as usize] -= penalty;
            }

            let shape = model.head_shape(1);
            let probs = model
                .softmax(&context.tensor_from_data(shape, logits.clone())?)?
                .to_vec();
            let token = sampler.sample(probs);
            let word = String::from_utf8(tokenizer.decode(&[token])?)?;

            model_text += &word;
            print!("{}", word);
            std::io::stdout().flush()?;

            tokens = vec![token];
            let count = occurrences.get(&token).unwrap_or(&1);
            occurrences.insert(token, *count);

            if token == 0 || model_text.contains("\n\n") {
                break;
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
    #[arg(short, long, value_name = "LAYERS")]
    quant: Option<u64>,
    #[command(flatten)]
    sampler: Sampler,
}

fn main() {
    let cli = Cli::parse();
    pollster::block_on(run(cli)).unwrap();
}
