use ahash::{HashMap, HashMapExt};
use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use memmap2::Mmap;
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::PathBuf,
};
use web_rwkv::{Environment, Model, Tokenizer};

struct Sampler {
    pub top_p: f32,
    pub temp: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Sampler {
    pub fn sample(&self, mut logits: Vec<f32>, occurrences: &HashMap<u16, usize>) -> u16 {
        for (&token, &count) in occurrences.iter() {
            logits[token as usize] -= self.presence_penalty + count as f32 * self.frequency_penalty;
        }

        let probs = softmax(logits);
        let sorted: Vec<_> = probs
            .into_iter()
            .enumerate()
            .sorted_by(|(_, x), (_, y)| x.total_cmp(&y).reverse())
            .scan((0, 0.0, 0.0), |(_, acc, _), (id, x)| {
                (*acc <= self.top_p).then(|| (id, *acc + x, x))
            })
            .map(|(id, _, p)| (id, p.powf(1.0 / self.temp)))
            .scan((0, 0.0), |(_, acc), (id, x)| Some((id, *acc + x)))
            .collect();

        let rand = fastrand::f32() * sorted.last().unwrap_or(&(0, 0.0)).1;
        let (token, _) = sorted
            .into_iter()
            .find_or_first(|&(_, cum)| rand <= cum)
            .unwrap_or((0, 0.0));
        token as u16
    }
}

fn softmax(data: Vec<f32>) -> Vec<f32> {
    let exp: Vec<_> = data.into_iter().map(f32::exp).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

async fn create_environment() -> Result<Environment> {
    let env = Environment::create(wgpu::PowerPreference::HighPerformance).await?;
    println!("{:#?}", env.adapter.get_info());
    Ok(env)
}

async fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/rwkv_vocab_v20230424.json")?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(Tokenizer::new(&contents)?)
}

async fn load_model(env: &Environment, model: PathBuf) -> Result<Model> {
    let file = File::open(model)?;
    let map = unsafe { Mmap::map(&file)? };
    let model = env.create_model_from_bytes(&map)?;
    println!("{:#?}", model.info());
    Ok(model)
}

async fn run(args: Args) -> Result<()> {
    let env = create_environment().await?;
    let tokenizer = load_tokenizer().await?;

    let model = args.model;
    let model = load_model(&env, model).await?;

    let sampler = Sampler {
        top_p: args.top_p,
        temp: args.temp,
        presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty,
    };

    let user = "User";
    let bot = "Assistant";
    let prompt = format!("\n\n{user}: Hi!\n\n{bot}: Hello! I'm an AI assistant trained by Peng Bo! I'm here to help you with various tasks, such as answering questions, brainstorming ideas, drafting emails, writing code, providing advice, and much more.\n\n");
    let mut tokens = tokenizer.encode(prompt.as_bytes())?;

    print!("{}", prompt);
    std::io::stdout().flush()?;

    let state = model.create_state();
    let _ = model.run(&tokens, &state);
    tokens.clear();

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

        if &user_text.to_lowercase() == "exit" {
            break;
        }

        print!("\n{}:", bot);
        std::io::stdout().flush()?;

        let prompt = format!("{user}: {user_text}\n\n{bot}:");
        tokens.append(&mut tokenizer.encode(prompt.as_bytes())?);

        while !model_text.contains("\n\n") {
            let mut logits = model.run(&tokens, &state)?;
            logits[0] = f32::NEG_INFINITY;

            let token = sampler.sample(logits, &occurrences);
            let word = String::from_utf8(tokenizer.decode(&[token])?)?;

            model_text += &word;
            print!("{}", word);
            std::io::stdout().flush()?;

            tokens = vec![token];
            let count = occurrences.get(&token).unwrap_or(&1);
            occurrences.insert(token, *count);
        }
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,
    #[arg(long, default_value_t = 0.5)]
    top_p: f32,
    #[arg(long, default_value_t = 1.0)]
    temp: f32,
    #[arg(long, default_value_t = 0.3)]
    presence_penalty: f32,
    #[arg(long, default_value_t = 0.3)]
    frequency_penalty: f32,
}

fn main() {
    let args = Args::parse();
    pollster::block_on(run(args)).unwrap();
}
