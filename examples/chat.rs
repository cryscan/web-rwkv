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

fn softmax(data: &[f32]) -> Vec<f32> {
    let exp = data.iter().copied().map(f32::exp).collect_vec();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect_vec()
}

fn sample(probs: &[f32], top_p: f32) -> u16 {
    let sorted = probs
        .iter()
        .copied()
        .enumerate()
        .sorted_by(|a, b| a.1.total_cmp(&b.1).reverse())
        .scan((0usize, 0.0f32), |state, x| {
            if state.1 > top_p {
                return None;
            }
            Some((x.0, state.1 + x.1))
        })
        .collect_vec();

    let rand = fastrand::f32() * sorted.last().unwrap().1;
    let (token, _) = sorted
        .iter()
        .find_or_first(|&&(_, cum)| rand <= cum)
        .unwrap();

    *token as u16
}

async fn create_environment() -> Result<Environment> {
    let env = Environment::create().await?;
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

async fn run(model: PathBuf) -> Result<()> {
    let env = create_environment().await?;

    let tokenizer = load_tokenizer().await?;
    let model = load_model(&env, model).await?;

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

        print!("{}: ", user);
        std::io::stdout().flush()?;

        while user_text.is_empty() {
            std::io::stdin().read_line(&mut user_text)?;
            user_text = user_text.trim().into();
        }

        if &user_text.to_lowercase() == "exit" {
            break;
        }

        print!("{}:", bot);
        std::io::stdout().flush()?;

        let prompt = format!("{user}: {user_text}\n\n{bot}:");
        tokens.append(&mut tokenizer.encode(prompt.as_bytes())?);

        while !model_text.contains("\n\n") {
            let mut logits = model.run(&tokens, &state)?;
            logits[0] = f32::NEG_INFINITY;

            let probs = softmax(&logits);

            let token = sample(&probs, 0.5);
            let word = String::from_utf8(tokenizer.decode(&[token])?)?;

            model_text += &word;
            print!("{}", word);
            std::io::stdout().flush()?;

            tokens = vec![token];
        }
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, value_name = "FILE")]
    model: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let model = args
        .model
        .unwrap_or("assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st".into());

    pollster::block_on(run(model)).unwrap();
}
