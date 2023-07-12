use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use memmap2::Mmap;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    time::Instant,
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

async fn run(model: PathBuf) -> Result<()> {
    let env = create_environment().await?;

    let tokenizer = load_tokenizer().await?;
    let model = load_model(&env, model).await?;

    let prompt = "The Eiffel Tower is located in the city of";
    let mut tokens = tokenizer.encode(prompt.as_bytes())?;
    print!("{}", prompt);

    let state = model.create_state();

    let mut start = Instant::now();
    let num_tokens = 100;
    for index in 0..=num_tokens {
        let logits = model.run(&tokens, &state)?;

        let probs = softmax(&logits);

        let token = sample(&probs, 0.5);
        let word = String::from_utf8(tokenizer.decode(&[token])?)?;
        print!("{}", word);

        tokens = vec![token];

        if index == 0 {
            start = Instant::now();
        }
    }

    println!(
        "\n{} tokens: {} mills",
        num_tokens,
        start.elapsed().as_millis()
    );

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
