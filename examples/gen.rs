use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use memmap2::Mmap;
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::PathBuf,
    time::Instant,
};
use web_rwkv::{Environment, Model, Quantization, Tokenizer};

fn softmax(data: Vec<f32>) -> Vec<f32> {
    let exp = data.iter().copied().map(f32::exp).collect_vec();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

fn sample(probs: Vec<f32>, top_p: f32) -> u16 {
    let sorted = probs
        .into_iter()
        .enumerate()
        .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(&y).reverse())
        .scan((0, 0.0), |(_, cum), (id, x)| {
            if *cum > top_p {
                None
            } else {
                *cum += x;
                Some((id, *cum))
            }
        })
        .collect_vec();
    let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
    let sorted = sorted.into_iter().map(|(id, x)| (id, x / sum));

    let rand = fastrand::f32();
    let token = sorted
        .into_iter()
        .find_or_first(|&(_, cum)| rand <= cum)
        .map(|(id, _)| id)
        .unwrap_or_default();
    token as u16
}

async fn create_environment() -> Result<Environment> {
    let env = Environment::create().await?;
    println!("{:#?}", env.adapter.get_info());
    Ok(env)
}

fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/rwkv_vocab_v20230424.json")?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(Tokenizer::new(&contents)?)
}

fn load_model(env: &Environment, model: PathBuf, quant: Option<usize>) -> Result<Model> {
    let file = File::open(model)?;
    let map = unsafe { Mmap::map(&file)? };
    let quantization = quant
        .map(|layer| Quantization::Int8((0..layer).collect()))
        .unwrap_or_default();
    let model = env.create_model_from_bytes(&map, &quantization)?;
    println!("{:#?}", model.info());
    Ok(model)
}

async fn run(cli: Cli) -> Result<()> {
    let env = create_environment().await?;

    let tokenizer = load_tokenizer()?;
    let model = cli
        .model
        .unwrap_or("assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st".into());
    let model = load_model(&env, model, cli.quant)?;

    let prompt = "The Eiffel Tower is located in the city of";
    let mut tokens = tokenizer.encode(prompt.as_bytes())?;
    print!("{}", prompt);

    let state = model.create_state();

    let mut start = Instant::now();
    let num_tokens = 100;
    for index in 0..=num_tokens {
        let logits = model.run(&tokens, &state)?;
        let probs = softmax(logits);
        let token = sample(probs, 0.5);
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
    std::io::stdout().flush()?;

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: Option<PathBuf>,
    #[arg(short, long, value_name = "LAYER")]
    quant: Option<usize>,
}

fn main() {
    let cli = Cli::parse();
    pollster::block_on(run(cli)).unwrap();
}
