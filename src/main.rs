use anyhow::Result;
use itertools::Itertools;
use std::{
    fs::File,
    io::{BufReader, Read},
    sync::Arc,
    time::Instant,
};

use crate::{environment::Environment, model::Model, tokenizer::Tokenizer};

mod environment;
mod model;
mod tokenizer;

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

    let rand = fastrand::f32() * top_p;
    let (token, _) = sorted
        .iter()
        .find_or_first(|&&(_, cum)| rand < cum)
        .unwrap();

    *token as u16
}

async fn run() -> Result<()> {
    let tokenizer = {
        let file = File::open("assets/rwkv_vocab_v20230424.json")?;
        let mut reader = BufReader::new(file);
        let mut contents = String::new();
        reader.read_to_string(&mut contents)?;
        Tokenizer::new(&contents)
    }?;

    let env = Arc::new(Environment::create().await?);
    let model = Model::from_file(
        "assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st".into(),
        env,
    )?;
    println!("{:#?}", model.info);

    let prompt = "The Eiffel Tower is located in the city of";
    let mut tokens = tokenizer.encode(prompt.as_bytes())?;
    println!("{:?}", tokens);
    print!("{}", prompt);

    let state = model.create_state();

    let start = Instant::now();
    let num_tokens = 100;
    for _ in 0..num_tokens {
        let buffer = model.create_buffer(&tokens);

        model.queue(&buffer, &state);

        let logits = model.read_back(&buffer);
        let probs = softmax(&logits);

        let token = sample(&probs, 0.5);
        let word = String::from_utf8(tokenizer.decode(&[token])?)?;
        print!("{}", word);

        tokens = vec![token];
    }

    println!(
        "\n{} tokens: {} mills",
        num_tokens,
        start.elapsed().as_millis()
    );

    Ok(())
}

fn main() {
    pollster::block_on(run()).unwrap();
}
