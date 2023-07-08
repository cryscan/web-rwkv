use anyhow::Result;
use std::{
    fs::File,
    io::{BufReader, Read},
    sync::Arc,
};

use crate::{environment::Environment, model::Model, tokenizer::Tokenizer};

mod environment;
mod model;
mod tokenizer;

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
    let tokens = tokenizer.encode(prompt.as_bytes())?;
    println!("{:?}", tokens);

    let buffer = model.create_buffer(&tokens);
    let state = model.create_state();
    model.queue(&buffer, &state);

    let logits = model.read_back(&buffer);
    println!("{:?}", &logits[0..32]);
    println!("{:?}", &logits[992..1024]);

    let token = logits
        .into_iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(id, _)| id)
        .unwrap() as u16;
    let word = String::from_utf8(tokenizer.decode(&[token])?)?;
    println!("{word}");

    Ok(())
}

fn main() {
    pollster::block_on(run()).unwrap();
}
