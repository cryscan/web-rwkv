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

    let prompt = "Hello, my name is";
    let tokens = tokenizer.encode(prompt.as_bytes())?;
    println!("{:?}", tokens);

    let buffer = model.create_buffer(&tokens);
    let state = model.create_state();
    let bind_group = model.create_bind_group(&buffer, &state);

    Ok(())
}

fn main() {
    pollster::block_on(run()).unwrap();
}
