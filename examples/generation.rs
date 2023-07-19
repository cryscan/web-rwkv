use anyhow::Result;
use itertools::Itertools;
use std::{
    fs::File,
    io::{BufReader, Read},
    sync::Arc,
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
    let env = Environment::create().await?;
    println!("{:#?}", env.adapter.get_info());
    #[cfg(target_arch = "wasm32")]
    log::info!("{:#?}", env.adapter.get_info());
    Ok(env)
}

async fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/rwkv_vocab_v20230424.json")?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(Tokenizer::new(&contents)?)
}

async fn load_model(env: Arc<Environment>) -> Result<Model> {
    let model = Model::from_file(
        "assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st".into(),
        env,
    )?;
    println!("{:#?}", model.info);
    #[cfg(target_arch = "wasm32")]
    log::info!("{:#?}", model.info);
    Ok(model)
}

async fn run() -> Result<()> {
    let env = Arc::new(create_environment().await?);

    let tokenizer = load_tokenizer().await?;
    let model = load_model(env.clone()).await?;

    let prompt = "The Eiffel Tower is located in the city of";
    let mut tokens = tokenizer.encode(prompt.as_bytes())?;
    print!("{}", prompt);

    let state = model.create_state();

    let mut start = Instant::now();
    let num_tokens = 100;
    for index in 0..=num_tokens {
        let buffer = model.create_buffer(&tokens);

        #[cfg(not(target_arch = "wasm32"))]
        let logits = model.run(&buffer, &state)?;

        #[cfg(target_arch = "wasm32")]
        let logits = model.run_async(&buffer, &state).await?;

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

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(run()).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(async { run().await.unwrap() });
    }
}
