use anyhow::Result;
use clap::Parser;
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use itertools::Itertools;
use memmap2::Mmap;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{LayerFlags, Model, ModelBuilder, ModelState, Quantization},
    tensor::{shape::Shape, TensorCpu, TensorExt},
    tokenizer::Tokenizer,
};

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

fn load_model(context: &Context, model: PathBuf, quant: Option<u64>) -> Result<Model<'_, '_>> {
    let file = File::open(model)?;
    let map = unsafe { Mmap::map(&file)? };
    let quant = quant
        .map(|bits| Quantization::Int8(LayerFlags::from_bits_retain(bits)))
        .unwrap_or_default();
    let model = ModelBuilder::new(context, &map).with_quant(quant).build()?;
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

    let prompt = "The Eiffel Tower is located in the city of";
    let tokens = tokenizer.encode(prompt.as_bytes())?;
    print!("{}", prompt);

    let state = ModelState::new(&context, model.info(), cli.batch);
    let mask = context.tensor_from_data(Shape::new(1, 1, 1), vec![u32::MAX])?;

    let shape = model.input_shape(tokens.len(), 1);
    let tokens = TensorCpu::from_data(&context, shape, tokens)?.repeat(0, cli.batch);
    let logits = model.run(&tokens, &mask, &state)?;

    let probs = model.softmax(&logits)?.split();
    for probs in probs {
        let token = sample(probs.to_vec(), 0.5);
        let word = String::from_utf8(tokenizer.decode(&[token])?)?;
        print!("{}", word);
    }

    // let mut start = Instant::now();
    // let num_tokens = 100;
    // for index in 0..=num_tokens {
    //     let shape = model.input_shape(tokens.len(), cli.batch);
    //     let logits = if index == 0 {
    //         let tokens = TensorCpu::from_data(&context, shape, tokens)?.repeat(0, cli.batch);
    //         model.run(&tokens, &mask, &state)?
    //     } else {
    //         model.run(&context.tensor_from_data(shape, tokens)?, &mask, &state)?
    //     };

    //     let probs = model.softmax(&logits)?.to_vec();
    //     let token = sample(probs, 0.5);
    //     let word = String::from_utf8(tokenizer.decode(&[token])?)?;
    //     print!("{}", word);

    //     tokens = vec![token];

    //     if index == 0 {
    //         start = Instant::now();
    //     }
    // }

    // println!(
    //     "\n{} tokens: {} mills",
    //     num_tokens,
    //     start.elapsed().as_millis()
    // );
    // std::io::stdout().flush()?;

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: Option<PathBuf>,
    #[arg(short, long, value_name = "LAYERS")]
    quant: Option<u64>,
    #[arg(short, long, default_value_t = 4)]
    batch: usize,
}

fn main() {
    let cli = Cli::parse();
    pollster::block_on(run(cli)).unwrap();
}
