use anyhow::Result;
use clap::Parser;
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::path::PathBuf;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ModelBuilder, ModelInfo, ModelVersion},
        v4, v5, v6, v7, Runtime, TokioRuntime,
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the SafeTensors model file.
    #[arg(long)]
    model: PathBuf,
    /// A sequence of token IDs to use as a prompt.
    #[arg(long, value_delimiter = ' ', num_args = 1..)]
    ids: Vec<u32>,
    /// Optional: The adapter index to use.
    #[arg(long)]
    adapter_index: Option<usize>,
}

/// A simple greedy sampler that picks the token with the highest probability.
fn sample(probs: &[f32]) -> u32 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .map(|(id, _)| id as u32)
        .unwrap_or(0)
}

async fn create_context(adapter_index: Option<usize>) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    
    let adapter = match adapter_index {
        Some(index) => adapters.into_iter().nth(index).ok_or_else(|| anyhow::anyhow!("Invalid adapter index"))?,
        None => {
            let names = adapters.iter().map(|adapter| adapter.get_info()).map(|info| format!("{} ({:?})", info.name, info.backend)).collect_vec();
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Please select a GPU adapter")
                .default(0)
                .items(&names)
                .interact()?;
            adapters.into_iter().nth(selection).unwrap()
        }
    };

    println!("\nUsing adapter: {:#?}", adapter.get_info());
    
    let limits = adapter.limits();
    let context = ContextBuilder::new(adapter).limits(limits).build().await?;
    Ok(context)
}


#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // 1. Load the model file.
    let file = File::open(&cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model_tensors = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model_tensors)?;

    // 2. Initialize the runtime.
    let context = create_context(cli.adapter_index).await?;
    let builder = ModelBuilder::new(&context, model_tensors);
    let runtime: Box<dyn Runtime<Rnn>> = match info.version {
        ModelVersion::V4 => {
            let model = builder.build_v4().await?;
            let bundle = v4::Bundle::<f32>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V5 => {
            let model = builder.build_v5().await?;
            let bundle = v5::Bundle::<f32>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V6 => {
            let model = builder.build_v6().await?;
            let bundle = v6::Bundle::<f32>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f32>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
    };

    // 3. Prepare the initial input.
    let mut inference = RnnInput::new(
        vec![RnnInputBatch::new(cli.ids, RnnOption::Last)],
        128, // token_chunk_size
    );

    println!("\n--- Generating 3 Tokens ---");
    let mut generated_count = 0;

    // 4. Main generation loop.
    loop {
        let (input, output) = runtime.infer(inference).await?;
        inference = input;

        // If there is no output, it means we are still processing the prompt.
        if output.is_empty() || output[0].is_empty() {
            continue;
        }

        generated_count += 1;

        let logits_tensor = output[0].0.clone();
        let logits_vec = logits_tensor.to_vec();

        // Print Top 5 logits for the current prediction.
        println!("\n--- Top 5 Logits (for token #{}) ---", generated_count);
        let mut sorted_logits: Vec<(u32, f32)> = logits_vec.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
        sorted_logits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (token_id, logit_value) in sorted_logits.iter().take(5) {
            println!("  Token ID: {:>5}, Logit: {:>8.4}", token_id, logit_value);
        }
        println!("-------------------------------------");

        let token = sample(&logits_vec);
        println!("Prediction #{}: {}", generated_count, token);

        // Check if we've generated enough tokens.
        if generated_count >= 3 {
            break;
        }

        // Feed the new token back into the model for the next iteration.
        inference.batches[0].replace(vec![token]);
    }

    println!("\n[INFO] Generation finished.");
    Ok(())
}