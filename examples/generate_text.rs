use anyhow::Result;
use clap::Parser;
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::io::Write;
use std::path::PathBuf;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime:{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        softmax::softmax_one,
        v4, v5, v6, v7, Runtime, TokioRuntime,
    },
    tokenizer::Tokenizer,
};

const MAX_TOKENS: usize = 1024;
const STOP_TOKEN: u32 = 8192;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the SafeTensors model file.
    #[arg(long)]
    model: PathBuf,
    /// Path to the Tokenizer JSON file.
    #[arg(long)]
    tokenizer: PathBuf,
    /// The prompt to start generation from.
    #[arg(long)]
    prompt: String,
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

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

async fn load_tokenizer(path: PathBuf) -> Result<Tokenizer> {
    let content = tokio::fs::read_to_string(path).await?;
    Ok(Tokenizer::new(&content)?)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load the tokenizer
    println!("[1/4] Loading tokenizer from {:?}...", cli.tokenizer);
    let tokenizer = load_tokenizer(cli.tokenizer).await?;

    // Load the model
    println!("[2/4] Loading model from {:?}...", cli.model);
    let file = File::open(cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model_tensors = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model_tensors)?;
    println!("Model info: {:#?}", info);

    // Create the context and runtime
    println!("\n[3/4] Initializing runtime and loading model weights...");
    let context = create_context(&info).await?;
    let builder = ModelBuilder::new(&context, model_tensors);

    let runtime: Box<dyn Runtime<Rnn>> = match info.version {
        ModelVersion::V4 => {
            let model = builder.build_v4().await?;
            let bundle = v4::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V5 => {
            let model = builder.build_v5().await?;
            let bundle = v5::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V6 => {
            let model = builder.build_v6().await?;
            let bundle = v6::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
    };

    // Encode the prompt and prepare for inference
    println!("\n[4/4] Encoding prompt and starting generation...");
    let prompt_tokens = tokenizer.encode(cli.prompt.as_bytes())?;
    let mut inference = RnnInput::new(
        vec![RnnInputBatch::new(prompt_tokens, RnnOption::Last)],
        128, // token_chunk_size
    );

    // Print the prompt
    print!("\n--- PROMPT ---\n{}\n--- OUTPUT ---\n", cli.prompt);
    std::io::stdout().flush()?;

    // Main generation loop
    for token_count in 0..MAX_TOKENS {
        let (input, output) = runtime.infer(inference).await?;
        inference = input;

        // If there is no output, it means we are still processing the prompt.
        if output.is_empty() || output[0].is_empty() {
            continue;
        }

        // Extract logits, run softmax, and sample the next token
        let logits = output[0].0.clone();
        let probs = softmax_one(&context, logits).await?;
        let token = sample(&probs.to_vec());

        // Check for stop token
        if token == STOP_TOKEN {
            println!("\n\n[INFO] Stop token ({}) reached.", STOP_TOKEN);
            break;
        }

        // Print the generated token ID directly
        print!("{} ", token);
        std::io::stdout().flush()?;

        // Feed the new token back into the model
        inference.batches[0].replace(vec![token]);

        if token_count == MAX_TOKENS - 1 {
            println!("\n\n[INFO] Max tokens ({}) reached.", MAX_TOKENS);
        }
    }

    println!("\n\n[INFO] Generation finished.");
    Ok(())
}
