use anyhow::Result;
use clap::Parser;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::path::PathBuf;
use tokio::fs::File;
use web_rwkv::runtime::loader::Loader;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the SafeTensors model file.
    #[arg(short, long)]
    model: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse the command-line argument to get the model path.
    let cli = Cli::parse();
    println!("Attempting to load model from: {:?}\n", cli.model);

    // 2. Open and memory-map the model file for efficient access.
    let file = File::open(cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };

    // 3. Deserialize the SafeTensors data.
    let model_tensors = SafeTensors::deserialize(&data)?;
    println!("Successfully deserialized the SafeTensors file.\n");

    // 4. Use the web-rwkv Loader to extract the ModelInfo.
    let info = Loader::info(&model_tensors)?;
    println!("--- Extracted ModelInfo ---");

    // 5. Pretty-print the info struct to the console.
    println!("{:#?}", info);
    println!("---------------------------");

    Ok(())
}
