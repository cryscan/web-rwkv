use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{
        head::ModelHead,
        loader::Loader,
        run::{HookMap, ModelRun},
        softmax::ModelSoftmax,
        v5, Lora, Model, ModelBase, ModelBuilder, ModelInfo, ModelVersion, Quant, StateBuilder,
    },
    tensor::{
        ops::{TensorCommand, TensorOp, TensorPass},
        shape::Shape,
        ReadBack, ReadWrite, TensorError, TensorGpu, TensorShape,
    },
    tokenizer::Tokenizer,
};

#[derive(Debug, Clone)]
struct Buffer {
    ffn_x: TensorGpu<f32, ReadWrite>,
    half: TensorGpu<f16, ReadWrite>,
    out: TensorGpu<f32, ReadWrite>,
    map: TensorGpu<f32, ReadBack>,
}

impl Buffer {
    fn new(context: &Context, info: &ModelInfo) -> Self {
        Self {
            ffn_x: context.tensor_init(Shape::new(info.num_emb, info.num_layer, 1, 1)),
            half: context.tensor_init(Shape::new(info.num_emb, info.num_layer, 1, 1)),
            out: context.tensor_init(Shape::new(info.num_vocab, info.num_layer, 1, 1)),
            map: context.tensor_init(Shape::new(info.num_vocab, info.num_layer, 1, 1)),
        }
    }
}

fn sample(probs: &[f32], _top_p: f32) -> u16 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap()
        .0 as u16
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = Instance::new();
    #[cfg(not(debug_assertions))]
    let adapter = {
        let backends = wgpu::Backends::all();
        let adapters = instance
            .enumerate_adapters(backends)
            .map(|adapter| adapter.get_info())
            .map(|info| format!("{} ({:?})", info.name, info.backend))
            .collect_vec();
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Please select an adapter")
            .default(0)
            .items(&adapters)
            .interact()?;
        instance.select_adapter(backends, selection)?
    };
    #[cfg(debug_assertions)]
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .with_default_pipelines()
        .with_auto_limits(info)
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

fn load_model<M: Model>(
    context: &Context,
    data: &[u8],
    lora: Option<PathBuf>,
    quant: Option<usize>,
    quant_nf4: Option<usize>,
    embed_device: Option<EmbedDevice>,
    turbo: bool,
) -> Result<M> {
    let quant = quant
        .map(|layer| (0..layer).map(|layer| (layer, Quant::Int8)).collect_vec())
        .unwrap_or_default();
    let quant_nf4 = quant_nf4
        .map(|layer| (0..layer).map(|layer| (layer, Quant::NF4)).collect_vec())
        .unwrap_or_default();
    let quant = quant.into_iter().chain(quant_nf4).collect();
    let model = ModelBuilder::new(context, data)
        .with_quant(quant)
        .with_turbo(turbo)
        .with_embed_device(embed_device.unwrap_or_default().into());
    match lora {
        Some(lora) => {
            let file = File::open(lora)?;
            let map = unsafe { Mmap::map(&file)? };
            model
                .add_lora(Lora {
                    data: map.to_vec(),
                    blend: Default::default(),
                })
                .build()
        }
        None => model.build(),
    }
}

async fn run(cli: Cli) -> Result<()> {
    let tokenizer = load_tokenizer()?;
    let model = cli.model.unwrap_or(
        std::fs::read_dir("assets/models")
            .unwrap()
            .filter_map(|x| x.ok())
            .find(|x| x.path().extension().is_some_and(|x| x == "st"))
            .unwrap()
            .path(),
    );

    let file = File::open(model)?;
    let map = unsafe { Mmap::map(&file)? };

    let info = Loader::info(&map)?;
    if info.version != ModelVersion::V5 {
        bail!("this demo only supports v5");
    }
    println!("{:#?}", info);

    let context = create_context(&info).await?;
    let model: v5::Model = load_model(
        &context,
        &map,
        cli.lora,
        cli.quant,
        cli.quant_nf4,
        cli.embed_device,
        cli.turbo,
    )?;
    let state: v5::ModelState = StateBuilder::new(&context, model.info()).build();

    // create a buffer to store each layer's output
    let buffer = Buffer::new(&context, &info);

    let mut hooks = HookMap::default();
    for layer in 0..info.num_layer {
        let buffer = buffer.clone();
        hooks.insert(
            v5::Hook::PostFfn(layer),
            Box::new(
                move |_model, _state, runtime: &v5::Runtime| -> Result<TensorOp, TensorError> {
                    // figure out how many tokens this run has
                    let shape = runtime.ffn_x.shape();
                    let num_token = shape[1];

                    // "steal" the layer's output (activation), and put it into our buffer
                    TensorOp::blit(
                        runtime.ffn_x.view(.., num_token - 1, .., ..)?,
                        buffer.ffn_x.view(.., layer, .., ..)?,
                    )
                },
            ),
        );
    }

    let prompt = cli
        .prompt
        .unwrap_or("The Space Needle is located in downtown".into());
    if prompt.is_empty() {
        bail!("prompt must not be empty")
    }

    let mut tokens = vec![tokenizer.encode(prompt.as_bytes())?];
    println!("Prompt: {}", prompt);

    // run initial prompt
    let logits = loop {
        let logits = model.run_with_hooks(&mut tokens, &state, &hooks).await?;
        if logits.iter().any(Option::is_some) {
            break logits;
        }
    };
    let probs = model.softmax(logits).await?;

    if let Some(probs) = &probs[0] {
        let token = sample(probs, 0.5);
        let word = tokenizer.decode(&[token])?;
        let word = String::from_utf8_lossy(&word);
        println!("Predict: {}", word);
        // tokens[0] = vec![token];
    }

    // tokens[0].clear();

    {
        // map the activations into vocab space
        let mut encoder = context.device.create_command_encoder(&Default::default());

        let ops = model.head_op(&buffer.ffn_x, &buffer.half, &buffer.out, true)?;

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.execute_tensor_op(&ops);
        drop(pass);

        encoder.copy_tensor(&buffer.out, &buffer.map)?;

        context.queue.submit(Some(encoder.finish()));

        // for each layer, choose the top 5 tokens
        let backed = buffer.map.back_async().await.to_vec();
        for layer in 0..info.num_layer {
            let start = layer * info.num_vocab;
            let end = start + info.num_vocab;
            let slice = &backed[start..end];

            let sorted = slice
                .iter()
                .enumerate()
                .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
                .take(5)
                .collect_vec();

            print!("layer {layer}:\t");
            for (index, (token, score)) in sorted.into_iter().enumerate() {
                let word = tokenizer.decode(&[token as u16]).unwrap_or_default();
                let word = String::from_utf8_lossy(&word);
                print!("{index}: {token} {word} ({score})\t");
            }
            println!()
        }
    }

    Ok(())
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EmbedDevice {
    #[default]
    Cpu,
    Gpu,
}

impl From<EmbedDevice> for web_rwkv::model::EmbedDevice {
    fn from(value: EmbedDevice) -> Self {
        match value {
            EmbedDevice::Cpu => Self::Cpu,
            EmbedDevice::Gpu => Self::Gpu,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: Option<PathBuf>,
    #[arg(short, long, value_name = "FILE")]
    lora: Option<PathBuf>,
    #[arg(short, long, value_name = "LAYERS")]
    quant: Option<usize>,
    #[arg(long, value_name = "LAYERS")]
    quant_nf4: Option<usize>,
    #[arg(short, long)]
    embed_device: Option<EmbedDevice>,
    #[arg(short, long, action)]
    turbo: bool,
    #[arg(short, long)]
    prompt: Option<String>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    run(cli).await.unwrap();
}
