use std::{
    convert::Infallible,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{
        loader::{Loader, Lora},
        run::{HookMap, ModelRun},
        softmax::ModelSoftmax,
        v5, Build, BuildFuture, ContextAutoLimits, Model, ModelBuilder, ModelInfo, ModelInput,
        ModelOutput, ModelState, ModelVersion, Quant, StateBuilder,
    },
    tensor::{
        kind::ReadWrite,
        ops::{TensorOp, TensorPass},
        TensorError, TensorGpu, TensorShape,
    },
    tokenizer::Tokenizer,
};

#[derive(Debug, Clone)]
struct Buffer {
    ffn_x: TensorGpu<f16, ReadWrite>,
    out: TensorGpu<f32, ReadWrite>,
}

impl Buffer {
    fn new(context: &Context, info: &ModelInfo) -> Self {
        Self {
            ffn_x: context.tensor_init([info.num_emb, info.num_layer, 1, 1]),
            out: context.tensor_init([info.num_vocab, info.num_layer, 1, 1]),
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

async fn create_context(info: &ModelInfo, _auto: bool) -> Result<Context> {
    let instance = Instance::new();
    #[cfg(not(debug_assertions))]
    let adapter = if _auto {
        instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await?
    } else {
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

#[allow(clippy::too_many_arguments)]
async fn load_model<'a, M, S>(
    context: &Context,
    data: &'a [u8],
    lora: Option<&'a [u8]>,
    quant: usize,
    quant_nf4: usize,
    embed_device: Option<EmbedDevice>,
    turbo: bool,
    token_chunk_size: usize,
) -> Result<(M, S)>
where
    M: Model<State = S>,
    S: ModelState,
    ModelBuilder<SafeTensors<'a>>: BuildFuture<M, Error = anyhow::Error>,
    StateBuilder: Build<S, Error = Infallible>,
{
    let quant = (0..quant)
        .map(|layer| (layer, Quant::Int8))
        .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
        .collect();

    let model = SafeTensors::deserialize(data)?;
    let model = ModelBuilder::new(context, model)
        .with_quant(quant)
        .with_turbo(turbo)
        .with_token_chunk_size(token_chunk_size)
        .with_embed_device(embed_device.unwrap_or_default().into());
    let model: M = match lora {
        Some(lora) => {
            let data = SafeTensors::deserialize(lora)?;
            model
                .add_lora(Lora {
                    data,
                    blend: Default::default(),
                })
                .build()
                .await?
        }
        None => model.build().await?,
    };

    let state: S = StateBuilder::new(context, model.info()).build()?;
    Ok((model, state))
}

async fn run(cli: Cli) -> Result<()> {
    let tokenizer = load_tokenizer()?;
    let model = cli.model.unwrap_or_else(|| {
        std::fs::read_dir("assets/models")
            .unwrap()
            .filter_map(|x| x.ok())
            .find(|x| x.path().extension().is_some_and(|x| x == "st"))
            .unwrap()
            .path()
    });

    let file = File::open(model)?;
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    if info.version != ModelVersion::V5 {
        bail!("this demo only supports v5");
    }
    println!("{:#?}", info);

    let lora = match cli.lora {
        Some(lora) => {
            let file = File::open(lora)?;
            let data = unsafe { Mmap::map(&file)? };
            Some(data)
        }
        None => None,
    };
    let lora = lora.as_deref();

    let context = create_context(&info, cli.adapter).await?;
    let (model, state) = load_model::<v5::Model<f16>, _>(
        &context,
        &data,
        lora,
        cli.quant,
        cli.quant_nf4,
        cli.embed_device,
        cli.turbo,
        cli.token_chunk_size,
    )
    .await?;

    // create a buffer to store each layer's output
    let buffer = Buffer::new(&context, &info);

    let mut hooks = HookMap::default();
    for layer in 0..info.num_layer {
        let buffer = buffer.clone();
        hooks.insert(
            v5::Hook::PostFfn(layer),
            Box::new(
                move |_model,
                      _state,
                      runtime: &v5::Runtime<_>,
                      _header|
                      -> Result<TensorOp, TensorError> {
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

    let mut tokens = vec![ModelInput {
        tokens: tokenizer.encode(prompt.as_bytes())?,
        ..Default::default()
    }];
    println!("Prompt: {}", prompt);

    // run initial prompt
    let logits = loop {
        let logits = model.run_with_hooks(&mut tokens, &state, &hooks).await?;
        if logits.iter().any(ModelOutput::is_some) {
            break logits;
        }
    };
    let probs = model.softmax(logits).await?;

    if let ModelOutput::Last(probs) = &probs[0] {
        let token = sample(probs, 0.5);
        let word = tokenizer.decode(&[token])?;
        let word = String::from_utf8_lossy(&word);
        println!("Predict: {}", word);
    }

    // map the activations into vocab space
    let mut encoder = context.device.create_command_encoder(&Default::default());

    let tensor = model.tensor();
    let ops = TensorOp::List(vec![
        TensorOp::layer_norm(
            &tensor.head.layer_norm.w,
            &tensor.head.layer_norm.b,
            &buffer.ffn_x,
            None,
            v5::Model::<f16>::LN_EPS,
        )?,
        tensor.head.w.matmul_mat_op(
            buffer.ffn_x.view(.., .., .., ..)?,
            buffer.out.view(.., .., .., ..)?,
            Default::default(),
        )?,
    ]);

    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.execute_tensor_op(&ops);
    drop(pass);

    context.queue.submit(Some(encoder.finish()));

    // for each layer, choose the top 5 tokens
    let backed = buffer.out.back().await.to_vec();
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
    #[arg(short, long, value_name = "LAYERS", default_value_t = 0)]
    quant: usize,
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant_nf4: usize,
    #[arg(short, long)]
    embed_device: Option<EmbedDevice>,
    #[arg(short, long, action)]
    turbo: bool,
    #[arg(long, default_value_t = 32)]
    token_chunk_size: usize,
    #[arg(short, long)]
    prompt: Option<String>,
    #[arg(short, long, action)]
    adapter: bool,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    run(cli).await.unwrap();
}
