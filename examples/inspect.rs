use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use clap::{Parser, ValueEnum};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
#[cfg(feature = "trace")]
use tracing_subscriber::layer::SubscriberExt;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    num::Float,
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption},
        loader::{Loader, Lora},
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        v7, Runtime, TokioRuntime,
    },
    tensor::ops::TensorOp,
    tokenizer::Tokenizer,
};

async fn create_context(info: &ModelInfo, _auto: bool) -> Result<Context> {
    let instance = wgpu::Instance::default();
    #[cfg(not(debug_assertions))]
    let adapter = if _auto {
        instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await?
    } else {
        let backends = wgpu::Backends::all();
        let adapters = instance.enumerate_adapters(backends);
        let names = adapters
            .iter()
            .map(|adapter| adapter.get_info())
            .map(|info| format!("{} ({:?})", info.name, info.backend))
            .collect_vec();
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Please select an adapter")
            .default(0)
            .items(&names)
            .interact()?;
        adapters.into_iter().nth(selection).unwrap()
    };
    #[cfg(debug_assertions)]
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

async fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/vocab/rwkv_vocab_v20230424.json").await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    Ok(Tokenizer::new(&contents)?)
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EmbedDevice {
    #[default]
    Cpu,
    Gpu,
}

impl From<EmbedDevice> for web_rwkv::runtime::model::EmbedDevice {
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
    model: PathBuf,
    #[arg(short, long, value_name = "FILE")]
    lora: Option<PathBuf>,
    #[arg(short, long, value_name = "LAYERS", default_value_t = 0)]
    quant: usize,
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant_nf4: usize,
    #[arg(short, long)]
    embed_device: Option<EmbedDevice>,
    #[arg(long, default_value_t = 128)]
    token_chunk_size: usize,
    #[arg(short, long, action)]
    adapter: bool,
}

fn make_hooks<F: Float>(info: &ModelInfo, frames: Vec<v7::Runtime<F>>) -> Result<v7::HookMap<F>> {
    let mut hooks = v7::HookMap::new();

    for (layer, x) in frames.iter().enumerate().take(info.num_layer) {
        let x = x.clone();
        hooks.insert(
            v7::Hook::PostFfn(layer),
            Box::new(move |frame: v7::Frame<F>| {
                let ops = vec![
                    TensorOp::blit(&frame.buffer.input, &x.input)?,
                    TensorOp::blit(&frame.buffer.x, &x.x)?,
                    TensorOp::blit(&frame.buffer.att_x, &x.att_x)?,
                    TensorOp::blit(&frame.buffer.att_v0, &x.att_v0)?,
                    TensorOp::blit(&frame.buffer.att_rx, &x.att_rx)?,
                    TensorOp::blit(&frame.buffer.att_wx, &x.att_wx)?,
                    TensorOp::blit(&frame.buffer.att_kx, &x.att_kx)?,
                    TensorOp::blit(&frame.buffer.att_vx, &x.att_vx)?,
                    TensorOp::blit(&frame.buffer.att_ax, &x.att_ax)?,
                    TensorOp::blit(&frame.buffer.att_gx, &x.att_gx)?,
                    TensorOp::blit(&frame.buffer.att_r, &x.att_r)?,
                    TensorOp::blit(&frame.buffer.att_w, &x.att_w)?,
                    TensorOp::blit(&frame.buffer.att_k, &x.att_k)?,
                    TensorOp::blit(&frame.buffer.att_v, &x.att_v)?,
                    TensorOp::blit(&frame.buffer.att_a, &x.att_a)?,
                    TensorOp::blit(&frame.buffer.att_g, &x.att_g)?,
                    TensorOp::blit(&frame.buffer.att_o, &x.att_o)?,
                    TensorOp::blit(&frame.buffer.att_kk, &x.att_kk)?,
                    TensorOp::blit(&frame.buffer.att_vv, &x.att_vv)?,
                    TensorOp::blit(&frame.buffer.att_n, &x.att_n)?,
                    TensorOp::blit(&frame.buffer.aux_w, &x.aux_w)?,
                    TensorOp::blit(&frame.buffer.aux_a, &x.aux_a)?,
                    TensorOp::blit(&frame.buffer.aux_g, &x.aux_g)?,
                    TensorOp::blit(&frame.buffer.aux_v, &x.aux_v)?,
                    TensorOp::blit(&frame.buffer.ffn_x, &x.ffn_x)?,
                    TensorOp::blit(&frame.buffer.ffn_kx, &x.ffn_kx)?,
                    TensorOp::blit(&frame.buffer.ffn_k, &x.ffn_k)?,
                    TensorOp::blit(&frame.buffer.ffn_v, &x.ffn_v)?,
                ];
                Ok(TensorOp::List(ops))
            }),
        );
    }

    Ok(hooks)
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("inspect", log::LevelFilter::Info)
        .init()?;
    #[cfg(feature = "trace")]
    {
        let registry = tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default());
        tracing::subscriber::set_global_default(registry)?;
    }

    let cli = Cli::parse();

    let tokenizer = load_tokenizer().await?;

    let file = File::open(cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    log::info!("{:#?}", info);

    let context = create_context(&info, cli.adapter).await?;
    log::info!("{:#?}", context.adapter.get_info());

    let quant = (0..cli.quant)
        .map(|layer| (layer, Quant::Int8))
        .chain((0..cli.quant_nf4).map(|layer| (layer, Quant::NF4)))
        .collect();
    let embed_device = cli.embed_device.unwrap_or(EmbedDevice::Cpu).into();
    let lora = match cli.lora {
        Some(path) => {
            let file = File::open(path).await?;
            let mut reader = BufReader::new(file);
            let mut data = vec![];
            reader.read_to_end(&mut data).await?;
            Some(data)
        }
        None => None,
    };

    let builder = ModelBuilder::new(&context, model)
        .embed_device(embed_device)
        .quant(quant);
    let builder = match &lora {
        Some(data) => {
            let data = SafeTensors::deserialize(data)?;
            let blend = Default::default();
            let lora = Lora { data, blend };
            builder.lora(lora)
        }
        None => builder,
    };

    let frames = (0..info.num_layer)
        .map(|_| v7::Runtime::<f16>::new(&context, &info, 1))
        .collect_vec();

    let runtime: Box<dyn Runtime> = match info.version {
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let hooks = make_hooks(&info, frames.clone())?;
            let bundle = v7::Bundle::<f16>::new_with_hooks(model, 1, hooks);
            Box::new(TokioRuntime::new(bundle).await)
        }
        _ => unimplemented!(),
    };

    const PROMPT: &str = "Tell me about the Eiffel Tower";
    let tokens = tokenizer.encode(PROMPT.as_bytes())?;

    let mut data = Vec::with_capacity(tokens.len());
    for (ti, token) in tokens.into_iter().enumerate() {
        let prompt = InferInputBatch {
            tokens: vec![token],
            option: InferOption::Last,
        };
        let input = InferInput::new(vec![prompt], cli.token_chunk_size);
        let (_input, _output) = runtime.infer(input).await?;

        let mut buffers = HashMap::<String, _>::new();
        for (index, layer) in frames.iter().enumerate() {
            buffers.insert(format!("{ti}_{index}_input"), layer.input.back().await);
            buffers.insert(format!("{ti}_{index}_x"), layer.x.back().await);
            buffers.insert(format!("{ti}_{index}_att_x"), layer.att_x.back().await);
            buffers.insert(format!("{ti}_{index}_att_v0"), layer.att_v0.back().await);
            buffers.insert(format!("{ti}_{index}_att_rx"), layer.att_rx.back().await);
            buffers.insert(format!("{ti}_{index}_att_wx"), layer.att_wx.back().await);
            buffers.insert(format!("{ti}_{index}_att_kx"), layer.att_kx.back().await);
            buffers.insert(format!("{ti}_{index}_att_vx"), layer.att_vx.back().await);
            buffers.insert(format!("{ti}_{index}_att_ax"), layer.att_ax.back().await);
            buffers.insert(format!("{ti}_{index}_att_gx"), layer.att_gx.back().await);
            buffers.insert(format!("{ti}_{index}_att_r"), layer.att_r.back().await);
            buffers.insert(format!("{ti}_{index}_att_w"), layer.att_w.back().await);
            buffers.insert(format!("{ti}_{index}_att_k"), layer.att_k.back().await);
            buffers.insert(format!("{ti}_{index}_att_v"), layer.att_v.back().await);
            buffers.insert(format!("{ti}_{index}_att_a"), layer.att_a.back().await);
            buffers.insert(format!("{ti}_{index}_att_g"), layer.att_g.back().await);
            buffers.insert(format!("{ti}_{index}_att_o"), layer.att_o.back().await);
            buffers.insert(format!("{ti}_{index}_att_kk"), layer.att_kk.back().await);
            buffers.insert(format!("{ti}_{index}_att_vv"), layer.att_vv.back().await);
            buffers.insert(format!("{ti}_{index}_att_n"), layer.att_n.back().await);
            buffers.insert(format!("{ti}_{index}_aux_w"), layer.aux_w.back().await);
            buffers.insert(format!("{ti}_{index}_aux_a"), layer.aux_a.back().await);
            buffers.insert(format!("{ti}_{index}_aux_g"), layer.aux_g.back().await);
            buffers.insert(format!("{ti}_{index}_aux_v"), layer.aux_v.back().await);
            buffers.insert(format!("{ti}_{index}_ffn_x"), layer.ffn_x.back().await);
            buffers.insert(format!("{ti}_{index}_ffn_kx"), layer.ffn_kx.back().await);
            buffers.insert(format!("{ti}_{index}_ffn_k"), layer.ffn_k.back().await);
            buffers.insert(format!("{ti}_{index}_ffn_v"), layer.ffn_v.back().await);
        }
        let buffers: HashMap<_, _> = buffers
            .into_iter()
            .map(|(key, value)| (key, value.map(|x| x.to_f32()).to_vec()))
            .collect();
        data.push(buffers);
    }

    print!("{}", serde_json::to_string(&data)?);

    Ok(())
}
