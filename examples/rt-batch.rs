use std::{path::PathBuf, str::FromStr};

use anyhow::Result;
use clap::{Parser, ValueEnum};
#[cfg(not(debug_assertions))]
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
#[cfg(not(debug_assertions))]
use ratatui::{
    prelude::{Constraint, CrosstermBackend, Direction, Layout},
    style::{Color, Modifier, Style, Stylize},
    text::{Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use safetensors::SafeTensors;
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    runtime::{
        infer::{InferInput, InferInputBatch},
        loader::{Loader, Lora},
        model::{Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        softmax::softmax,
        v4, v5, v6, JobRuntime, Submission,
    },
    tokenizer::Tokenizer,
};

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
    Ok(context)
}

async fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/rwkv_vocab_v20230424.json").await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    Ok(Tokenizer::new(&contents)?)
}

#[cfg(not(debug_assertions))]
fn setup_terminal() -> Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    let mut stdout = std::io::stdout();
    enable_raw_mode()?;
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    Ok(Terminal::new(CrosstermBackend::new(stdout))?)
}

#[cfg(not(debug_assertions))]
fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen,)?;
    Ok(terminal.show_cursor()?)
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
    #[arg(short, long, action)]
    turbo: bool,
    #[arg(short, long)]
    embed_device: Option<EmbedDevice>,
    #[arg(long, default_value_t = 32)]
    token_chunk_size: usize,
    #[arg(short, long, default_value_t = 4)]
    batch: usize,
    #[arg(short, long, action)]
    adapter: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("rt_gen", log::LevelFilter::Info)
        .init()
        .unwrap();
    let cli = Cli::parse();
    let batch = cli.batch;

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
        .with_embed_device(embed_device)
        .with_quant(quant)
        .with_num_batch(batch);
    let builder = match &lora {
        Some(data) => {
            let data = SafeTensors::deserialize(data)?;
            let blend = Default::default();
            let lora = Lora { data, blend };
            builder.add_lora(lora)
        }
        None => builder,
    };

    let runtime = match info.version {
        ModelVersion::V4 => {
            let runtime = Build::<v4::ModelRuntime<f16>>::build(builder).await?;
            JobRuntime::new(runtime).await
        }
        ModelVersion::V5 => {
            let runtime = Build::<v5::ModelRuntime<f16>>::build(builder).await?;
            JobRuntime::new(runtime).await
        }
        ModelVersion::V6 => {
            let runtime = Build::<v6::ModelRuntime<f16>>::build(builder).await?;
            JobRuntime::new(runtime).await
        }
    };

    #[cfg(not(debug_assertions))]
    let mut terminal = setup_terminal()?;

    let prompts = [
        "The Eiffel Tower is located in the city of",
        "The name of the capital of Italy is",
        "The Space Needle is located in downtown",
        "人们发现",
    ];
    let mut prompts = prompts
        .to_vec()
        .repeat((batch + prompts.len() - 1) / prompts.len())[..batch]
        .iter()
        .map(|str| String::from_str(str).unwrap())
        .collect_vec();
    let tokens = prompts
        .clone()
        .iter()
        .map(|prompt| tokenizer.encode(prompt.as_bytes()).unwrap())
        .collect_vec();

    let mut inference = InferInput::new(
        tokens
            .into_iter()
            .map(|tokens| InferInputBatch {
                tokens,
                ..Default::default()
            })
            .collect(),
        cli.token_chunk_size,
    );

    let mut num_tokens =
        [100usize, 400, 200, 300].to_vec().repeat((batch + 3) / 4)[..batch].to_vec();

    loop {
        #[cfg(not(debug_assertions))]
        terminal.draw(|frame| {
            let size = frame.size();

            let block = Block::default().black();
            frame.render_widget(block, size);

            let constraints = (0..batch)
                .map(|_| Constraint::Percentage(100 / batch as u16))
                .collect_vec();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(size);

            let create_block = |title| {
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Gray))
                    .title(Span::styled(
                        title,
                        Style::default().add_modifier(Modifier::BOLD),
                    ))
            };

            for (index, (text, chunk)) in prompts.iter().zip(chunks.iter()).enumerate() {
                let text = Text::from(text.as_str());
                let text_height_estimation: usize = text
                    .lines
                    .iter()
                    .map(|line| (line.width() / 1.max(chunk.width as usize - 2)).max(1))
                    .sum();
                let scroll =
                    (text_height_estimation as isize - chunk.height as isize + 2).max(0) as u16;
                let paragraph = Paragraph::new(text)
                    .style(Style::default().fg(Color::Gray))
                    .block(create_block(format!("Batch {index}")))
                    .wrap(Wrap { trim: true })
                    .scroll((scroll, 0));
                frame.render_widget(paragraph, *chunk);
            }
        })?;

        #[cfg(debug_assertions)]
        for (index, prompt) in prompts.iter().enumerate() {
            println!("{index}: {prompt}");
        }

        let input = inference.clone();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let submission = Submission { input, sender };

        let _ = runtime.send(submission).await;
        let (input, output) = receiver.await.unwrap();
        inference = input;

        let output = output.iter().map(|x| x.output.clone()).collect_vec();
        let output = softmax(&context, output).await?;
        for (index, batch) in output.iter().enumerate() {
            if batch.size() == 0 {
                continue;
            }
            if num_tokens[index] > 0 {
                let batch = batch.clone().map(|x| x.to_f32()).to_vec();
                let token = sample(&batch, 0.5);
                let decoded = tokenizer.decode(&[token])?;
                let word = String::from_utf8_lossy(&decoded);
                inference.batches[index].tokens = vec![token];
                prompts[index].push_str(&word);
                num_tokens[index] -= 1;
            }
        }

        if num_tokens.iter().all(|x| *x == 0) {
            break;
        }
    }

    #[cfg(not(debug_assertions))]
    restore_terminal(&mut terminal)?;

    Ok(())
}
