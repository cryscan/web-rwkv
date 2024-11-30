use std::{
    convert::Infallible,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    str::FromStr,
};

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
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    model::{
        loader::{Loader, Lora},
        v4, v5, v6, Build, BuildFuture, ContextAutoLimits, Model, ModelBuilder, ModelInfo,
        ModelInput, ModelOutput, ModelState, ModelVersion, Quant, StateBuilder,
    },
    tokenizer::Tokenizer,
};

fn sample(probs: Vec<f32>, _top_p: f32) -> u16 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap()
        .0 as u16
}

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
    batch: usize,
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
        .quant(quant)
        .turbo(turbo)
        .token_chunk_size(token_chunk_size)
        .embed_device(embed_device.unwrap_or_default().into());
    let model: M = match lora {
        Some(lora) => {
            let data = SafeTensors::deserialize(lora)?;
            model
                .lora(Lora {
                    data,
                    blend: Default::default(),
                })
                .build()
                .await?
        }
        None => model.build().await?,
    };

    // The model state should keep the same batch as input.
    // [`BackedState::repeat`] is helpful if you want to create batch of states from the same input.
    let state = StateBuilder::new(context, model.info())
        .with_num_batch(batch)
        .with_chunk_size(4)
        .build()?;
    Ok((model, state))
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
    match info.version {
        ModelVersion::V4 => {
            let (model, state) = load_model(
                &context,
                &data,
                lora,
                cli.quant,
                cli.quant_nf4,
                cli.embed_device,
                cli.turbo,
                cli.token_chunk_size,
                cli.batch,
            )
            .await?;
            run_internal::<v4::Model<f16>, _>(model, state, tokenizer, cli.batch).await
        }
        ModelVersion::V5 => {
            let (model, state) = load_model(
                &context,
                &data,
                lora,
                cli.quant,
                cli.quant_nf4,
                cli.embed_device,
                cli.turbo,
                cli.token_chunk_size,
                cli.batch,
            )
            .await?;
            run_internal::<v5::Model<f16>, _>(model, state, tokenizer, cli.batch).await
        }
        ModelVersion::V6 => {
            let (model, state) = load_model(
                &context,
                &data,
                lora,
                cli.quant,
                cli.quant_nf4,
                cli.embed_device,
                cli.turbo,
                cli.token_chunk_size,
                cli.batch,
            )
            .await?;
            run_internal::<v6::Model<f16>, _>(model, state, tokenizer, cli.batch).await
        }
    }
}

async fn run_internal<M, S>(model: M, state: S, tokenizer: Tokenizer, batch: usize) -> Result<()>
where
    S: ModelState,
    M: Model<State = S>,
{
    #[cfg(not(debug_assertions))]
    let mut terminal = setup_terminal()?;

    let prompts = [
        "The Eiffel Tower is located in the city of",
        "The name of the capital of Italy is",
        "The Space Needle is located in downtown",
        "人们发现",
    ];
    let mut prompts = prompts.to_vec().repeat(batch.div_ceil(prompts.len()))[..batch]
        .iter()
        .map(|str| String::from_str(str).unwrap())
        .collect_vec();
    let tokens = prompts
        .clone()
        .iter()
        .map(|prompt| tokenizer.encode(prompt.as_bytes()).unwrap())
        .collect_vec();
    let mut tokens = tokens
        .into_iter()
        .map(|tokens| ModelInput {
            tokens,
            ..Default::default()
        })
        .collect();

    let mut num_token =
        [100usize, 400, 200, 300].to_vec().repeat((batch + 3) / 4)[..batch].to_vec();
    loop {
        #[cfg(not(debug_assertions))]
        terminal.draw(|frame| {
            let size = frame.area();

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

        let logits = model.run(&mut tokens, &state).await?;
        let probs = model.softmax(logits).await?;
        for (index, probs) in probs.iter().enumerate().filter_map(|(index, x)| match x {
            ModelOutput::Full(x) => Some((index, x.last()?)),
            ModelOutput::Last(x) => Some((index, x)),
            _ => None,
        }) {
            if num_token[index] > 0 {
                let token = sample(probs.to_vec(), 0.5);
                let decoded = tokenizer.decode(&[token])?;
                let word = String::from_utf8_lossy(&decoded);
                tokens[index].tokens = vec![token];
                prompts[index].push_str(&word);
                num_token[index] -= 1;
            } else {
                tokens[index].tokens = vec![];
            }
        }

        if num_token.iter().all(|x| *x == 0) {
            break;
        }
    }

    #[cfg(not(debug_assertions))]
    restore_terminal(&mut terminal)?;

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
    #[arg(long, default_value_t = 128)]
    token_chunk_size: usize,
    #[arg(short, long, default_value_t = 4)]
    batch: usize,
    #[arg(short, long, action)]
    adapter: bool,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    run(cli).await.unwrap();
}
