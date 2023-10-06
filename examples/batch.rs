use anyhow::Result;
use clap::Parser;
#[cfg(not(debug_assertions))]
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
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
use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    str::FromStr,
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{
        v5::{Model, ModelBuilder, ModelState},
        LayerFlags, Lora, ModelExt, Quantization,
    },
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

fn load_model(
    context: &Context,
    model: PathBuf,
    lora: Option<PathBuf>,
    quant: Option<u64>,
) -> Result<Model<'_>> {
    let file = File::open(model)?;
    let map = unsafe { Mmap::map(&file)? };
    let quant = quant
        .map(|bits| Quantization::Int8(LayerFlags::from_bits_retain(bits)))
        .unwrap_or_default();
    let model = ModelBuilder::new(&context, &map).with_quant(quant);

    let model = match lora {
        Some(lora) => {
            let file = File::open(lora)?;
            let map = unsafe { Mmap::map(&file)? };
            model
                .add_lora(Lora {
                    data: &map,
                    blend: Default::default(),
                })
                .build()?
        }
        None => model.build()?,
    };

    println!("{:#?}", model.info());
    Ok(model)
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
    let context = create_context().await?;
    #[cfg(not(debug_assertions))]
    let mut terminal = setup_terminal()?;

    let tokenizer = load_tokenizer()?;
    let model = cli.model.unwrap_or(
        std::fs::read_dir("assets/models")
            .unwrap()
            .filter_map(|x| x.ok())
            .filter(|x| x.path().extension().is_some_and(|x| x == "st"))
            .next()
            .unwrap()
            .path(),
    );
    let model = load_model(&context, model, cli.lora, cli.quant)?;

    let prompts = [
        "The Eiffel Tower is located in the city of",
        "The name of the capital of Italy is",
        "The Space Needle is located in downtown",
        "人们发现",
    ];
    let mut prompts = prompts
        .to_vec()
        .repeat((cli.batch + prompts.len() - 1) / prompts.len())[..cli.batch]
        .into_iter()
        .map(|str| String::from_str(str).unwrap())
        .collect_vec();
    let mut tokens = prompts
        .clone()
        .iter()
        .map(|prompt| tokenizer.encode(prompt.as_bytes()).unwrap())
        .collect_vec();

    // The model state should keep the same batch as input.
    // [`BackedState::repeat`] is helpful if you want to create batch of states from the same input.
    let state = ModelState::new(&context, model.info(), tokens.len(), 4);

    let mut num_tokens = [100usize, 400, 200, 300]
        .to_vec()
        .repeat((cli.batch + 3) / 4)[..cli.batch]
        .to_vec();
    loop {
        #[cfg(not(debug_assertions))]
        terminal.draw(|frame| {
            let size = frame.size();

            let block = Block::default().black();
            frame.render_widget(block, size);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(
                    [
                        Constraint::Percentage(25),
                        Constraint::Percentage(25),
                        Constraint::Percentage(25),
                        Constraint::Percentage(25),
                    ]
                    .as_ref(),
                )
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

        let logits = model.run(&mut tokens, &state)?;
        let probs = model.softmax(logits)?;
        for (index, probs) in probs
            .into_iter()
            .enumerate()
            .filter_map(|(index, x)| x.map(|x| (index, x)))
        {
            if num_tokens[index] > 0 {
                let token = sample(probs.to_vec(), 0.5);
                let word = String::from_utf8(tokenizer.decode(&[token])?)?;
                tokens[index] = vec![token];
                prompts[index].push_str(&word);
                num_tokens[index] -= 1;
            } else {
                tokens[index] = vec![];
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: Option<PathBuf>,
    #[arg(short, long, value_name = "FILE")]
    lora: Option<PathBuf>,
    #[arg(short, long, value_name = "LAYERS")]
    quant: Option<u64>,
    #[arg(short, long, default_value_t = 4)]
    batch: usize,
}

fn main() {
    let cli = Cli::parse();
    pollster::block_on(run(cli)).unwrap();
}
