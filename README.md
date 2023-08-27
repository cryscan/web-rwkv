# Web-RWKV
[![crates.io](https://img.shields.io/crates/v/web-rwkv)](https://crates.io/crates/web-rwkv)
[![docs.rs](https://docs.rs/web-rwkv/badge.svg)](https://docs.rs/web-rwkv)

This is an implementation of the [language model of RWKV](https://github.com/BlinkDL/RWKV-LM) in pure WebGPU.

## Compile and Run
1. [Install Rust](https://rustup.rs/).
2. Run `cargo run --release --example gen` to generate 100 tokens and measure the time cost.
3. Run `cargo run --release --example chat` to chat with the model.
4. To specify the location of your safetensors model, use `cargo run --release --example chat -- --model /path/to/model`.

Or you can download the pre-compiled binaries from the release page and run
```bash
$ chat --model /path/to/model
```

## Use in Your Project
To use in your own rust project, simply add `web-rwkv = "0.2"` as a dependency in your `Cargo.toml`.
Check examples on how to create the environment, the tokenizer and how to run the model.

## Convert Models
You can now download the coverted models [here](https://huggingface.co/cgisky/RWKV-safetensors-fp16).

You may download the official RWKV World series models from [HuggingFace](https://huggingface.co/BlinkDL/rwkv-4-world), and convert them via the provided [`convert_safetensors.py`](convert_safetensors.py).

An already-converted 0.4B model can be found under [`assets/models`](assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st).

## Credits
- Tokenizer is implemented by [@koute](https://github.com/koute/rwkv_tokenizer).