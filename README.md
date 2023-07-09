# Web-RWKV
[![crates.io](https://img.shields.io/crates/v/web-rwkv)](https://crates.io/crates/web-rwkv)
[![docs.rs](https://docs.rs/web-rwkv/badge.svg)](https://docs.rs/web-rwkv)

This is an implementation of the [language model of RWKV](https://github.com/BlinkDL/RWKV-LM) in pure WebGPU.

## Compile and Run
1. [Install Rust](https://rustup.rs/).
2. Run `cargo run --release --example gen` to generate 100 tokens and test the time cost.

## Convert Models
You may download the official RWKV World series models from [HuggingFace](https://huggingface.co/BlinkDL/rwkv-4-world), and convert them via the provided [`convert_safetensors.py`](convert_safetensors.py).

An already-converted 0.4B model can be found under [`assets/models`](assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st).

## Credits
- Tokenizer is implemented by [@koute](https://github.com/koute/rwkv_tokenizer).