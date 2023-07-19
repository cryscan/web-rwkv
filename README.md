# Web-RWKV
This is an implementation of the [language model of RWKV](https://github.com/BlinkDL/RWKV-LM) in pure WebGPU.

## Compile and Run
1. [Install Rust](https://rustup.rs/).
2. Run `cargo r -r --example generation` to generate 100 tokens and test the time cost.

## Credits
- Tokenizer is implemented by [@koute](https://github.com/koute/rwkv_tokenizer).