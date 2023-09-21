# Web-RWKV
[![crates.io](https://img.shields.io/crates/v/web-rwkv)](https://crates.io/crates/web-rwkv)
[![docs.rs](https://docs.rs/web-rwkv/badge.svg)](https://docs.rs/web-rwkv)

This is an inference engine for the [language model of RWKV](https://github.com/BlinkDL/RWKV-LM) implemented in pure WebGPU.

## Features
- No dependencies on CUDA/Python.
- Support Nvidia/AMD/Intel GPUs, including integrated GPUs.
- Batched inference.
- Int8 quantization.
- Very fast.

![chat](screenshots/chat.gif)
![batch](screenshots/batch.gif)

## Compile and Run
1. [Install Rust](https://rustup.rs/).
2. Run `cargo run --release --example gen` to generate 100 tokens and measure the time cost.
3. Run `cargo run --release --example chat` to chat with the model.
4. Run `cargo run --release --example batch` to generate 4 batches of text with various lengths simultaneously.
5. To specify the location of your safetensors model, use `cargo run --release --example chat -- --model /path/to/model`.

Or you can download the pre-compiled binaries from the release page and run
```bash
$ chat --model /path/to/model
```

## Use in Your Project
To use in your own rust project, simply add `web-rwkv = "0.2"` as a dependency in your `Cargo.toml`.
Check examples on how to create the environment, the tokenizer and how to run the model.

### Explanation of Batched Inference
Since version v0.2.4, the engine supports batched inference, i.e., inference of a batch of prompts (with different length) in parallel.
This is achieved by a modified `WKV` kernel.

When building the model, the user specifies `token_chunk_size` (default: 32, but for powerful GPUs this could be much higher), which is the maximum number of tokens the engine could process in one `run` call.

After creating the model, the user creates a `ModelState` with `max_batch` specified.
This means that there are `max_batch` slots that could consume the inputs in parallel.

Before calling `run()`, the user fills each slot with some tokens as prompt.
If a slot is empty, no inference will be run for it.

After calling `run()`, some (but may not be all) input tokens are consumed, and `logits` appears in their corresponding returned slots if the inference of that slot is finished during this run.
Since there are only `token_chunk_size` tokens are processed during each `run()` call, there may be none of `logits` appearing in the results.

## Convert Models
*You must download the model and put in `assets/models` before running if you are building from source.*
You can now download the coverted models [here](https://huggingface.co/cgisky/RWKV-safetensors-fp16).

You may download the official RWKV World series models from [HuggingFace](https://huggingface.co/BlinkDL/rwkv-4-world), and convert them via the provided [`convert_safetensors.py`](convert_safetensors.py).

An already-converted 0.4B model can be found under [`assets/models`](assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st).

## Troubleshoot
- "thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: HeaderTooLarge'"
  
  Your model is broken, mainly because you cloned the repo but did not set up git-lfs.Please download the model manually and overwrite that one in `assets/models`.

- "thread 'main' panicked at 'wgpu error: Validation Error"
  
  It's most likely that you are using the D3D backend.
  Please use Vulkan backend instead.

- "thread 'main' panicked at 'Error in Queue::submit: parent device is lost'"

  Your GPU is not responding.
  Maybe you are running a model that is just too big for your device. If the model doesn't fit into your VRam, the driver needs to constantly swap and transfer the model parameters, causing it to be 10x slower.
  Try to quantize your model first.


## Credits
- Tokenizer is implemented by [@koute](https://github.com/koute/rwkv_tokenizer).