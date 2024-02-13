# Web-RWKV
[![crates.io](https://img.shields.io/crates/v/web-rwkv)](https://crates.io/crates/web-rwkv)
[![docs.rs](https://docs.rs/web-rwkv/badge.svg)](https://docs.rs/web-rwkv)

<p align='center'><image src="assets/logo-ba.png"></p>

This is an inference engine for the [language model of RWKV](https://github.com/BlinkDL/RWKV-LM) implemented in pure WebGPU.

## Features
- No dependencies on CUDA/Python.
- Support Nvidia/AMD/Intel GPUs, including integrated GPUs.
- Vulkan/Dx12/OpenGL backends.
- Batched inference.
- Int8 and NF4 quantization.
- Very fast.
- LoRA merging at loading time.
- Support RWKV V4, V5 and V6.
- Hooks to intervene the inference process at any point.

<p align='center'>
<image src="screenshots/chat.gif">
<image src="screenshots/batch.gif">
</p>

Note that `web-rwkv` is only an inference engine. It only provides the following functionalities:
- A tokenizer.
- Model loading.
- State creation and updating.
- A `run` function that takes in prompt tokens and returns logits (predicted next token probabilities after calling `softmax`).

It *does not* provide the following:
- OpenAI API or APIs of any kind.
  - If you would like to deploy an API server, check [AI00 RWKV Server](https://github.com/cgisky1980/ai00_rwkv_server) which is a fully-functional OpenAI-compatible API server built upon `web-rwkv`.
  - You could also check the [`web-rwkv-axum`](https://github.com/Prunoideae/web-rwkv-axum) project if you want some fancy inference pipelines, including Classifier-Free Guidance (CFG), Backus–Naur Form (BNF) guidance, and more.
- Samplers, though in the examples a basic nucleus sampler is implemented, this is *not* included in the library itself.
- State caching or management system.
- Python (or any other languages) binding.
- Runtime. Without a runtime makes it easy to be integrated into any applications from servers, front-end apps (yes, `web-rwkv` can run in browser) to game engines.

## Compile
1. [Install Rust](https://rustup.rs/).
2. Download the model from [HuggingFace](https://huggingface.co/BlinkDL/rwkv-5-world), and convert it using [`convert_safetensors.py`](./convert_safetensors.py). Put the `.st` model under `assets/models`.
3. Compile
   ```bash
   $ cargo build --release --examples
   ```

## Examples

### Performance Test
The test generates 500 tokens and measure the time cost.
```bash
$ cargo run --release --example gen
```

### Chat Demo
To chat with the model, run
```bash
$ cargo run --release --example chat
```

In this demo, type `+` to retry last round's generation; type `-` to exit.

- To specify the location of your safetensors model, use 
   ```bash
   $ cargo run --release --example chat -- --model /path/to/model
   ```

- To load custom prompts for chat, use 
   ```bash
   $ cargo run --release --example chat -- --prompt /path/to/prompt
   ```
   See [`assets/prompt.json`](./assets/prompt.json) for details.

- To specify layer quantization, use `--quant <LAYERS>` or `--quant-nf4 <LAYERS>` to quantize the first `<LAYERS>` layers. For example, use 
  ```bash
  $ cargo run --release --example chat -- --quant 32
  ```
  to quantize all 32 layers.

- Use `--turbo` flag to switch to alternative `GEMM` kernel when inferring long prompts.


### Batched Inference
This demo showcases generation of 4 batches of text with various lengths simultaneously.
```bash
$ cargo run --release --example batch
```

### Inspector
The inspector demo is a guide to an advanced usage called hooks. Hooks allow user to inject any tensor ops into the model's inference process, fetching and modifying the contents of the runtime buffer, state, and even the model parameters. Hooks enable certain third-party implementations like dynamic LoRA, control net, and so on.

## Use in Your Project
To use in your own rust project, simply add `web-rwkv = "0.6"` as a dependency in your `Cargo.toml`.
Check examples on how to create the environment, the tokenizer and how to run the model.

### Explanation of Batched Inference
Since version v0.2.4, the engine supports batched inference, i.e., inference of a batch of prompts (with different length) in parallel.
This is achieved by a modified `WKV` kernel.

When building the model, the user specifies `token_chunk_size` (default: 32, but for powerful GPUs this could be much higher), which is the maximum number of tokens the engine could process in one `run` call.

After creating the model, the user creates a `ModelState` with `num_batch` specified.
This means that there are `num_batch` slots that could consume the inputs in parallel.

Before calling `run()`, the user fills each slot with some tokens as prompt.
If a slot is empty, no inference will be run for it.

After calling `run()`, some (but may not be all) input tokens are consumed, and `logits` appears in their corresponding returned slots if the inference of that slot is finished during this run.
Since there are only `token_chunk_size` tokens are processed during each `run()` call, there may be none of `logits` appearing in the results.

### Explanation of Hooks
Hooks are a very powerful tool for customizing model inference process.
The library provides with the `Model::run_with_hooks` function, which takes into a `HookMap` as a parameter.

A `HookMap` is essentially a hashmap from `Model::Hook` to functions.
A `Model::Hook` defines a certain place the hook function can be injected into. A model generally has dozens of hooking points.
A hook function is a function of `Fn(&Model<'_>, &ModelState, &Runtime) -> Result<TensorOp, TensorError>`, where you can create tensor ops that reads/writes all the tensors you get here.

An example that reads out every layer's output:
```rust
let info = model.info();
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
let output = model.run_with_hooks(&mut tokens, &state, &hooks).await?;
```

## Convert Models
*You must download the model and put in `assets/models` before running if you are building from source.*
You can now download the converted models [here](https://huggingface.co/cgisky/RWKV-safetensors-fp16).

You may download the official RWKV World series models from [HuggingFace](https://huggingface.co/BlinkDL/rwkv-5-world), and convert them via the provided [`convert_safetensors.py`](convert_safetensors.py).

If you don't have python installed or don't want to, there is a pure rust [`converter`](https://github.com/cryscan/web-rwkv-converter).
You can clone that repo and run
```bash
$ cd /path/to/web-rwkv-converter
$ cargo run --release --example converter -- --input /path/to/model.pth
```

## Troubleshoot
- "thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: HeaderTooLarge'"
  
  Your model is broken, mainly because you cloned the repo but did not set up git-lfs.Please download the model manually and overwrite that one in `assets/models`.

- "thread 'main' panicked at 'Error in Queue::submit: parent device is lost'"

  Your GPU is not responding.
  Maybe you are running a model that is just too big for your device. If the model doesn't fit into your VRam, the driver needs to constantly swap and transfer the model parameters, causing it to be 10x slower.
  Try to quantize your model first.


## Credits
- Tokenizer is implemented by [@koute](https://github.com/koute/rwkv_tokenizer).
