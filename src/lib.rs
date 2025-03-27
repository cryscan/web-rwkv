//! # Web-RWKV
//!
//! This is an inference engine for the [language model of RWKV](https://github.com/BlinkDL/RWKV-LM) implemented in pure WebGPU.
//!
//! ## Features
//!
//! - No dependencies on CUDA/Python.
//! - Support Nvidia/AMD/Intel GPUs, including integrated GPUs.
//! - Vulkan/Dx12/OpenGL backends.
//! - Batched inference.
//! - Int8 and NF4 quantization.
//! - Very fast.
//! - LoRA merging at loading time.
//! - Support RWKV V4, V5 and V6.
//!
//! ## Notes
//!
//! Note that `web-rwkv` is only an inference engine. It only provides the following functionalities:
//! - A tokenizer.
//! - Model loading.
//! - State creation and updating.
//! - A `run` function that takes in prompt tokens and returns logits (predicted next token probabilities after calling `softmax`).
//!
//! It *does not* provide the following:
//! - OpenAI API or APIs of any kind.
//!   - If you would like to deploy an API server, check [AI00 RWKV Server](https://github.com/cgisky1980/ai00_rwkv_server) which is a fully-functional OpenAI-compatible API server built upon `web-rwkv`.
//!   - You could also check the [`web-rwkv-axum`](https://github.com/Prunoideae/web-rwkv-axum) project if you want some fancy inference pipelines, including Classifier-Free Guidance (CFG), Backus–Naur Form (BNF) guidance, and more.
//! - Samplers, though in the examples a basic nucleus sampler is implemented, this is *not* included in the library itself.
//! - State caching or management system.
//! - Python (or any other languages) binding.
//! - Runtime. Without a runtime makes it easy to be integrated into any applications from servers, front-end apps (yes, `web-rwkv` can run in browser) to game engines.
//!
//! ## Crate Features
//!
#![doc = document_features::document_features!()]

pub mod context;
pub mod loom;
pub mod num;
pub mod runtime;
pub mod tensor;
pub mod tokenizer;

pub use wgpu;
