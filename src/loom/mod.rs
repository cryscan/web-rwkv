//! The *Loom* layout module.

#![cfg_attr(target_arch = "wasm32", allow(async_fn_in_trait))]

pub mod device;
pub mod layout;
pub mod num;
pub mod ops;
pub mod tensor;
