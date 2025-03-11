use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

mod deref;
mod kind;
mod serde;

#[proc_macro_derive(Deref)]
pub fn derive_deref(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    deref::expand_derive_deref(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_derive(DerefMut)]
pub fn derive_deref_mut(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    deref::expand_derive_deref_mut(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_derive(Kind, attributes(usage))]
pub fn derive_kind(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    kind::expand_derive_kind(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_derive(DeserializeSeed, attributes(serde_seed))]
pub fn derive_deserialize_seed(input: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);
    serde::de::expand_derive_deserialize(&mut input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
