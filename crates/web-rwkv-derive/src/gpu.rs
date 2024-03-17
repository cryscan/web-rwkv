use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::spanned::Spanned;

pub fn expand_derive_kind(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    let mut usages = vec![];
    for attr in &input.attrs {
        if attr.path().is_ident("usage") {
            attr.parse_nested_meta(|meta| match meta.path.get_ident() {
                Some(ident) => {
                    usages.push(ident.to_token_stream());
                    Ok(())
                }
                None => Err(meta.error("unrecognized buffer usage")),
            })?;
        }
    }
    let usages = usages
        .into_iter()
        .reduce(|acc, usage| quote! { #acc | wgpu::BufferUsages::#usage });

    let name = input.ident;
    Ok(quote! {
        impl Kind for #name {
            fn buffer_usages() -> wgpu::BufferUsages {
                #usages
            }
        }
    })
}

pub fn expand_derive_deserialize_seed(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    match input.data {
        syn::Data::Struct(data) => todo!(),
        syn::Data::Enum(data) => todo!(),
        _ => Err(syn::Error::new_spanned(
            input.into_token_stream(),
            "expect a struct or an enum",
        )),
    }
}
