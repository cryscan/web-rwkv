use proc_macro2::TokenStream;
use quote::quote;

pub fn expand_derive_kind(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    let mut usages = vec![];
    for attr in &input.attrs {
        if attr.path().is_ident("usage") {
            attr.parse_nested_meta(|meta| match meta.path.get_ident() {
                Some(ident) => {
                    usages.push(ident.clone());
                    Ok(())
                }
                None => Err(meta.error("unrecognized buffer usage")),
            })?;
        }
    }
    let usages = usages.into_iter().fold(
        quote!(wgpu::BufferUsages::empty()),
        |acc, ident| quote!(#acc | wgpu::BufferUsages::#ident),
    );

    let name = input.ident;
    Ok(quote! {
        impl Kind for #name {
            fn buffer_usages() -> wgpu::BufferUsages {
                #usages
            }
        }
    })
}
