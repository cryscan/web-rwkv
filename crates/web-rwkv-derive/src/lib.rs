use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(Deref)]
pub fn derive_deref(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Unnamed(fields) => {
                if fields.unnamed.len() == 1 {
                    let name = input.ident;
                    let generics = input.generics;
                    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
                    let target_type = &fields.unnamed.first().unwrap().ty;
                    quote! {
                        impl #impl_generics std::ops::Deref for #name #ty_generics #where_clause {
                            type Target = #target_type;

                            fn deref(&self) -> &Self::Target {
                                &self.0
                            }
                        }
                    }
                    .into()
                } else {
                    panic!("Expected a tuple struct with one field");
                }
            }
            _ => panic!("Expected a tuple struct"),
        },
        _ => panic!("Expected a struct"),
    }
}

#[proc_macro_derive(DerefMut)]
pub fn derive_deref_mut(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Unnamed(fields) => {
                if fields.unnamed.len() == 1 {
                    let name = input.ident;
                    let generics = input.generics;
                    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
                    quote! {
                        impl #impl_generics std::ops::DerefMut for #name #ty_generics #where_clause {
                            fn deref_mut(&mut self) -> &mut Self::Target {
                                &mut self.0
                            }
                        }
                    }
                    .into()
                } else {
                    panic!("Expected a tuple struct with one field");
                }
            }
            _ => panic!("Expected a tuple struct"),
        },
        _ => panic!("Expected a struct"),
    }
}

#[proc_macro_derive(Id)]
pub fn derive_id(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    quote! {
        impl #name {
            fn new() -> Self {
                uid::Id::new().into()
            }
        }

        impl From<uid::Id<#name>> for #name {
            fn from(value: uid::Id<#name>) -> Self {
                Self(value.get())
            }
        }
    }
    .into()
}

#[proc_macro_derive(Kind, attributes(usage))]
pub fn derive_kind(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let mut usages = vec![];

    for attr in &ast.attrs {
        if attr.path().is_ident("usage") {
            attr.parse_nested_meta(|meta| match meta.path.get_ident() {
                Some(ident) => {
                    usages.push(ident.to_owned());
                    Ok(())
                }
                None => Err(meta.error("Unrecognized buffer usage")),
            })
            .unwrap();
        }
    }

    let usages = usages
        .into_iter()
        .fold(quote! { wgpu::BufferUsages::empty() }, |acc, usage| {
            quote! { #acc | wgpu::BufferUsages::#usage }
        });

    quote! {
        impl Kind for #name {
            fn buffer_usages() -> wgpu::BufferUsages {
                #usages
            }
        }
    }
    .into()
}
