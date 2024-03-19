use proc_macro2::TokenStream;
use quote::quote;
use syn::{spanned::Spanned, Data, Fields};

pub fn expand_derive_deref(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Unnamed(fields) => {
                if fields.unnamed.len() == 1 {
                    let name = input.ident;
                    let generics = input.generics;
                    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
                    let target_type = &fields.unnamed.first().unwrap().ty;
                    Ok(quote! {
                        impl #impl_generics std::ops::Deref for #name #ty_generics #where_clause {
                            type Target = #target_type;

                            fn deref(&self) -> &Self::Target {
                                &self.0
                            }
                        }
                    })
                } else {
                    Err(syn::Error::new(
                        fields.span(),
                        "expect a tuple struct with one field",
                    ))
                }
            }
            _ => Err(syn::Error::new(data.fields.span(), "expect a tuple struct")),
        },
        _ => Err(syn::Error::new(input.span(), "expect a struct")),
    }
}

pub fn expand_derive_deref_mut(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Unnamed(fields) => {
                if fields.unnamed.len() == 1 {
                    let name = input.ident;
                    let generics = input.generics;
                    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
                    Ok(quote! {
                        impl #impl_generics std::ops::DerefMut for #name #ty_generics #where_clause {
                            fn deref_mut(&mut self) -> &mut Self::Target {
                                &mut self.0
                            }
                        }
                    })
                } else {
                    Err(syn::Error::new(
                        fields.span(),
                        "expect a tuple struct with one field",
                    ))
                }
            }
            _ => Err(syn::Error::new(data.fields.span(), "expect a tuple struct")),
        },
        _ => Err(syn::Error::new(input.span(), "expect a struct")),
    }
}
