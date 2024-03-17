use proc_macro2::TokenStream;
use quote::quote;
use syn::{spanned::Spanned, Data};

pub fn expand_derive_js_error(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    match input.data {
        Data::Struct(_) | Data::Enum(_) => {
            let name = input.ident;
            let generics = input.generics;
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            let ty_into = quote!(wasm_bindgen::JsValue);
            Ok(quote! {
                impl #impl_generics Into<#ty_into> for #name #ty_generics #where_clause  {
                    fn into(self) -> #ty_into {
                        let err: wasm_bindgen::JsError = self.into();
                        err.into()
                    }
                }
            })
        }
        _ => Err(syn::Error::new(
            input.span(),
            "expected a struct or an enum",
        )),
    }
}
