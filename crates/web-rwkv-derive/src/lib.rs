use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Fields, Ident, Meta};

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
                    let expanded = quote! {
                        impl #impl_generics std::ops::Deref for #name #ty_generics #where_clause {
                            type Target = #target_type;

                            fn deref(&self) -> &Self::Target {
                                &self.0
                            }
                        }
                    };
                    TokenStream::from(expanded)
                } else {
                    panic!("Expected a tuple struct with one field");
                }
            }
            _ => panic!("Expected a tuple struct"),
        },
        _ => panic!("Expected a struct"),
    }
}

#[proc_macro_derive(TensorBuffer, attributes(data_type))]
pub fn derive_tensor_buffer(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let name = &ast.ident;
    let data_type = get_data_type(&ast.attrs);
    let (buffer_return, type_lt, trait_lt, ref_lt) = match &ast.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Unnamed(fields_unnamed) => {
                let field = &fields_unnamed.unnamed.first().unwrap();
                let field_type = &field.ty;
                let field_type_string = field_type.to_token_stream().to_string();
                if field_type_string.contains("Arc") {
                    (
                        quote! { Ok(&self.0) },
                        quote! {},
                        quote! { <'_> },
                        quote! { '_ },
                    )
                } else if field_type_string.contains("Cow") {
                    (
                        quote! { Err(TensorError::DeviceError) },
                        quote! { <'a> },
                        quote! { <'a> },
                        quote! { 'a },
                    )
                } else {
                    panic!("Only supports Arc and Cow types")
                }
            }
            _ => panic!("Only supports unnamed fields"),
        },
        _ => panic!("Only supports structs"),
    };

    TokenStream::from(quote! {
        impl #type_lt TensorBuffer #trait_lt for #name #type_lt {
            fn data_type(&#ref_lt self) -> DataType {
                DataType::#data_type
            }
            fn buffer(&#ref_lt self) -> Result<&#ref_lt Buffer, TensorError> {
                #buffer_return
            }
        }
    })
}

fn get_data_type(attrs: &[Attribute]) -> Ident {
    for attr in attrs {
        if attr.path().is_ident("data_type") {
            match attr.parse_args::<Meta>() {
                Ok(Meta::Path(path)) => {
                    return path
                        .get_ident()
                        .expect("Expected an identifier for data_type attribute")
                        .clone()
                }
                _ => panic!("Expected an identifier for data_type attribute"),
            }
        }
    }
    panic!("data_type attribute not found");
}
