#![allow(dead_code)]

extern crate proc_macro2;
extern crate quote;
extern crate syn;

extern crate proc_macro;

use proc_macro2::{Ident, Span};
use quote::{ToTokens, TokenStreamExt as _};

mod internals;
#[macro_use]
mod bound;
#[macro_use]
mod fragment;
mod deprecated;
mod dummy;
mod pretend;
mod this;

pub mod de;

#[allow(non_camel_case_types)]
struct private;

impl private {
    fn ident(&self) -> Ident {
        Ident::new(
            concat!("__private", env!("SERDE_PATCH_VERSION")),
            Span::call_site(),
        )
    }
}

impl ToTokens for private {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        tokens.append(self.ident());
    }
}
