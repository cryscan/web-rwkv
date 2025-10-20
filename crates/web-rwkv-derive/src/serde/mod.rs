//! This crate provides Serde's derive macro for `DeserializeSeed`.

#![cfg_attr(not(check_cfg), allow(unexpected_cfgs))]
// Ignored clippy lints
#![allow(
    // clippy false positive: https://github.com/rust-lang/rust-clippy/issues/7054
    clippy::branches_sharing_code,
    clippy::cognitive_complexity,
    // clippy bug: https://github.com/rust-lang/rust-clippy/issues/7575
    clippy::collapsible_match,
    clippy::derive_partial_eq_without_eq,
    clippy::enum_variant_names,
    // clippy bug: https://github.com/rust-lang/rust-clippy/issues/6797
    clippy::manual_map,
    clippy::match_like_matches_macro,
    clippy::needless_lifetimes,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::trivially_copy_pass_by_ref,
    clippy::used_underscore_binding,
    clippy::wildcard_in_or_patterns,
    // clippy bug: https://github.com/rust-lang/rust-clippy/issues/5704
    clippy::unnested_or_patterns,
)]
// Ignored clippy_pedantic lints
#![allow(
    clippy::cast_possible_truncation,
    clippy::checked_conversions,
    clippy::doc_markdown,
    clippy::elidable_lifetime_names,
    clippy::enum_glob_use,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::let_underscore_untyped,
    clippy::manual_assert,
    clippy::map_err_ignore,
    clippy::match_same_arms,
    // clippy bug: https://github.com/rust-lang/rust-clippy/issues/6984
    clippy::match_wildcard_for_single_variants,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::unseparated_literal_suffix,
    clippy::unused_self,
    clippy::use_self,
    clippy::wildcard_imports
)]
#![cfg_attr(all(test, exhaustive), feature(non_exhaustive_omitted_patterns_lint))]
#![allow(unknown_lints, mismatched_lifetime_syntaxes)]
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
