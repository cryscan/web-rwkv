use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use derive_getters::Getters;
use std::collections::BTreeMap;
use thiserror::Error;
use wasm_bindgen::{prelude::wasm_bindgen, JsError};

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to parse vocabulary: {0}")]
    FailedToParseVocabulary(serde_json::Error),
    #[error("no matching token found")]
    NoMatchingTokenFound,
    #[error("out of range token: {0}")]
    OutOfRangeToken(u32),
}

#[derive(Debug, Clone, Getters)]
pub struct Tokenizer {
    first_bytes_to_lengths: Vec<Box<[u16]>>,
    bytes_to_token_index: HashMap<Vec<u8>, u32>,
    token_index_to_bytes: Vec<Vec<u8>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum StrOrBytes {
    Str(String),
    Bytes(Vec<u8>),
}

impl Tokenizer {
    pub fn new(vocab: &str) -> Result<Self, TokenizerError> {
        let map: BTreeMap<u32, StrOrBytes> = 
            serde_json::from_str(vocab).map_err(TokenizerError::FailedToParseVocabulary)?;
        
        let list: Vec<(Vec<u8>, u32)> = map
            .into_iter()
            .map(|(token, pattern)| {
                let pattern = match pattern {
                    StrOrBytes::Str(string) => string.into_bytes(),
                    StrOrBytes::Bytes(bytes) => bytes,
                };
                (pattern, token)
            })
            .collect();

        let mut first_bytes_to_len = Vec::new();
        first_bytes_to_len.resize(u16::MAX as usize, 2);

        let mut first_bytes_to_lengths = Vec::new();
        first_bytes_to_lengths.resize(u16::MAX as usize, {
            let mut set = HashSet::new();
            set.insert(1);
            set
        });

        let mut token_index_to_bytes = Vec::new();
        // Find the max token index to determine the size of the vector.
        let max_token_index = list.iter().map(|(_, index)| *index).max().unwrap_or(0) as usize;
        token_index_to_bytes.resize_with(max_token_index + 1, Vec::new);

        let mut bytes_to_token_index = HashMap::new();
        for (token_bytes, token_index) in list {
            if token_bytes.len() >= 2 {
                let key = u16::from_ne_bytes([token_bytes[0], token_bytes[1]]) as usize;
                let max_length = &mut first_bytes_to_len[key];
                if token_bytes.len() > *max_length {
                    *max_length = token_bytes.len();
                }

                first_bytes_to_lengths[key].insert(token_bytes.len() as u16);
            }

            bytes_to_token_index.insert(token_bytes.clone(), token_index);
            token_index_to_bytes[token_index as usize] = token_bytes;
        }

        let first_bytes_to_lengths: Vec<Box<[_]>> = first_bytes_to_lengths
            .into_iter()
            .map(|inner| {
                let mut inner: Vec<_> = inner.into_iter().collect();
                inner.sort_unstable_by_key(|l| !*l);
                inner.into_boxed_slice()
            })
            .collect();

        Ok(Tokenizer {
            first_bytes_to_lengths,
            bytes_to_token_index,
            token_index_to_bytes,
        })
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        self.encode_into(input, &mut output)?;
        Ok(output)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::with_capacity(tokens.len());
        self.decode_into(tokens, &mut output)?;
        Ok(output)
    }
}

impl Tokenizer {
    pub fn encode_into(
        &self,
        mut input: &[u8],
        output: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        'next_token: while !input.is_empty() {
            let lengths = if input.len() >= 2 {
                let key = u16::from_ne_bytes([input[0], input[1]]) as usize;
                &self.first_bytes_to_lengths[key][..]
            } else {
                &[1][..]
            };

            for &length in lengths {
                let length = length as usize;
                if length > input.len() {
                    continue;
                }

                if let Some(&token_index) = self.bytes_to_token_index.get(&input[..length]) {
                    output.push(token_index);
                    input = &input[length..];
                    continue 'next_token;
                }
            }

            return Err(TokenizerError::NoMatchingTokenFound);
        }

        Ok(())
    }

    pub fn decode_into(&self, tokens: &[u32], output: &mut Vec<u8>) -> Result<(), TokenizerError> {
        for &token in tokens {
            let bytes = self
                .token_index_to_bytes
                .get(token as usize)
                .ok_or(TokenizerError::OutOfRangeToken(token))?;

            output.extend_from_slice(bytes);
        }

        Ok(())
    }
}

#[wasm_bindgen(js_name = Tokenizer)]
pub struct JsTokenizer(Tokenizer);

#[wasm_bindgen(js_class = Tokenizer)]
impl JsTokenizer {
    #[wasm_bindgen(constructor)]
    pub fn new(vocab: &str) -> Result<Self, JsError> {
        Ok(Self(Tokenizer::new(vocab)?))
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u32>, JsError> {
        Ok(self.0.encode(input)?)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, JsError> {
        Ok(self.0.decode(tokens)?)
    }
}