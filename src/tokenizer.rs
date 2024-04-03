use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use derive_getters::Getters;
use std::collections::BTreeMap;
use thiserror::Error;
use wasm_bindgen::prelude::wasm_bindgen;
use web_rwkv_derive::JsError;
use tokenizers::tokenizer::{Tokenizer as BaseHFTokenizer};

#[derive(Debug, Error, JsError)]
pub enum TokenizerError {
    #[error("failed to parse vocabulary: {0}")]
    FailedToParseVocabulary(serde_json::Error),
    #[error("no matching token found")]
    NoMatchingTokenFound,
    #[error("out of range token: {0}")]
    OutOfRangeToken(u16),
    #[error("HFTokenizer error: {0}")]
    HFTokenizerError(String),
}

#[wasm_bindgen]
#[derive(Debug, Clone, Getters)]
pub struct RWKVTokenizer {
    first_bytes_to_lengths: Vec<Box<[u16]>>,
    bytes_to_token_index: HashMap<Vec<u8>, u16>,
    token_index_to_bytes: Vec<Vec<u8>>,
}

#[derive(Debug)]
pub struct HFTokenizer {
    tokenizer: BaseHFTokenizer,
    bytes_to_token_index: HashMap<Vec<u8>, u16>,
}

#[derive(Debug)]
pub enum Tokenizer {
    RWKV(RWKVTokenizer),
    HF(HFTokenizer),
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum StrOrBytes {
    Str(String),
    Bytes(Vec<u8>),
}

impl HFTokenizer {
    pub fn new(json_str: &str) -> Result<Self, TokenizerError> {
        let tokenizer: BaseHFTokenizer = BaseHFTokenizer::from_bytes(json_str.as_bytes())
            .map_err(|e| TokenizerError::HFTokenizerError(e.to_string()))?;

        let bytes_to_token_index: HashMap<Vec<u8>, u16> = tokenizer
            .get_vocab(true)
            .iter()
            .map(|(key, value)| (key.as_bytes().to_vec(), *value as u16))
            .collect();      

        Ok(Self {
            tokenizer,
            bytes_to_token_index,
        })
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u16>, TokenizerError> {
        let text = std::str::from_utf8(input).expect("Failed to convert byte slice to UTF-8 string slice");
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| TokenizerError::HFTokenizerError(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| id as u16).collect())
    }

    pub fn decode(&self, tokens: &[u16]) -> Result<Vec<u8>, TokenizerError> {
        let decoded_text = self.tokenizer.decode(tokens.iter().map(|&t| t as u32).collect::<Vec<u32>>().as_slice(), true)
            .map_err(|e| TokenizerError::HFTokenizerError(e.to_string()))?;
        Ok(decoded_text.into_bytes())
    }
}

#[wasm_bindgen]
impl RWKVTokenizer {
    #[wasm_bindgen(constructor)]
    pub fn new(vocab: &str) -> Result<RWKVTokenizer, TokenizerError> {
        let map: BTreeMap<u16, StrOrBytes> =
            serde_json::from_str(vocab).map_err(TokenizerError::FailedToParseVocabulary)?;

        let list: Vec<(Vec<u8>, u16)> = map
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
        token_index_to_bytes.resize_with(u16::MAX as usize, Vec::new);

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

        Ok(RWKVTokenizer {
            first_bytes_to_lengths,
            bytes_to_token_index,
            token_index_to_bytes,
        })
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u16>, TokenizerError> {
        let mut output = Vec::new();
        self.encode_into(input, &mut output)?;
        Ok(output)
    }

    pub fn decode(&self, tokens: &[u16]) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::with_capacity(tokens.len());
        self.decode_into(tokens, &mut output)?;
        Ok(output)
    }
}

impl RWKVTokenizer {
    pub fn encode_into(
        &self,
        mut input: &[u8],
        output: &mut Vec<u16>,
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

    pub fn decode_into(&self, tokens: &[u16], output: &mut Vec<u8>) -> Result<(), TokenizerError> {
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


impl Tokenizer {
    pub fn new(json_str: &str) -> Result<Self, TokenizerError> {
        if json_str.contains("\"version\":") {
            let hf_tokenizer = HFTokenizer::new(json_str)?;
            Ok(Tokenizer::HF(hf_tokenizer))
        } else {
            let rwkv_tokenizer = RWKVTokenizer::new(json_str)?;
            Ok(Tokenizer::RWKV(rwkv_tokenizer))
        }
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u16>, TokenizerError> {
        match self {
            Tokenizer::HF(tokenizer) => tokenizer.encode(input),
            Tokenizer::RWKV(tokenizer) => tokenizer.encode(input)
        }
    }

    pub fn decode(&self, tokens: &[u16]) -> Result<Vec<u8>, TokenizerError> {
        match self {
            Tokenizer::HF(tokenizer) => tokenizer.decode(tokens),
            Tokenizer::RWKV(tokenizer) => tokenizer.decode(tokens)
        }
    }

    pub fn bytes_to_token_index(&self) -> &HashMap<Vec<u8>, u16> {
        match self {
            Tokenizer::HF(tokenizer) => &tokenizer.bytes_to_token_index,
            Tokenizer::RWKV(tokenizer) => &tokenizer.bytes_to_token_index,
        }
    }
}

