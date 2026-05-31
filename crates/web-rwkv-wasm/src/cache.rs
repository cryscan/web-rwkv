use qp_trie::Trie;
use web_rwkv::tensor::TensorCpu;

#[repr(transparent)]
#[derive(Debug, Default, Clone)]
struct Tokens(Vec<u32>);

impl std::ops::Deref for Tokens {
    type Target = TokenSlice;

    fn deref(&self) -> &Self::Target {
        self.0.as_token_slice()
    }
}

impl std::borrow::Borrow<[u8]> for Tokens {
    fn borrow(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

impl std::borrow::Borrow<[u32]> for Tokens {
    fn borrow(&self) -> &[u32] {
        &self.0
    }
}

impl std::borrow::Borrow<TokenSlice> for Tokens {
    fn borrow(&self) -> &TokenSlice {
        self.0[..].as_token_slice()
    }
}

impl qp_trie::Break for Tokens {
    type Split = TokenSlice;

    fn empty<'a>() -> &'a Self::Split {
        Default::default()
    }

    fn find_break(&self, loc: usize) -> &Self::Split {
        self.0[..loc >> 2].as_token_slice()
    }
}

#[repr(transparent)]
struct TokenSlice([u32]);

impl std::ops::Deref for TokenSlice {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::borrow::Borrow<[u8]> for TokenSlice {
    fn borrow(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

impl Default for &TokenSlice {
    fn default() -> Self {
        <&[u32]>::default().as_token_slice()
    }
}

trait AsTokenSlice {
    fn as_token_slice(&self) -> &TokenSlice;
}

impl AsTokenSlice for [u32] {
    fn as_token_slice(&self) -> &TokenSlice {
        let ptr = self as *const [u32] as *const TokenSlice;
        // SAFETY: `TokenSlice` is `#[repr(transparent)]` over `[u32]`, so it has an
        // identical layout (including the slice length metadata in the fat pointer).
        // The cast preserves the pointer and length, and the resulting reference
        // borrows `self` for the same lifetime, so no aliasing or dangling occurs.
        unsafe { &*ptr }
    }
}

#[derive(Debug, Default)]
pub struct Cache(Trie<Tokens, CachedItem>);

#[derive(Debug, Clone)]
pub struct CachedItem {
    pub state: TensorCpu<f32>,
    pub output: TensorCpu<f32>,
}

#[derive(Debug, Clone)]
pub struct CacheCheckout<'a> {
    pub prefix: &'a [u32],
    pub suffix: &'a [u32],
    pub item: Option<CachedItem>,
}

impl Cache {
    pub fn checkout<'b>(&self, tokens: &'b [u32]) -> CacheCheckout<'b> {
        let cache = &self.0;

        let prefix = cache.longest_common_prefix(tokens.as_token_slice());
        let len = (1..=prefix.len())
            .rev()
            .find(|len| cache.contains_key(prefix[0..*len].as_token_slice()))
            .unwrap_or_default();
        let prefix = &tokens[0..len];

        match cache.get(prefix.as_token_slice()).cloned() {
            Some(item) => CacheCheckout {
                prefix: &tokens[..len],
                suffix: &tokens[len..],
                item: Some(item),
            },
            None => CacheCheckout {
                prefix: &[],
                suffix: tokens,
                item: None,
            },
        }
    }

    pub fn insert(&mut self, tokens: &[u32], state: TensorCpu<f32>, output: TensorCpu<f32>) {
        let tokens = Tokens(tokens.to_vec());
        self.0.insert(tokens, CachedItem { state, output });
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}
