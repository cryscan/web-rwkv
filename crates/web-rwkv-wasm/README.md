# web-rwkv-wasm

Browser-ready [`wasm-bindgen`](https://github.com/rustwasm/wasm-bindgen) bindings for
[**web-rwkv**](https://github.com/cryscan/web-rwkv) — a pure-WebGPU implementation of the
RWKV language model. This crate compiles to a self-contained WebAssembly module and is
published to npm so a web app can run RWKV inference locally in the browser with a plain
`pnpm add …` instead of setting up a Rust/`wasm-pack` toolchain.

The binding surface is upstreamed from the official demo
[`web-rwkv-puzzles`](https://github.com/cryscan/web-rwkv-puzzles), so it is the same API
that powers the live demos — only the packaging differs (`--target web` ESM instead of
the demo's `--target no-modules` global).

## Requirements

- A browser with **WebGPU** (`navigator.gpu`) — Chrome/Edge 113+, or Firefox/Safari with
  WebGPU enabled. There is **no** SharedArrayBuffer / threads requirement, so you do **not**
  need cross-origin isolation (COOP/COEP headers).
- Models are **f16 `safetensors`** files (RWKV v4/v5/v6/v7), or a CBOR "prefab".
- A tokenizer **vocab JSON** (e.g. `rwkv_vocab_v20230424.json`).

Inference runs on the GPU; the wasm module only orchestrates and moves tensors across the
JS↔GPU boundary as typed arrays (`Uint32Array` tokens, `Float32Array` logits/state).

## Install

```sh
pnpm add @cryscan/web-rwkv-wasm
```

## API surface

The generated `.d.ts` is authoritative; this is the overview.

```ts
// init (target web): default export instantiates the wasm module
export default function init(module_or_path?: …): Promise<InitOutput>

class Tensor        { constructor(name: string, shape: Uint32Array | number[], buffer: ArrayBuffer) }
class TensorReader  { constructor(tensors: Tensor[]) }      // implements web-rwkv's Reader

enum SessionType    { Puzzle, Chat, Music, Othello }        // numeric in JS

class Session {
  // static async factories — `await Session.from_reader(...)` / `await Session.from_prefab(...)`
  static from_reader(model: TensorReader, quant: number, quant_nf4: number, quant_sf4: number, ty: SessionType): Promise<Session>
  static from_prefab(data: Uint8Array, ty: SessionType): Promise<Session>

  run(tokens: Uint32Array, output: Float32Array): Promise<void>   // last-token logits -> output (len = num_vocab)
  softmax(input: Float32Array, output: Float32Array): Promise<void>

  info(): ModelInfo            // { num_vocab, num_layer, num_head, num_emb, version, ... }
  session_type(): SessionType
  state_len(): number          // flat f32 length of the RNN state

  back(state: Float32Array): Promise<void>   // read current GPU state (len = state_len())
  load(state: Float32Array): void            // upload a state vector

  // built-in prefix cache (reuse the state/logits of a shared token prefix)
  checkout(tokens: Uint32Array, state: Float32Array, output: Float32Array): number  // matched prefix length
  cache(tokens: Uint32Array, state: Float32Array, output: Float32Array): void
  clear_cache(): void
}

class SimpleSampler  { constructor(info: ModelInfo); update(t: Uint32Array): void; transform(l: Float32Array): void; sample(p: Float32Array): number }  // argmax
class NucleusSampler { constructor(info: ModelInfo, temp: number, top_p: number, presence_penalty: number, count_penalty: number, penalty_decay: number); /* + mutable fields */ update; transform; sample }

class StateVisual    { constructor(info: ModelInfo, state: Float32Array); json(): string }  // state heatmaps as base64 PNGs

// re-exported from the web-rwkv crate itself:
class Tokenizer      { constructor(vocab: string); encode(input: Uint8Array): Uint32Array; decode(tokens: Uint32Array): Uint8Array }
```

Notes:
- `Session` quantization: `quant` = number of `Int8` layers, `quant_nf4` = `NF4` layers,
  `quant_sf4` = `SF4` layers (apply from layer 0). Pass `0, 0, 0` for full f16.
- `NucleusSampler` "temperature" is applied as `prob^(1/temp)`; repetition penalties are
  `presence_penalty + count_penalty * count`, with `count` decayed by `penalty_decay`
  each step.

## Usage (Web Worker)

Because compute is single-threaded JS driving the GPU, run it in a Worker to keep the UI
responsive. The generation loop lives in your TS (this package ships the raw bindings).

```ts
// worker.ts  (a module worker: new Worker(url, { type: 'module' }))
import init, {
  Session, SessionType, TensorReader, Tensor, NucleusSampler, Tokenizer,
} from '@cryscan/web-rwkv-wasm'

// Build a TensorReader by parsing a .safetensors ArrayBuffer (all tensors assumed f16).
function readerFromSafetensors(buffer: ArrayBuffer): TensorReader {
  const view = new DataView(buffer)
  const headerLen = Number(view.getBigUint64(0, true))
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen)))
  const base = 8 + headerLen
  const tensors: Tensor[] = []
  for (const [name, info] of Object.entries<any>(header)) {
    if (name === '__metadata__') continue
    const [start, end] = info.data_offsets
    tensors.push(new Tensor(name, info.shape, buffer.slice(base + start, base + end)))
  }
  return new TensorReader(tensors)
}

let session: Session
let tokenizer: Tokenizer
let sampler: NucleusSampler

async function load(modelBytes: ArrayBuffer, vocabJson: string) {
  await init()                                   // instantiate the wasm module
  const reader = readerFromSafetensors(modelBytes)
  session = await Session.from_reader(reader, 0, 0, 0, SessionType.Chat)
  tokenizer = new Tokenizer(vocabJson)
  sampler = new NucleusSampler(session.info(), /*temp*/ 1.0, /*top_p*/ 0.5, 0.4, 0.4, 0.996)
}

async function* generate(prompt: string, maxTokens = 256, stop: number[] = [0]) {
  const info = session.info()
  const output = new Float32Array(info.num_vocab)
  const probs = new Float32Array(info.num_vocab)
  const state = new Float32Array(session.state_len())

  let tokens = tokenizer.encode(new TextEncoder().encode(prompt))

  // reuse cached prefix state, if any
  const cutoff = session.checkout(tokens, state, output)
  session.load(state)
  const history = tokens
  tokens = tokens.slice(cutoff)

  for (let i = 0; i < maxTokens; i++) {
    if (tokens.length > 0) await session.run(tokens, output)
    sampler.transform(output)
    await session.softmax(output, probs)
    const token = sampler.sample(probs)
    if (stop.includes(token)) break
    sampler.update(Uint32Array.of(token))
    tokens = Uint32Array.of(token)
    yield new TextDecoder().decode(tokenizer.decode(tokens))
  }

  // repopulate the prefix cache for the next turn
  await session.back(state)
  session.cache(history, state, output)
}
```

Fetch the model and vocab yourself (e.g. with the Cache Storage API) and pass the
`ArrayBuffer` / vocab string into `load()`. If `Session.from_reader(reader, …)` throws on
a file that is actually a CBOR prefab, fall back to
`Session.from_prefab(new Uint8Array(buffer), ty)`.

## Building locally

```sh
# from this directory (crates/web-rwkv-wasm)
./build.bash               # -> ./pkg  (ESM, --target web)
./build.bash --scope cryscan   # -> name "@cryscan/web-rwkv-wasm"
```

Requires `wasm-pack` and the `wasm32-unknown-unknown` target
(`rustup target add wasm32-unknown-unknown`). `wasm-pack` downloads `wasm-opt` itself.

## License

Dual-licensed under either of MIT or Apache-2.0 at your option, matching the parent
`web-rwkv` crate. See [LICENSE](./LICENSE).
