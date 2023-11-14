@group(0) @binding(0) var<uniform> shape: vec4<u32>;                    // [B, N / B, M]
@group(0) @binding(1) var<uniform> quant: array<vec4<f32>, 4>;

@group(0) @binding(1) var<storage, read> input: array<vec4<u32>>;       // (M, N / B, B)
@group(0) @binding(2) var<storage, read_write> bound: array<f32>;       // (M, N / B)
@group(0) @binding(3) var<storage, read_write> output: array<u32>;      // (M, N)

const BLOCK_SIZE: u32 = 32u;

var<workgroup> sketch: array<array<vec4<f32>, 2>, BLOCK_SIZE>;

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn reduce_max(index: u32, stride: u32) {
    if index < stride {
        sketch[index][0] = max(sketch[index][0], sketch[index + stride][0]);
        sketch[index][1] = max(sketch[index][1], sketch[index + stride][1]);
    }
}

struct Input {
    @builtin(workgroup_id) bid: vec3<u32>,
    @builtin(global_invocation_id) uid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
};

@compute @workgroup_size(32, 1, 1)
fn quantize(in: Input) {}