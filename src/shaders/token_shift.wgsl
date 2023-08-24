struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
};

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]
@group(0) @binding(1) var<uniform> view: View;                              // [C, _, B] / [C, 5L, B]

@group(0) @binding(2) var<storage, read> time_mix: array<vec2<u32>>;        // (C)
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;               // (B, T, C)
@group(0) @binding(4) var<storage, read> sx: array<vec4<f32>>;              // (B, C)
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, C)

const BLOCK_SIZE: u32 = 128u;

fn compute_index(batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x / 4u;
    let offset = view.offset.x / 4u;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

@compute @workgroup_size(128, 1, 1)
fn token_shift(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index >= stride || token >= shape[1] || batch >= shape[2] {
        return;
    }

    let bti = (batch * shape[1] + token) * stride + index;
    if token == 0u {
        output[bti] = mix(sx[compute_index(batch, 0u, index)], x[bti], unpack4x16float(time_mix[index]));
    } else {
        output[bti] = mix(x[bti - stride], x[bti], unpack4x16float(time_mix[index]));
    }
}