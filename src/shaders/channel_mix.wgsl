struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
};

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]
@group(0) @binding(1) var<uniform> view: View;                              // [C, 1, B] / [C, 5L, B]
@group(0) @binding(2) var<uniform> mask: u32;                               // [B]

@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;               // (B, T, C)
@group(0) @binding(4) var<storage, read> r: array<vec4<f32>>;               // (B, T, C)
@group(0) @binding(5) var<storage, read> v: array<vec4<f32>>;               // (B, T, C)

@group(0) @binding(6) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, C)
@group(0) @binding(7) var<storage, read_write> state: array<vec4<f32>>;     // (B, C)

const BLOCK_SIZE: u32 = 128u;

fn compute_index(batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x / 4u;
    let offset = view.offset.x / 4u;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

fn batch_masked(batch: u32) -> bool {
    return ((mask >> batch) & 1u) == 0u;
}

@compute @workgroup_size(128, 1, 1)
fn channel_mix(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index >= stride || token >= shape[1] || batch >= shape[2] {
        return;
    }

    let bti = (batch * shape[1] + token) * stride + index;

    if !batch_masked(batch) && token == shape[1] - 1u {
        state[compute_index(batch, 0u, index)] = x[bti];
    }

    let rr = 1.0 / (1.0 + exp(-r[bti]));
    output[bti] = rr * v[bti];
}