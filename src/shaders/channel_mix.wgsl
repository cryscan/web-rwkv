@group(0) @binding(0) var<uniform> num_emb: u32;
@group(0) @binding(1) var<uniform> num_tokens: u32;

@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;               // (T, C)
@group(0) @binding(3) var<storage, read> r: array<vec4<f32>>;               // (T, C)
@group(0) @binding(4) var<storage, read> v: array<vec4<f32>>;               // (T, C)
@group(0) @binding(5) var<storage, read_write> state: array<vec4<f32>>;     // (C)
@group(0) @binding(6) var<storage, read_write> output: array<vec4<f32>>;    // (T, C)

const BLOCK_SIZE: u32 = 128u;

@compute @workgroup_size(128, 1, 1)
fn channel_mix(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let stride = num_emb / 4u;

    if token >= num_tokens {
        return;
    }

    if index < stride {
        let ti = token * stride + index;
        let rr = 1.0 / (1.0 + exp(-r[ti]));
        output[ti] = rr * v[ti];

        if token == num_tokens - 1u {
            state[index] = x[ti];
        }
    }
}