@group(0) @binding(0) var<uniform> num_emb: u32;
@group(0) @binding(1) var<storage, read> time_mix: array<vec2<u32>>;        // (C)
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;               // (T, C)
@group(0) @binding(3) var<storage, read> sx: array<vec4<f32>>;              // (C)
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;    // (T, C)

const BLOCK_SIZE: u32 = 128u;

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

@compute @workgroup_size(128, 1, 1)
fn token_shift(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let stride = num_emb / 4u;
    let num_tokens = num_blocks.y;

    if index < stride {
        let ti = token * stride + index;
        if token == 0u {
            output[ti] = mix(sx[index], x[ti], unpack4x16float(time_mix[index]));
        } else {
            output[ti] = mix(x[ti - stride], x[ti], unpack4x16float(time_mix[index]));
        }
    }
}