@group(0) @binding(0) var<uniform> num_emb: u32;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;               // (T, C)
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;    // (T, C)

const BLOCK_SIZE: u32 = 128u;

@compute @workgroup_size(128, 1, 1)
fn add(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let stride = num_emb / 4u;

    if index < stride {
        let ti = token * stride + index;
        output[ti] = x[ti] + output[ti];
    }
}