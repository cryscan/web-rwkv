@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]
@group(0) @binding(1) var<uniform> extent: vec4<u32>;                       // [M, N, K]
@group(0) @binding(2) var<uniform> offset: vec4<u32>;

@group(0) @binding(3) var<storage, read> input: array<vec4<f32>>;           // (B, T, C)
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;    // (K, N, M)

const BLOCK_SIZE: u32 = 128u;

@compute @workgroup_size(128, 1, 1)
fn blit(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index >= stride || token >= shape[1] || batch >= shape[2] {
        return;
    }

    let bti = (batch * shape[1] + token) * stride + index;
    let o_bti = ((batch + offset.z) * extent.y + token + offset.y) * (extent.x / 4u) + index + offset.x / 4u;
    output[o_bti] = input[bti];
}