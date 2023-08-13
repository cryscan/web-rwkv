@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;               // (B, T, C)
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, C)

const BLOCK_SIZE: u32 = 128u;

@compute @workgroup_size(128, 1, 1)
fn activation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if token >= shape[1] || batch >= shape[2] {
        return;
    }

    if index < stride {
        let bti = (batch * shape[1] + token) * stride + index;
        let p = max(x[bti], vec4<f32>(0.0));
        output[bti] = p * p;
    }
}