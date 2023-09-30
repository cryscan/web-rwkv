@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;           // (B, T, C)
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, C)

const BLOCK_SIZE: u32 = 128u;

fn sigmoid(x: vec4<f32>) -> vec4<f32> {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(128, 1, 1)
fn silu(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index < stride {
        let bti = (batch * shape[1] + token) * stride + index;
        let x = input[bti];
        let s = select(1.0 - sigmoid(-x), sigmoid(x), x >= vec4<f32>(0.0));
        output[bti] = x * s * output[bti];
    }
}