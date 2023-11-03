@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;           // (B, T, C)
@group(0) @binding(2) var<storage, read_write> output: array<vec2<u32>>;    // (B, T, C)

const BLOCK_SIZE: u32 = 128u;

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

@compute @workgroup_size(128, 1, 1)
fn quantize(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index < stride {
        let bti = (batch * shape[1] + token) * stride + index;
        output[bti] = pack4x16float(input[bti]);
    }
}

@compute @workgroup_size(128, 1, 1)
fn quantize_half(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index < stride {
        let bti = (batch * shape[1] + token) * stride + index;
        output[bti] = pack4x16float(input[bti] * 0.5);
    }
}
