@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

@group(0) @binding(1) var<storage, read> tokens: array<u32>;                // (B, T)
@group(0) @binding(2) var<storage, read> input: array<vec2<u32>>;           // (V, C)
#ifdef FP16
@group(0) @binding(3) var<storage, read_write> output: array<vec2<u32>>;    // (B, T, C)
#else
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, C)
#endif

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn embed(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let fetch = tokens[batch * shape[1] + token];

    if index < stride {
        let bti = (batch * shape[1] + token) * stride + index;
        let bei = fetch * stride + index;

#ifdef FP16
        output[bti] = input[bei];
#else
        output[bti] = unpack4x16float(input[bei]);
#endif
    }
}