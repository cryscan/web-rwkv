struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

@group(0) @binding(0) var<uniform> source: View;
@group(0) @binding(1) var<uniform> destination: View;
@group(0) @binding(2) var<uniform> factor: vec4<f32>;

#ifdef IN_FP16
@group(0) @binding(3) var<storage, read> input: array<vec2<u32>>;           // (B, T, C)
#else
@group(0) @binding(3) var<storage, read> input: array<vec4<f32>>;           // (B, T, C)
#endif
#ifdef OUT_FP16
@group(0) @binding(4) var<storage, read_write> output: array<vec2<u32>>;    // (B, T, C)
#else
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, C)
#endif

fn compute_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x >> 2u;
    let offset = vec3<u32>(view.offset.zy, view.offset.x >> 2u);
    return dot(vec3<u32>(batch, token, index) + offset, vec3<u32>(view.stride.y * stride, stride, 1u));
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn blend(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = destination.shape.x / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index < stride {
#ifdef IN_FP16
        let x = unpack4x16float(input[compute_index(source, batch, token, index)]);
#else
        let x = input[compute_index(source, batch, token, index)];
#endif
        let bti = compute_index(destination, token, batch, index);
#ifdef OUT_FP16
        let y = unpack4x16float(output[bti]);
        output[bti] = pack4x16float(factor.x * x + factor.y * y);
#else
        let y = output[bti];
        output[bti] = factor.x * x + factor.y * y;
#endif
    }
}
