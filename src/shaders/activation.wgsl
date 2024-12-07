struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

@group(0) @binding(0) var<uniform> view: View;                          // [C, T, B]
#ifdef FP16
@group(0) @binding(1) var<storage, read_write> x: array<vec2<u32>>;     // (B, T, C)
#else
@group(0) @binding(1) var<storage, read_write> x: array<vec4<f32>>;     // (B, T, C)
#endif

fn squared_relu(x: vec4<f32>) -> vec4<f32> {
    let p = max(x, vec4<f32>(0.0));
    return p * p;
}

fn stable_exp(x: vec4<f32>) -> vec4<f32> {
    return exp(-exp(x));
}

fn opposite_exp(x: vec4<f32>) -> vec4<f32> {
    return -exp(x);
}

fn softplus(x: vec4<f32>) -> vec4<f32> {
    return log(1.0 + exp(x));
}

fn sigmoid(x: vec4<f32>) -> vec4<f32> {
    return 1.0 / (1.0 + exp(-x));
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn compute_index(batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x >> 2u;
    let offset = vec3<u32>(view.offset.zy, view.offset.x >> 2u);
    return dot(vec3<u32>(batch, token, index) + offset, vec3<u32>(view.stride.y * stride, stride, 1u));
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn act(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = view.shape.x / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index < stride {
        let bti = compute_index(batch, token, index);
#ifdef FP16
        let input = unpack4x16float(x[bti]);
        x[bti] = pack4x16float(ACT(input));
#else
        x[bti] = ACT(x[bti]);
#endif
    }
}
