struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
};

@group(0) @binding(0) var<uniform> source: View;
@group(0) @binding(1) var<uniform> destination: View;

@group(0) @binding(2) var<storage, read> input: array<vec4<f32>>;           // (B, T, C)
@group(0) @binding(3) var<storage, read_write> output: array<vec2<u32>>;    // (B, T, C)

const BLOCK_SIZE: u32 = 128u;

fn compute_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x / 4u;
    let offset = view.offset.x / 4u;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

@compute @workgroup_size(128, 1, 1)
fn quantize(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = destination.shape.x / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index < stride {
        let x = input[compute_index(source, batch, token, index)];
        output[compute_index(destination, batch, token, index)] = pack4x16float(x);
    }
}
