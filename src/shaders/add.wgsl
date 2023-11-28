struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
};

@group(0) @binding(0) var<uniform> source: View;
@group(0) @binding(1) var<uniform> destination: View;

@group(0) @binding(2) var<storage, read> input: array<vec4<f32>>;           // (B, T, C)
@group(0) @binding(3) var<storage, read> input_fp16: array<vec2<u32>>;      // (B, T, C)
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, C)

const BLOCK_SIZE: u32 = 128u;

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn compute_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x / 4u;
    let offset = view.offset.x / 4u;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

fn fetch_input(batch: u32, token: u32, index: u32) -> vec4<f32> {
    let _token = select(token, 0u, source.shape.y == 1u);
    return input[compute_index(source, batch, _token, index)];
}

@compute @workgroup_size(128, 1, 1)
fn add(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = destination.shape.x / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index < stride {
        let x = fetch_input(batch, token, index);
        let bti = compute_index(destination, batch, token, index);
        output[bti] = x + output[bti];
    }
}

fn fetch_input_fp16(batch: u32, token: u32, index: u32) -> vec4<f32> {
    let _token = select(token, 0u, source.shape.y == 1u);
    return unpack4x16float(input_fp16[compute_index(source, batch, _token, index)]);
}

@compute @workgroup_size(128, 1, 1)
fn add_fp16(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = destination.shape.x / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if index < stride {
        let x = fetch_input_fp16(batch, token, index);
        let bti = compute_index(destination, batch, token, index);
        output[bti] = x + output[bti];
    }
}