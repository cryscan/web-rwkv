struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
};

struct Cursor {
    batch: u32,
    token: u32,
    len: u32,
};

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                    // [S, H, A]
@group(0) @binding(1) var<uniform> view: View;                          // [S, S, H]
@group(0) @binding(2) var<storage, read> stack: array<u32>;             // [B]

@group(0) @binding(3) var<storage, read> time_decay: array<vec4<f32>>;  // (H, S)
@group(0) @binding(4) var<storage, read> time_first: array<vec4<f32>>;  // (H, S)

@group(0) @binding(5) var<storage, read> k: array<vec4<f32>>;           // (A, H, S)
@group(0) @binding(6) var<storage, read> v: array<vec4<f32>>;           // (A, H, S)
@group(0) @binding(7) var<storage, read> r: array<vec4<f32>>;           // (A, H, S)

@group(0) @binding(8) var<storage, read_write> x: array<vec4<f32>>;     // (A, H, S)
@group(0) @binding(9) var<storage, read_write> state: array<vec4<f32>>; // (B, H, S, S)

const BLOCK_SIZE: u32 = 128u;

var<workgroup> shared_k: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_r: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_w: array<>;

fn compute_index(batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x / 4u;
    let offset = view.offset.x / 4u;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

fn compute_cursor(x: u32) -> Cursor {
    // let unpacked = vec4<u32>(unpack4x8unorm(x) * 255.0 + 0.5);
    var cursor: Cursor;
    cursor.batch = x & 0xffu;
    cursor.token = (x >> 8u) & 0xffffu;
    cursor.len = (x >> 24u) & 0xffu;
    return cursor;
}

@compute @workgroup_size(128, 1, 1)
fn time_mix(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let channel = stride * shape[1];

    let index = invocation_id.x;
    let batch = invocation_id.y;
    let cursor = compute_cursor(stack[batch]);

    if index >= channel {
        return;
    }
}