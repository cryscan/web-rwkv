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

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, A, 1]
@group(0) @binding(1) var<uniform> view: View;                              // [C, 1, B] / [C, 5L, B]
@group(0) @binding(2) var<storage, read> cursors: array<u32>;               // [A]

@group(0) @binding(3) var<storage, read> r: array<vec4<f32>>;               // (1, A, C)
@group(0) @binding(4) var<storage, read> v: array<vec4<f32>>;               // (1, A, C)

@group(0) @binding(5) var<storage, read_write> x: array<vec4<f32>>;         // (1, A, C)
@group(0) @binding(6) var<storage, read_write> state: array<vec4<f32>>;     // (B, C)

const BLOCK_SIZE: u32 = 128u;

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
fn channel_mix(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let stack = invocation_id.y;
    let cursor = compute_cursor(cursors[stack]);
    let token = stack - cursor.token;

    let bti = stack * stride + index;

    if token + 1u == cursor.len {
        state[compute_index(cursor.batch, 0u, index)] = x[bti];
    }

    let rr = 1.0 / (1.0 + exp(-r[bti]));
    x[bti] = rr * v[bti];
}