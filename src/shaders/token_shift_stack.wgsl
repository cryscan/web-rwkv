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
@group(0) @binding(1) var<uniform> view: View;                              // [C, _, B] / [C, 5L, B]
@group(0) @binding(2) var<storage, read> cursors: array<u32>;               // [A]

@group(0) @binding(3) var<storage, read> time_mix: array<vec2<u32>>;        // (C)
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;               // (1, A, C)
@group(0) @binding(5) var<storage, read> sx: array<vec4<f32>>;              // (B, 1, C)
@group(0) @binding(6) var<storage, read_write> output: array<vec4<f32>>;    // (1, A, C)

const BLOCK_SIZE: u32 = 128u;

fn compute_index(batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x / 4u;
    let offset = view.offset.x / 4u;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

fn compute_cursor(x: u32) -> Cursor {
    let unpacked = vec4<u32>(unpack4x8unorm(x) * 255.0);
    var cursor: Cursor;
    cursor.batch = unpacked.x;
    cursor.token = (unpacked.z << 8u) | unpacked.y;
    cursor.len = unpacked.w;
    return cursor;
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

@compute @workgroup_size(128, 1, 1)
fn token_shift(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let stack = invocation_id.y;
    let cursor = compute_cursor(cursors[stack]);
    let token = stack - cursor.token;

    if index >= stride || stack >= shape[1] {
        return;
    }

    let bti = stack * stride + index;
    if token == 0u {
        output[bti] = mix(sx[compute_index(cursor.batch, 0u, index)], x[bti], unpack4x16float(time_mix[index]));
    } else {
        output[bti] = mix(x[bti - stride], x[bti], unpack4x16float(time_mix[index]));
    }
}