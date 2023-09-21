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
@group(0) @binding(1) var<uniform> view: View;                              // [C, 4, B] / [C, 5L, B]
@group(0) @binding(2) var<storage, read> stack: array<u32>;                 // [B]

@group(0) @binding(3) var<storage, read> time_decay: array<vec4<f32>>;      // (C)
@group(0) @binding(4) var<storage, read> time_first: array<vec4<f32>>;      // (C)

@group(0) @binding(5) var<storage, read> k: array<vec4<f32>>;               // (1, A, C)
@group(0) @binding(6) var<storage, read> v: array<vec4<f32>>;               // (1, A, C)
@group(0) @binding(7) var<storage, read> r: array<vec4<f32>>;               // (1, A, C)

@group(0) @binding(8) var<storage, read_write> x: array<vec4<f32>>;         // (1, A, C)
@group(0) @binding(9) var<storage, read_write> state: array<vec4<f32>>;    // (B, 4, C)

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
fn time_mix(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let cursor = compute_cursor(stack[batch]);

    let xi = compute_index(cursor.batch, 0u, index);
    let ai = compute_index(cursor.batch, 1u, index);
    let bi = compute_index(cursor.batch, 2u, index);
    let pi = compute_index(cursor.batch, 3u, index);

    state[xi] = x[(cursor.token + cursor.len - 1u) * stride + index];

    for (var t = 0u; t < cursor.len; t += 1u) {
        let bti = (cursor.token + t) * stride + index;

        let kk = k[bti];
        let vv = v[bti];

        let aa = state[ai];
        let bb = state[bi];
        let pp = state[pi];
        var ww = time_first[index] + kk;
        var q = max(pp, ww);
        var e1 = exp(pp - q);
        var e2 = exp(ww - q);

        let rr = 1.0 / (1.0 + exp(-r[bti]));
        x[bti] = rr * (e1 * aa + e2 * vv) / (e1 * bb + e2);

        ww = time_decay[index] + pp;
        q = max(ww, kk);
        e1 = exp(ww - q);
        e2 = exp(kk - q);

        state[ai] = e1 * aa + e2 * vv;
        state[bi] = e1 * bb + e2;
        state[pi] = q;
    }
}