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
@group(0) @binding(1) var<uniform> view: View;                          // [C, S + 1, B]
@group(0) @binding(2) var<storage, read> stack: array<u32>;             // [B]

@group(0) @binding(3) var<storage, read> time_decay: array<vec4<f32>>;  // (C)
@group(0) @binding(4) var<storage, read> time_first: array<vec4<f32>>;  // (C)

@group(0) @binding(5) var<storage, read> k: array<vec4<f32>>;           // (A, C)
@group(0) @binding(6) var<storage, read> v: array<vec4<f32>>;           // (A, C)
@group(0) @binding(7) var<storage, read> r: array<vec4<f32>>;           // (A, C)

@group(0) @binding(8) var<storage, read_write> x: array<vec4<f32>>;     // (A, H, S)
@group(0) @binding(9) var<storage, read_write> state: array<vec4<f32>>; // (B, S + 1, C)

const BLOCK_SIZE: u32 = 32u;

var<workgroup> shared_k: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_r: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_u: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_w: array<vec4<f32>, BLOCK_SIZE>;

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

@compute @workgroup_size(32, 1, 1)
fn time_mix(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let stride = shape[0] / 4u;
    let channel = shape[1] * stride;

    let index = invocation_id.x;
    let batch = invocation_id.y;
    let cursor = compute_cursor(stack[batch]);

    let head = index / stride;
    let h = head * stride;

    shared_u[local_id.x] = time_first[index];
    shared_w[local_id.x] = time_decay[index];

    state[compute_index(batch, shape[0], index)] = x[(cursor.token + cursor.len - 1u) * channel + index];

    for (var t = 0u; t < cursor.len; t += 1u) {
        let ti = (cursor.token + t) * channel + index;
        let vv = v[ti];
        shared_k[local_id.x] = k[ti];
        shared_r[local_id.x] = r[ti];
        workgroupBarrier();

        var y: vec4<f32>;
        for (var j = 0u; j < stride; j += 1u) {
            let kk = shared_k[h + j];
            let rr = shared_r[h + j];
            let uu = shared_u[h + j];
            let ww = shared_w[h + j];

            var ss: array<vec4<f32>, 4>;
            var kv: array<vec4<f32>, 4>;

            let bji = compute_index(batch, j << 2u, index);

            ss[0] = state[bji + channel * 0u];
            ss[1] = state[bji + channel * 1u];
            ss[2] = state[bji + channel * 2u];
            ss[3] = state[bji + channel * 3u];

            kv[0] = kk[0] * vv;
            kv[1] = kk[1] * vv;
            kv[2] = kk[2] * vv;
            kv[3] = kk[3] * vv;

            state[bji + channel * 0u] = fma(vec4<f32>(ww[0]), ss[0], kv[0]);
            state[bji + channel * 1u] = fma(vec4<f32>(ww[1]), ss[1], kv[1]);
            state[bji + channel * 2u] = fma(vec4<f32>(ww[2]), ss[2], kv[2]);
            state[bji + channel * 3u] = fma(vec4<f32>(ww[3]), ss[3], kv[3]);

            y += rr[0] * fma(vec4<f32>(uu[0]), kv[0], ss[0]);
            y += rr[1] * fma(vec4<f32>(uu[1]), kv[1], ss[1]);
            y += rr[2] * fma(vec4<f32>(uu[2]), kv[2], ss[2]);
            y += rr[3] * fma(vec4<f32>(uu[3]), kv[3], ss[3]);
        }

        x[ti] = y;

        workgroupBarrier();
    }
}