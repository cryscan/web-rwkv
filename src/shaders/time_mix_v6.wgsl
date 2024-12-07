struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

struct Cursor {
    batch: u32,
    token: u32,
    len: u32,
};

struct Input {
    @builtin(global_invocation_id) uid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
};

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                    // [S, H, A]
@group(0) @binding(1) var<uniform> view: View;                          // [C, S + 1, B]
@group(0) @binding(2) var<storage, read> cursors: array<u32>;           // [A]

@group(0) @binding(3) var<storage, read> time_decay: array<vec4<f32>>;  // (A, H, S)
@group(0) @binding(4) var<storage, read> time_first: array<vec4<f32>>;  // (H, S)
@group(0) @binding(5) var<storage, read_write> state: array<vec4<f32>>; // (B, S + 1, C)

#ifdef FP16
@group(0) @binding(6) var<storage, read> k: array<vec2<u32>>;           // (A, H, S)
@group(0) @binding(7) var<storage, read> v: array<vec2<u32>>;           // (A, H, S)
@group(0) @binding(8) var<storage, read> r: array<vec2<u32>>;           // (A, H, S)
@group(0) @binding(9) var<storage, read_write> x: array<vec2<u32>>;     // (A, H, S)
#else
@group(0) @binding(6) var<storage, read> k: array<vec4<f32>>;           // (A, H, S)
@group(0) @binding(7) var<storage, read> v: array<vec4<f32>>;           // (A, H, S)
@group(0) @binding(8) var<storage, read> r: array<vec4<f32>>;           // (A, H, S)
@group(0) @binding(9) var<storage, read_write> x: array<vec4<f32>>;     // (A, H, S)
#endif

var<workgroup> shared_k: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_r: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_u: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_w: array<vec4<f32>, BLOCK_SIZE>;

fn compute_index(batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x >> 2u;
    let offset = vec3<u32>(view.offset.zy, view.offset.x >> 2u);
    return dot(vec3<u32>(batch, token, index) + offset, vec3<u32>(view.stride.y * stride, stride, 1u));
}

fn compute_cursor(x: u32) -> Cursor {
    // let unpacked = vec4<u32>(unpack4x8unorm(x) * 255.0 + 0.5);
    var cursor: Cursor;
    cursor.batch = x & 0xffu;
    cursor.token = (x >> 8u) & 0xffffu;
    cursor.len = (x >> 24u) & 0xffu;
    return cursor;
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn time_mix(in: Input) {
    let stride_head = shape[0] / 4u;
    let stride = shape[1] * stride_head;

    let index = in.uid.x;
    let head = in.tid.x / stride_head;
    let h = head * stride_head;

    if index < stride {
        shared_u[in.tid.x] = time_first[index];
    }

    for (var t = 0u; t < shape[2]; t += 1u) {
        let bti = t * stride + index;
        let cursor = compute_cursor(cursors[t]);

        if index < stride {
#ifdef FP16
            state[compute_index(cursor.batch, 0u, index)] = unpack4x16float(x[(cursor.token + cursor.len - 1u) * stride + index]);
#else
            state[compute_index(cursor.batch, 0u, index)] = x[(cursor.token + cursor.len - 1u) * stride + index];
#endif
        }

        workgroupBarrier();
        if index < stride {
            shared_w[in.tid.x] = time_decay[bti];
#ifdef FP16
            shared_k[in.tid.x] = unpack4x16float(k[bti]);
            shared_r[in.tid.x] = unpack4x16float(r[bti]);
#else
            shared_k[in.tid.x] = k[bti];
            shared_r[in.tid.x] = r[bti];
#endif
        }
        workgroupBarrier();

        if index < stride {
#ifdef FP16
            let vv = unpack4x16float(v[bti]);
#else
            let vv = v[bti];
#endif
            var y = vec4<f32>(0.0);
            for (var j = 0u; j < stride_head; j += 1u) {
                let kk = shared_k[h + j];
                let rr = shared_r[h + j];
                let uu = shared_u[h + j];
                let ww = shared_w[h + j];

                var ss: array<vec4<f32>, 4>;
                var kv: array<vec4<f32>, 4>;

                let bji = compute_index(cursor.batch, j * 4u + 1u, index);

                ss[0] = state[bji + stride * 0u];
                ss[1] = state[bji + stride * 1u];
                ss[2] = state[bji + stride * 2u];
                ss[3] = state[bji + stride * 3u];

                kv[0] = kk[0] * vv;
                kv[1] = kk[1] * vv;
                kv[2] = kk[2] * vv;
                kv[3] = kk[3] * vv;

                y += rr[0] * fma(vec4<f32>(uu[0]), kv[0], ss[0]);
                y += rr[1] * fma(vec4<f32>(uu[1]), kv[1], ss[1]);
                y += rr[2] * fma(vec4<f32>(uu[2]), kv[2], ss[2]);
                y += rr[3] * fma(vec4<f32>(uu[3]), kv[3], ss[3]);

                state[bji + stride * 0u] = fma(vec4<f32>(ww[0]), ss[0], kv[0]);
                state[bji + stride * 1u] = fma(vec4<f32>(ww[1]), ss[1], kv[1]);
                state[bji + stride * 2u] = fma(vec4<f32>(ww[2]), ss[2], kv[2]);
                state[bji + stride * 3u] = fma(vec4<f32>(ww[3]), ss[3], kv[3]);
            }
#ifdef FP16
            x[bti] = pack4x16float(y);
#else
            x[bti] = y;
#endif
        }
    }
}