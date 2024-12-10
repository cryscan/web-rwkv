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

@group(0) @binding(3) var<storage, read_write> state: array<vec4<f32>>; // (B, S + 1, C)
#ifdef FP16
@group(0) @binding(4) var<storage, read> r: array<vec2<u32>>;           // (A, H, S)
@group(0) @binding(5) var<storage, read> w: array<vec2<u32>>;           // (A, H, S)
@group(0) @binding(6) var<storage, read> kv: array<vec2<u32>>;          // (A, H, S)
@group(0) @binding(8) var<storage, read> ab: array<vec2<u32>>;          // (A, H, S)
@group(0) @binding(10) var<storage, read_write> x: array<vec2<u32>>;    // (A, H, S)
#else
@group(0) @binding(4) var<storage, read> r: array<vec4<f32>>;           // (A, H, S)
@group(0) @binding(5) var<storage, read> w: array<vec4<f32>>;           // (A, H, S)
@group(0) @binding(6) var<storage, read> kv: array<vec4<f32>>;          // (2, A, H, S)
@group(0) @binding(8) var<storage, read> ab: array<vec4<f32>>;          // (2, A, H, S)
@group(0) @binding(10) var<storage, read_write> x: array<vec4<f32>>;    // (A, H, S)
#endif

var<workgroup> shared_r: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_k: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_w: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_a: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_b: array<vec4<f32>, BLOCK_SIZE>;

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

// ACTIVATION_DEFINE

fn sigmoid_exp(x: vec4<f32>) -> vec4<f32> {
    return exp(-0.606531 * sigmoid(x)); // 0.606531 = exp(-0.5)
}

fn __r(ti: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(r[ti]);
#else
    return r[ti];
#endif
}

fn __w(ti: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(w[ti]);
#else
    return w[ti];
#endif
}

fn __k(ti: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(kv[ti]);
#else
    return kv[ti];
#endif
}

fn __v(ti: u32) -> vec4<f32> {
    let offset = shape[2] * shape[1] * shape[0] / 4u;
#ifdef FP16
    return unpack4x16float(kv[ti + offset]);
#else
    return kv[ti + offset];
#endif
}

fn __a(ti: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(ab[ti]);
#else
    return ab[ti];
#endif
}

fn __kk(ti: u32) -> vec4<f32> {
    let offset = shape[2] * shape[1] * shape[0] / 4u;
#ifdef FP16
    return unpack4x16float(ab[ti + offset]);
#else
    return ab[ti + offset];
#endif
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn time_mix(in: Input) {
    // const HEAD_SIZE = shape[0] / 4u;
    let stride = shape[1] * shape[0] / 4u;

    let index = in.uid.x;
    let head = in.tid.x / HEAD_SIZE;
    let h = head * HEAD_SIZE;

    for (var t = 0u; t < shape[2]; t += 1u) {
        let ti = t * stride + index;
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
            shared_r[in.tid.x] = __r(ti);
            shared_w[in.tid.x] = sigmoid_exp(__w(ti));
            shared_k[in.tid.x] = __k(ti);
            let a = __a(ti);
            let kk = __kk(ti);
            shared_a[in.tid.x] = -kk;
            shared_b[in.tid.x] = kk * a;
        }
        workgroupBarrier();

        if index < stride {
            var sa = vec4<f32>(0.0);
            for (var j = 0u; j < HEAD_SIZE; j += 1u) {
                var ss: mat4x4<f32>;
                let aa = shared_a[h + j];

                var bji = compute_index(cursor.batch, j * 4u + 1u, index);
                ss[0] = state[bji]; bji += stride;
                ss[1] = state[bji]; bji += stride;
                ss[2] = state[bji]; bji += stride;
                ss[3] = state[bji];

                sa += ss * aa;
            }

            let vv = __v(ti);
            var y = vec4<f32>(0.0);
            for (var j = 0u; j < HEAD_SIZE; j += 1u) {
                let rr = shared_r[h + j];
                let ww = shared_w[h + j];
                let kk = shared_k[h + j];
                let bb = shared_b[h + j];

                var ss: array<vec4<f32>, 4>;

                let bji = compute_index(cursor.batch, j * 4u + 1u, index);
                ss[0] = state[bji + stride * 0u];
                ss[1] = state[bji + stride * 1u];
                ss[2] = state[bji + stride * 2u];
                ss[3] = state[bji + stride * 3u];

                ss[0] = ss[0] * ww[0] + kk[0] * vv + sa * bb[0];
                ss[1] = ss[1] * ww[1] + kk[1] * vv + sa * bb[1];
                ss[2] = ss[2] * ww[2] + kk[2] * vv + sa * bb[2];
                ss[3] = ss[3] * ww[3] + kk[3] * vv + sa * bb[3];

                y += rr[0] * ss[0];
                y += rr[1] * ss[1];
                y += rr[2] * ss[2];
                y += rr[3] * ss[3];

                state[bji + stride * 0u] = ss[0];
                state[bji + stride * 1u] = ss[1];
                state[bji + stride * 2u] = ss[2];
                state[bji + stride * 3u] = ss[3];
            }
#ifdef FP16
            x[ti] = pack4x16float(y);
#else
            x[ti] = y;
#endif
        }
    }
}