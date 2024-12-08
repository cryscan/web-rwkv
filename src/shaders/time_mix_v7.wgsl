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

fn load_k(ti: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(kv[ti]);
#else
    return kv[ti];
#endif
}

fn load_v(ti: u32) -> vec4<f32> {
    let offset = shape[2] * shape[1] * shape[0] / 4u;
#ifdef FP16
    return unpack4x16float(kv[ti + offset]);
#else
    return kv[ti + offset];
#endif
}

fn load_a(ti: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(ab[ti]);
#else
    return ab[ti];
#endif
}

fn load_kk(ti: u32) -> vec4<f32> {
    let offset = shape[2] * shape[1] * shape[0] / 4u;
#ifdef FP16
    return unpack4x16float(ab[ti + offset]);
#else
    return ab[ti + offset];
#endif
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn time_mix(in: Input) {
    let stride_head = shape[0] / 4u;
    let stride = shape[1] * stride_head;

    let index = in.uid.x;
    let head = in.tid.x / stride_head;
    let h = head * stride_head;

    var _state: array<vec4<f32>, BLOCK_SIZE>;

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
#ifdef FP16
            shared_r[in.tid.x] = unpack4x16float(r[ti]);
            shared_w[in.tid.x] = sigmoid_exp(unpack4x16float(w[ti]));
#else
            shared_r[in.tid.x] = r[ti];
            shared_w[in.tid.x] = sigmoid_exp(w[ti]);
#endif
            shared_k[in.tid.x] = load_k(ti);
            let _a = load_a(ti);
            let _kk = load_kk(ti);
            shared_a[in.tid.x] = -kk;
            shared_b[in.tid.x] = kk * a;
        }
        workgroupBarrier();

        if index < stride {

        }
    }
}