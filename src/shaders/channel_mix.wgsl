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

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, A, 1]
@group(0) @binding(1) var<uniform> view: View;                              // [C, 1, B] / [C, 5L, B]
@group(0) @binding(2) var<storage, read> cursors: array<u32>;               // [A]
@group(0) @binding(3) var<storage, read_write> state: array<vec4<f32>>;     // (B, C)

#ifdef FP16
@group(0) @binding(4) var<storage, read> r: array<vec2<u32>>;               // (1, A, C)
@group(0) @binding(5) var<storage, read> v: array<vec2<u32>>;               // (1, A, C)
@group(0) @binding(6) var<storage, read_write> x: array<vec2<u32>>;         // (1, A, C)
#else
@group(0) @binding(4) var<storage, read> r: array<vec4<f32>>;               // (1, A, C)
@group(0) @binding(5) var<storage, read> v: array<vec4<f32>>;               // (1, A, C)
@group(0) @binding(6) var<storage, read_write> x: array<vec4<f32>>;         // (1, A, C)
#endif

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

fn load_r(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(r[index]);
#else
    return r[index];
#endif
}

fn load_v(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(v[index]);
#else
    return v[index];
#endif
}

fn load_x(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(x[index]);
#else
    return x[index];
#endif
}

fn store_x(index: u32, value: vec4<f32>) {
#ifdef FP16
    x[index] = pack4x16float(value);
#else
    x[index] = value;
#endif
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn channel_mix(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let stack = invocation_id.y;
    let cursor = compute_cursor(cursors[stack]);
    let token = stack - cursor.token;

    let bti = stack * stride + index;

    if index >= stride {
        return;
    }

    if token + 1u == cursor.len {
        state[compute_index(cursor.batch, 0u, index)] = load_x(bti);
    }

#ifdef V7
    store_x(bti, load_v(bti));
#else
    let rr = 1.0 / (1.0 + exp(-load_r(bti)));
    let vv = load_v(bti);
    store_x(bti, rr * vv);
#endif
}