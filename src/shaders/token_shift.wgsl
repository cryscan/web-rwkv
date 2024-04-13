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

@group(0) @binding(0) var<uniform> vx: View;                                // [C, A, 1] | [C, A, I]
@group(0) @binding(1) var<uniform> vt: View;                                // [C, 1, I] | [C, A, I]
@group(0) @binding(2) var<uniform> vs: View;                                // [C, _, B] / [C, 5L, B]
@group(0) @binding(3) var<storage, read> cursors: array<u32>;               // [A]

#ifdef TIME_MIX_FP16
@group(0) @binding(4) var<storage, read> time_mix: array<vec2<u32>>;        // (I, 1, C) | (I, A, C)
#else
@group(0) @binding(4) var<storage, read> time_mix: array<vec4<f32>>;        // (I, 1, C) | (I, A, C)
#endif

@group(0) @binding(5) var<storage, read> sx: array<vec4<f32>>;              // (B, 1, C)
#ifdef IN_FP16
@group(0) @binding(6) var<storage, read> x: array<vec2<u32>>;               // (1, A, C)
#else
@group(0) @binding(6) var<storage, read> x: array<vec4<f32>>;               // (1, A, C)
#endif
#ifdef OUT_FP16
@group(0) @binding(7) var<storage, read_write> output: array<vec2<u32>>;    // (I, A, C)
#else
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;    // (I, A, C)
#endif

fn compute_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
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

fn load_time_mix(count: u32, stack: u32, index: u32) -> vec4<f32> {
#ifdef TIME_MIX_FP16
    let token = select(stack, 0u, vt.shape.y == 1u);
    return unpack4x16float(time_mix[compute_index(vt, count, token, index)]);
#else
    let token = select(stack, 0u, vt.shape.y == 1u);
    return time_mix[compute_index(vt, count, token, index)];
#endif
}

fn load_input(stack: u32, index: u32) -> vec4<f32> {
#ifdef IN_FP16
    return unpack4x16float(x[compute_index(vx, 0u, stack, index)]);
#else
    return x[compute_index(vx, 0u, stack, index)];
#endif
}

fn store_output(count: u32, stack: u32, index: u32, value: vec4<f32>) {
#ifdef OUT_FP16
    output[compute_index(vx, count, stack, index)] = pack4x16float(value);
#else
    output[compute_index(vx, count, stack, index)] = value;
#endif
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn token_shift(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_blocks: vec3<u32>) {
    let stride = vec3<u32>(vx.shape.x >> 2u, vx.shape.yz);
    let index = invocation_id.x;
    let stack = invocation_id.y;
    let count = invocation_id.z;
    let cursor = compute_cursor(cursors[stack]);
    let token = stack - cursor.token;

    if any(vec3<u32>(index, stack, count) > stride) {
        return;
    }

    let factor = load_time_mix(count, stack, index);

#ifdef REVERSED
    if token == 0u {
        let out = mix(load_input(stack, index), sx[compute_index(vs, cursor.batch, 0u, index)], factor);
        store_output(count, stack, index, out);
    } else {
        let out = mix(load_input(stack, index), load_input(stack - 1u, index), factor);
        store_output(count, stack, index, out);
    }
#else
    if token == 0u {
        let out = mix(sx[compute_index(vs, cursor.batch, 0u, index)], load_input(stack, index), factor);
        store_output(count, stack, index, out);
    } else {
        let out = mix(load_input(stack - 1u, index), load_input(stack, index), factor);
        store_output(count, stack, index, out);
    }
#endif
}
