@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

@group(0) @binding(1) var<storage, read> w: array<vec2<u32>>;               // (C)
@group(0) @binding(2) var<storage, read> b: array<vec2<u32>>;               // (C)
#ifdef FP16
@group(0) @binding(3) var<storage, read_write> x: array<vec2<u32>>;         // (B, T, C)
#else
@group(0) @binding(3) var<storage, read_write> x: array<vec4<f32>>;         // (B, T, C)
#endif

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> mean: f32;
var<workgroup> norm: f32;

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
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
fn recenter(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = (batch * shape[1] + token) * stride;

    var _sum_4: vec4<f32>;
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = load_x(bb + i);
        _sum_4 += value;
    }
    sketch[index] = _sum_4;
    workgroupBarrier();

    reduce_sum(index, 64u);
    reduce_sum(index, 32u);
    reduce_sum(index, 16u);
    reduce_sum(index, 8u);
    reduce_sum(index, 4u);
    reduce_sum(index, 2u);
    reduce_sum(index, 1u);

    if index == 0u {
        mean = dot(sketch[0], vec4<f32>(1.0)) / f32(shape[0]);
    }
    workgroupBarrier();

    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = load_x(bb + i);
        store_x(bb + i, value - mean);
    }
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn rms_norm(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = (batch * shape[1] + token) * stride;

    var _sum_4: vec4<f32>;
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = load_x(bb + i);
        _sum_4 += value * value;
    }
    sketch[index] = _sum_4;
    workgroupBarrier();

    reduce_sum(index, 64u);
    reduce_sum(index, 32u);
    reduce_sum(index, 16u);
    reduce_sum(index, 8u);
    reduce_sum(index, 4u);
    reduce_sum(index, 2u);
    reduce_sum(index, 1u);

    if index == 0u {
        norm = inverseSqrt(dot(sketch[0], vec4<f32>(1.0)) / f32(shape[0]) + EPS);
    }
    workgroupBarrier();

    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = load_x(bb + i) * norm;
        store_x(bb + i, fma(value, unpack4x16float(w[i]), unpack4x16float(b[i])));
    }
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn l2_norm(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = (batch * shape[1] + token) * stride;

    var _sum_4: vec4<f32>;
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = load_x(bb + i);
        _sum_4 += value * value;
    }
    sketch[index] = _sum_4;
    workgroupBarrier();

    reduce_sum(index, 64u);
    reduce_sum(index, 32u);
    reduce_sum(index, 16u);
    reduce_sum(index, 8u);
    reduce_sum(index, 4u);
    reduce_sum(index, 2u);
    reduce_sum(index, 1u);

    if index == 0u {
        norm = inverseSqrt(dot(sketch[0], vec4<f32>(1.0)) + EPS);
    }
    workgroupBarrier();

    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = load_x(bb + i) * norm;
        store_x(bb + i, value);
    }
}
