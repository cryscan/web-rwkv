@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, R]
@group(0) @binding(1) var<storage, read_write> input: array<vec2<u32>>;     // (R, C)

@group(0) @binding(2) var<storage, read_write> mx: array<vec4<f32>>;        // (C)
@group(0) @binding(3) var<storage, read_write> rx: array<vec4<f32>>;        // (C)
@group(0) @binding(4) var<storage, read_write> my: array<f32>;              // (R)
@group(0) @binding(5) var<storage, read_write> ry: array<f32>;              // (R)

@group(0) @binding(6) var<storage, read_write> output: array<u32>;          // (R, C)

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> rmx: vec4<f32>;
var<workgroup> rmy: f32;

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn reduce_min(index: u32, stride: u32) {
    if index < stride {
        sketch[index] = min(sketch[index], sketch[index + stride]);
    }
    workgroupBarrier();
}

fn reduce_max(index: u32, stride: u32) {
    if index < stride {
        sketch[index] = max(sketch[index], sketch[index + stride]);
    }
    workgroupBarrier();
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn compute_my(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let stride = vec2<u32>(shape.x / 4u, shape.y);

    sketch[index] = vec4<f32>(1.0e30);
    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let value = unpack4x16float(input[stride.x * batch + i]);
        sketch[index] = min(sketch[index], value);
    }
    workgroupBarrier();

    reduce_min(index, 64u);
    reduce_min(index, 32u);
    reduce_min(index, 16u);
    reduce_min(index, 8u);
    reduce_min(index, 4u);
    reduce_min(index, 2u);
    reduce_min(index, 1u);

    if index == 0u {
        rmy = sketch[0].x;
        rmy = min(rmy, sketch[0].y);
        rmy = min(rmy, sketch[0].z);
        rmy = min(rmy, sketch[0].w);
        my[batch] = rmy;
    }
    workgroupBarrier();

    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let value = unpack4x16float(input[stride.x * batch + i]);
        input[stride.x * batch + i] = pack4x16float(value - rmy);
    }
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn compute_mx(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let stride = vec2<u32>(shape.x / 4u, shape.y);

    sketch[index] = vec4<f32>(1.0e30);
    for (var j = index; j < stride.y; j += BLOCK_SIZE) {
        let value = unpack4x16float(input[stride.x * j + batch]);
        sketch[index] = min(sketch[index], value);
    }
    workgroupBarrier();

    reduce_min(index, 64u);
    reduce_min(index, 32u);
    reduce_min(index, 16u);
    reduce_min(index, 8u);
    reduce_min(index, 4u);
    reduce_min(index, 2u);
    reduce_min(index, 1u);

    if index == 0u {
        rmx = sketch[0];
        mx[batch] = rmx;
    }
    workgroupBarrier();

    for (var j = index; j < stride.y; j += BLOCK_SIZE) {
        let value = unpack4x16float(input[stride.x * j + batch]);
        input[stride.x * j + batch] = pack4x16float(value - rmx);
    }
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn compute_ry(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let stride = vec2<u32>(shape.x / 4u, shape.y);

    sketch[index] = vec4<f32>(-1.0e30);
    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let value = unpack4x16float(input[stride.x * batch + i]);
        sketch[index] = max(sketch[index], value);
    }
    workgroupBarrier();

    reduce_max(index, 64u);
    reduce_max(index, 32u);
    reduce_max(index, 16u);
    reduce_max(index, 8u);
    reduce_max(index, 4u);
    reduce_max(index, 2u);
    reduce_max(index, 1u);

    if index == 0u {
        rmy = sketch[0].x;
        rmy = max(rmy, sketch[0].y);
        rmy = max(rmy, sketch[0].z);
        rmy = max(rmy, sketch[0].w);
        ry[batch] = rmy;
    }
    workgroupBarrier();

    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let value = unpack4x16float(input[stride.x * batch + i]);
        input[stride.x * batch + i] = pack4x16float(value / rmy);
    }
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn compute_rx(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let stride = vec2<u32>(shape.x / 4u, shape.y);

    sketch[index] = vec4<f32>(-1.0e30);
    for (var j = index; j < stride.y; j += BLOCK_SIZE) {
        let value = unpack4x16float(input[stride.x * j + batch]);
        sketch[index] = max(sketch[index], value);
    }
    workgroupBarrier();

    reduce_max(index, 64u);
    reduce_max(index, 32u);
    reduce_max(index, 16u);
    reduce_max(index, 8u);
    reduce_max(index, 4u);
    reduce_max(index, 2u);
    reduce_max(index, 1u);

    if index == 0u {
        rmx = sketch[0];
        rx[batch] = rmx;
    }
    workgroupBarrier();

    for (var j = index; j < stride.y; j += BLOCK_SIZE) {
        let value = unpack4x16float(input[stride.x * j + batch]);
        input[stride.x * j + batch] = pack4x16float(value / rmx);
    }
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn quantize(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let stride = vec2<u32>(shape.x / 4u, shape.y);

    if index >= stride.x || batch >= stride.y {
        return;
    }

    let value = unpack4x16float(input[stride.x * batch + index]);
    output[stride.x * batch + index] = pack4x8unorm(value);
}