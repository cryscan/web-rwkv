@group(0) @binding(0) var<uniform> dims: vec2<u32>;                         // [C, R]
@group(0) @binding(1) var<storage, read_write> input: array<vec4<f32>>;     // (R, C)

@group(0) @binding(2) var<storage, read_write> mx: array<vec4<f32>>;        // (C)
@group(0) @binding(3) var<storage, read_write> rx: array<vec4<f32>>;        // (C)
@group(0) @binding(4) var<storage, read_write> my: array<f32>;              // (R)
@group(0) @binding(5) var<storage, read_write> ry: array<f32>;              // (R)

@group(0) @binding(6) var<storage, read_write> output: array<u32>;          // (R, C)

const BLOCK_SIZE: u32 = 128u;

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> local_mx: vec4<f32>;
var<workgroup> local_rx: vec4<f32>;
var<workgroup> local_my: f32;
var<workgroup> local_ry: f32;

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

@compute @workgroup_size(128, 1, 1)
fn compute_my(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let stride = vec2<u32>(dims.x / 4u, dims.y);

    sketch[index] = vec4<f32>(1.0e30);
    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let value = input[stride.x * batch + i];
        sketch[index] = min(sketch[index], value);
    }
    workgroupBarrier();

    reduce_min(index, 64u);
    reduce_min(index, 32u);

    if index < 32u {
        sketch[index] = min(sketch[index], sketch[index + 16u]);
        sketch[index] = min(sketch[index], sketch[index + 8u]);
        sketch[index] = min(sketch[index], sketch[index + 4u]);
        sketch[index] = min(sketch[index], sketch[index + 2u]);
        sketch[index] = min(sketch[index], sketch[index + 1u]);
    }
    workgroupBarrier();

    if index == 0u {
        local_my = sketch[0].x;
        local_my = min(local_my, sketch[0].y);
        local_my = min(local_my, sketch[0].z);
        local_my = min(local_my, sketch[0].w);
        my[batch] = local_my;
    }
    workgroupBarrier();

    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let value = input[stride.x * batch + i];
        input[stride.x * batch + i] = value - local_my;
    }
}

@compute @workgroup_size(128, 1, 1)
fn compute_mx(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let stride = vec2<u32>(dims.x / 4u, dims.y);

    sketch[index] = vec4<f32>(1.0e30);
    for (var j = index; j < stride.y; j += BLOCK_SIZE) {
        let value = input[stride.x * j + batch];
        sketch[index] = min(sketch[index], value);
    }
    workgroupBarrier();

    reduce_min(index, 64u);
    reduce_min(index, 32u);

    if index < 32u {
        sketch[index] = min(sketch[index], sketch[index + 16u]);
        sketch[index] = min(sketch[index], sketch[index + 8u]);
        sketch[index] = min(sketch[index], sketch[index + 4u]);
        sketch[index] = min(sketch[index], sketch[index + 2u]);
        sketch[index] = min(sketch[index], sketch[index + 1u]);
    }
    workgroupBarrier();

    if index == 0u {
        local_mx = sketch[0];
        mx[batch] = local_mx;
    }
    workgroupBarrier();

    for (var j = index; j < stride.y; j += BLOCK_SIZE) {
        let value = input[stride.x * j + batch];
        input[stride.x * j + batch] = value - local_mx;
    }
}

@compute @workgroup_size(128, 1, 1)
fn compute_ry(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let stride = vec2<u32>(dims.x / 4u, dims.y);

    sketch[index] = vec4<f32>(-1.0e30);
    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let value = input[stride.x * batch + i];
        sketch[index] = max(sketch[index], value);
    }
    workgroupBarrier();

    reduce_max(index, 64u);
    reduce_max(index, 32u);

    if index < 32u {
        sketch[index] = max(sketch[index], sketch[index + 16u]);
        sketch[index] = max(sketch[index], sketch[index + 8u]);
        sketch[index] = max(sketch[index], sketch[index + 4u]);
        sketch[index] = max(sketch[index], sketch[index + 2u]);
        sketch[index] = max(sketch[index], sketch[index + 1u]);
    }
    workgroupBarrier();

    if index == 0u {
        local_ry = sketch[0].x;
        local_ry = max(local_ry, sketch[0].y);
        local_ry = max(local_ry, sketch[0].z);
        local_ry = max(local_ry, sketch[0].w);
        ry[batch] = local_ry;
    }
    workgroupBarrier();

    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let value = input[stride.x * batch + i];
        input[stride.x * batch + i] = value / local_ry;
    }
}

@compute @workgroup_size(128, 1, 1)
fn compute_rx(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let stride = vec2<u32>(dims.x / 4u, dims.y);

    sketch[index] = vec4<f32>(-1.0e30);
    for (var j = index; j < stride.y; j += BLOCK_SIZE) {
        let value = input[stride.x * j + batch];
        sketch[index] = max(sketch[index], value);
    }
    workgroupBarrier();

    reduce_max(index, 64u);
    reduce_max(index, 32u);

    if index < 32u {
        sketch[index] = max(sketch[index], sketch[index + 16u]);
        sketch[index] = max(sketch[index], sketch[index + 8u]);
        sketch[index] = max(sketch[index], sketch[index + 4u]);
        sketch[index] = max(sketch[index], sketch[index + 2u]);
        sketch[index] = max(sketch[index], sketch[index + 1u]);
    }
    workgroupBarrier();

    if index == 0u {
        local_rx = sketch[0];
        rx[batch] = local_rx;
    }
    workgroupBarrier();

    for (var j = index; j < stride.y; j += BLOCK_SIZE) {
        let value = input[stride.x * j + batch];
        input[stride.x * j + batch] = value / local_rx;
    }
}

@compute @workgroup_size(128, 1, 1)
fn quantize(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let batch = invocation_id.y;
    let stride = vec2<u32>(dims.x / 4u, dims.y);

    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let value = input[stride.x * batch + i];
        output[stride.x * batch + i] = pack4x8unorm(value);
    }
}