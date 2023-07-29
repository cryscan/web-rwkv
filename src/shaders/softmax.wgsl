@group(0) @binding(0) var<uniform> dim: u32;

@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;               // (T, C)
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;    // (T, C)

const BLOCK_SIZE: u32 = 128u;

var<workgroup> local: array<vec4<f32>, BLOCK_SIZE>;

fn reduce_max(index: u32, stride: u32) {
    if index < stride {
        local[index] = max(local[index], local[index + stride]);
    }
    workgroupBarrier();
}

fn reduce_add(index: u32, stride: u32) {
    if index < stride {
        local[index] += local[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(128, 1, 1)
fn softmax(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let stride = dim / 4u;

    local[index] = vec4<f32>(-1.0e30);
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = x[stride * token + i];
        local[index] = max(local[index], value);
    }
    workgroupBarrier();

    reduce_max(index, 64u);
    reduce_max(index, 32u);

    if index < 32u {
        local[index] = max(local[index], local[index + 16u]);
        local[index] = max(local[index], local[index + 8u]);
        local[index] = max(local[index], local[index + 4u]);
        local[index] = max(local[index], local[index + 2u]);
        local[index] = max(local[index], local[index + 1u]);
    }
    workgroupBarrier();

    if index == 0u {
        var block_max = local[0].x;
        block_max = max(block_max, local[0].y);
        block_max = max(block_max, local[0].z);
        block_max = max(block_max, local[0].w);
        local[0] = vec4<f32>(block_max);
    }
    workgroupBarrier();

    let block_max = local[0];

    local[index] = vec4<f32>(0.0);
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = x[stride * token + i];
        local[index] += exp(value - block_max);
    }
    workgroupBarrier();

    reduce_add(index, 64u);
    reduce_add(index, 32u);

    if index < 32u {
        local[index] += local[index + 16u];
        local[index] += local[index + 8u];
        local[index] += local[index + 4u];
        local[index] += local[index + 2u];
        local[index] += local[index + 1u];
    }
    workgroupBarrier();

    if index == 0u {
        let block_sum = dot(local[0], vec4<f32>(1.0));
        local[0] = vec4<f32>(block_sum);
    }
    workgroupBarrier();

    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = x[stride * token + i];
        output[stride * token + i] = exp(value - block_max) / local[0];
    }
}