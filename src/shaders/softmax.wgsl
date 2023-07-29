@group(0) @binding(0) var<uniform> dim: u32;

@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;               // (T, C)
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;    // (T, C)

const BLOCK_SIZE: u32 = 128u;

var<workgroup> sum: array<vec4<f32>, BLOCK_SIZE>;

fn reduce_step_barrier(index: u32, stride: u32) {
    if index < stride {
        sum[index] += sum[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(128, 1, 1)
fn softmax(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let stride = dim / 4u;

    sum[index] = vec4<f32>(0.0);
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = x[stride * token + i];
        sum[index] += exp(value);
    }
    workgroupBarrier();

    reduce_step_barrier(index, 64u);
    reduce_step_barrier(index, 32u);

    if index < 32u {
        sum[index] += sum[index + 16u];
        sum[index] += sum[index + 8u];
        sum[index] += sum[index + 4u];
        sum[index] += sum[index + 2u];
        sum[index] += sum[index + 1u];
    }
    workgroupBarrier();

    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = x[stride * token + i];
        output[stride * token + i] = exp(value) / sum[0];
    }
}