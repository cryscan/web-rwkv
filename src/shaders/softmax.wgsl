@group(0) @binding(0) var<uniform> num_vocab: u32;

@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;               // (T, C)
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;    // (T, C)

const BLOCK_SIZE: u32 = 128u;

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> sum: f32;
var<workgroup> maximum: f32;

fn reduce_max(index: u32, stride: u32) {
    if index < stride {
        sketch[index] = max(sketch[index], sketch[index + stride]);
    }
    workgroupBarrier();
}

fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(128, 1, 1)
fn softmax(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let token = invocation_id.y;
    let stride = num_vocab / 4u;

    sketch[index] = vec4<f32>(-1.0e30);
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = x[stride * token + i];
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
        maximum = sketch[0].x;
        maximum = max(maximum, sketch[0].y);
        maximum = max(maximum, sketch[0].z);
        maximum = max(maximum, sketch[0].w);
    }
    workgroupBarrier();

    sketch[index] = vec4<f32>(0.0);
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = x[stride * token + i];
        sketch[index] += exp(value - maximum);
    }
    workgroupBarrier();

    reduce_sum(index, 64u);
    reduce_sum(index, 32u);
    reduce_sum(index, 16u);
    reduce_sum(index, 8u);
    reduce_sum(index, 4u);
    reduce_sum(index, 2u);
    reduce_sum(index, 1u);

    if index == 0u {
        sum = dot(sketch[0], vec4<f32>(1.0));
    }
    workgroupBarrier();

    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = x[stride * token + i];
        output[stride * token + i] = exp(value - maximum) / sum;
    }
}