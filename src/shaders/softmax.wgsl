@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

#ifdef FP16
@group(0) @binding(1) var<storage, read_write> x: array<vec2<u32>>;         // (B, T, C)
#else
@group(0) @binding(1) var<storage, read_write> x: array<vec4<f32>>;         // (B, T, C)
#endif

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> sum: f32;
var<workgroup> maximum: f32;

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

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

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn softmax(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = (batch * shape[1] + token) * stride;

    sketch[index] = vec4<f32>(-1.0e30);
    for (var i = index; i < stride; i += BLOCK_SIZE) {
#ifdef FP16
        let value = unpack4x16float(x[bb + i]);
#else
        let value = x[bb + i];
#endif
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
#ifdef FP16
        let value = unpack4x16float(x[bb + i]);
#else
        let value = x[bb + i];
#endif
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
#ifdef FP16
        let value = unpack4x16float(x[bb + i]);
        x[bb + i] = pack4x16float(exp(value - maximum) / sum);
#else
        let value = x[bb + i];
        x[bb + i] = exp(value - maximum) / sum;
#endif
    }
}