@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

#ifdef FP16
@group(0) @binding(1) var<storage, read_write> x: array<vec2<u32>>;         // (B, T, C)
#else
@group(0) @binding(1) var<storage, read_write> x: array<vec4<f32>>;         // (B, T, C)
#endif

const NUM_SUBGROUPS: u32 = BLOCK_SIZE / MIN_SUBGROUP_SIZE;

var<workgroup> sketch: array<vec4<f32>, NUM_SUBGROUPS>;
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
fn softmax(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32
) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = (batch * shape[1] + token) * stride;

    var _max_4 = vec4<f32>(-1.0e30);
    for (var i = index; i < stride; i += BLOCK_SIZE) {
#ifdef FP16
        let value = unpack4x16float(x[bb + i]);
#else
        let value = x[bb + i];
#endif
        _max_4 = max(_max_4, value);
    }
    _max_4 = subgroupMax(_max_4);

    if subgroup_invocation_id == 0u { sketch[subgroup_id] = _max_4; }
    workgroupBarrier();

#ifdef SUBGROUP_SIZE_32_32
    reduce_max(index, 2u);
    reduce_max(index, 1u);
#else
#ifdef SUBGROUP_SIZE_32_64
    if subgroup_size == 32u { reduce_max(index, 2u); }
    reduce_max(index, 1u);
#else
#ifdef SUBGROUP_SIZE_64_64
    reduce_max(index, 1u);
#else
    for (var step = num_subgroups >> 1u; step > 0u; step >>= 1u) {
        if index < step {
            sketch[index] = max(sketch[index], sketch[index + step]);
        }
        workgroupBarrier();
    }
#endif
#endif
#endif

    if index == 0u {
        _max_4 = sketch[0];
        var _max = _max_4.x;
        _max = max(_max, _max_4.y);
        _max = max(_max, _max_4.z);
        _max = max(_max, _max_4.w);
        maximum = _max;
    }
    workgroupBarrier();

    var _sum_4: vec4<f32>;
    for (var i = index; i < stride; i += BLOCK_SIZE) {
#ifdef FP16
        let value = unpack4x16float(x[bb + i]);
#else
        let value = x[bb + i];
#endif
        _sum_4 += exp(value - maximum);
    }
    _sum_4 = subgroupAdd(_sum_4);

    if subgroup_invocation_id == 0u { sketch[subgroup_id] = _sum_4; }
    workgroupBarrier();

#ifdef SUBGROUP_SIZE_32_32
    reduce_sum(index, 2u);
    reduce_sum(index, 1u);
#else
#ifdef SUBGROUP_SIZE_32_64
    if subgroup_size == 32u { reduce_sum(index, 2u); }
    reduce_sum(index, 1u);
#else
#ifdef SUBGROUP_SIZE_64_64
    reduce_sum(index, 1u);
#else
    for (var step = num_subgroups >> 1u; step > 0u; step >>= 1u) {
        if index < step {
            sketch[index] += sketch[index + step];
        }
        workgroupBarrier();
    }
#endif
#endif
#endif

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