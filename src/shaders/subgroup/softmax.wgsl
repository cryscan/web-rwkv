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

    if subgroup_invocation_id == 0u {
        sketch[subgroup_id] = _max_4;
    }
    workgroupBarrier();

    for (var step = num_subgroups >> 1u; step > 0u; step >>= 1u) {
        if index < step {
            sketch[index] = max(sketch[index], sketch[index + step]);
        }
        workgroupBarrier();
    }

    var _maximum: f32;
    if index == 0u {
        _max_4 = sketch[0];
        _maximum = _max_4.x;
        _maximum = max(_maximum, _max_4.y);
        _maximum = max(_maximum, _max_4.z);
        _maximum = max(_maximum, _max_4.w);
        maximum = _maximum;
    }
    workgroupBarrier();

    if subgroup_invocation_id == 0u {
        _maximum = maximum;
    }
    _maximum = subgroupBroadcast(_maximum, 0u);

    var _sum_4: vec4<f32>;
    for (var i = index; i < stride; i += BLOCK_SIZE) {
#ifdef FP16
        let value = unpack4x16float(x[bb + i]);
#else
        let value = x[bb + i];
#endif
        _sum_4 += exp(value - _maximum);
    }
    _sum_4 = subgroupAdd(_sum_4);

    if subgroup_invocation_id == 0u {
        sketch[subgroup_id] = _sum_4;
    }
    workgroupBarrier();

    for (var step = num_subgroups >> 1u; step > 0u; step >>= 1u) {
        if index < step {
            sketch[index] += sketch[index + step];
        }
        workgroupBarrier();
    }

    var _sum: f32;
    if index == 0u {
        _sum = dot(sketch[0], vec4<f32>(1.0));
        sum = _sum;
    }
    workgroupBarrier();

    if subgroup_invocation_id == 0u {
        _sum = sum;
    }
    _sum = subgroupBroadcast(_sum, 0u);

    for (var i = index; i < stride; i += BLOCK_SIZE) {
#ifdef FP16
        let value = unpack4x16float(x[bb + i]);
        x[bb + i] = pack4x16float(exp(value - _maximum) / _sum);
#else
        let value = x[bb + i];
        x[bb + i] = exp(value - _maximum) / _sum;
#endif
    }
}