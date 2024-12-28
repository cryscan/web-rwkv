struct Input {
    @builtin(global_invocation_id) uid: vec3<u32>,
    @builtin(num_workgroups) b: vec3<u32>,
};

@group(0) @binding(0) var<uniform> len: vec4<u32>;                          // [L]
@group(0) @binding(1) var<uniform> shape: vec4<u32>;                        // [C, R]

@group(0) @binding(2) var<storage, read> input: array<vec2<u32>>;           // (R, C)

@group(0) @binding(3) var<storage, read_write> minmax: array<u32>;          // (R, C / S)
@group(0) @binding(4) var<storage, read_write> output: array<u32>;          // (R, C / 2)

const INT8_BLOCK_STEP: u32 = INT8_BLOCK_SIZE / 4u;

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn compute_minmax(in: Input) {
    let bti = in.uid.x;
    if bti >= len.x >> 1u {
        return;
    }

    let stride = shape.w * shape.z * shape.y * shape.x >> 2u;
    var _min = vec4<f32>(65504.0);
    var _max = vec4<f32>(-65504.0);
    for (var i = 0u; i < INT8_BLOCK_STEP; i += 1u) {
        let index = bti * INT8_BLOCK_STEP + i;
        if index < stride {
            let v = unpack4x16float(input[index]);
            _min = min(v, _min);
            _max = max(v, _max);
        }
    }

    _min[0] = min(min(_min[0], _min[1]), min(_min[2], _min[3]));
    _max[0] = max(max(_max[0], _max[1]), max(_max[2], _max[3]));
    minmax[bti] = pack2x16float(vec2<f32>(_min[0], _max[0]));
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn quantize(in: Input) {
    let bti = dot(in.uid, vec3<u32>(1u, BLOCK_SIZE * in.b.x, BLOCK_SIZE * in.b.x * in.b.y));
    if bti >= len.x >> 2u {
        return;
    }

    let m = unpack2x16float(minmax[bti / INT8_BLOCK_STEP]);
    let v = unpack4x16float(input[bti]);
    let x = saturate((v - m[0]) / (m[1] - m[0]));
    output[bti] = pack4x8unorm(x);
}
