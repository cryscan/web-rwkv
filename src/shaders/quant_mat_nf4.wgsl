struct Input {
    @builtin(global_invocation_id) uid: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
    @builtin(num_workgroups) b: vec3<u32>,
};

@group(0) @binding(0) var<uniform> len: vec4<u32>;                          // [L]
@group(0) @binding(1) var<uniform> shape: vec4<u32>;                        // [C, R]
@group(0) @binding(2) var<uniform> quant: array<vec4<f32>, 4>; 

@group(0) @binding(3) var<storage, read> input: array<vec4<u32>>;           // (R, C)

@group(0) @binding(4) var<storage, read_write> absmax: array<f32>;          // (R, C / S)
@group(0) @binding(5) var<storage, read_write> output: array<u32>;          // (R, C / 2)

var<workgroup> q: array<vec4<f32>, 4u>;

const NF4_BLOCK_STEP: u32 = NF4_BLOCK_SIZE / 8u;

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn compute_absmax(in: Input) {
    let bti = dot(in.uid, vec3<u32>(1u, BLOCK_SIZE * in.b.x, BLOCK_SIZE * in.b.x * in.b.y));
    if bti >= len.x {
        return;
    }

    let stride = shape.w * shape.z * shape.y * shape.x >> 2u;
    var _max = vec4<f32>(0.0);
    for (var i = 0u; i < NF4_BLOCK_STEP; i += 1u) {
        let index = bti * NF4_BLOCK_STEP + i;
        if index < stride {
            let v = input[index];
            let x = unpack4x16float(v.xy);
            let y = unpack4x16float(v.zw);

            _max = max(abs(x), _max);
            _max = max(abs(y), _max);
        }
    }
    absmax[bti] = max(max(_max[0], _max[1]), max(_max[2], _max[3]));
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn quantize(in: Input) {
    let bti = dot(in.uid, vec3<u32>(1u, BLOCK_SIZE * in.b.x, BLOCK_SIZE * in.b.x * in.b.y));
    if bti >= len.x >> 2u {
        return;
    }

    if in.index == 0u {
        q = quant;
    }
    workgroupBarrier();

    let a = 1.0 / absmax[bti / NF4_BLOCK_STEP];
    let v = input[bti];
    var x: array<vec4<f32>, 2>;
    x[0] = unpack4x16float(v.xy) * a;
    x[1] = unpack4x16float(v.zw) * a;

    var y = 0u;
    for (var i = 0u; i < 8u; i += 1u) {
        var min_err = 1.0;
        var min_index = 0u;
        let xx = x[i >> 2u][i & 3u];
        for (var j = 0u; j < 16u; j += 1u) {
            let qq = q[j >> 2u][j & 3u];
            if abs(qq - xx) <= min_err {
                min_err = abs(qq - xx);
                min_index = j;
            }
        }
        y |= min_index << (i * 4u);
    }

    output[bti] = y;
}
