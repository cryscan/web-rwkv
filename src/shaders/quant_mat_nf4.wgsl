@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C / S, R]. [C / 2, R]
@group(0) @binding(1) var<uniform> quant: array<vec4<f32>, 4>; 

@group(0) @binding(2) var<storage, read> input: array<vec4<u32>>;           // (R, C)

@group(0) @binding(3) var<storage, read_write> absmax: array<f32>;          // (R, C / S)
@group(0) @binding(4) var<storage, read_write> output: array<u32>;          // (R, C / 2)

var<workgroup> q: array<f32, 16>;

const BLOCK_SIZE: u32 = 128u;
const NF4_BLOCK_SIZE: u32 = 64u;

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

struct Input {
    @builtin(global_invocation_id) uid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(num_workgroups) b: vec3<u32>,
};

@compute @workgroup_size(128, 1, 1)
fn compute_absmax(in: Input) {
    let step = NF4_BLOCK_SIZE / 8u;
    let bti = in.uid.x + (BLOCK_SIZE * in.b.x) * in.uid.y + (BLOCK_SIZE * in.b.x * in.b.y) * in.uid.z;

    var maximum = vec4<f32>(0.0);
    for (var i = 0u; i < step; i += 1u) {
        let v = input[bti * step + i];
        let x = unpack4x16float(v.xy);
        let y = unpack4x16float(v.zw);

        maximum = max(abs(x), maximum);
        maximum = max(abs(y), maximum);
    }
    absmax[bti] = max(max(maximum[0], maximum[1]), max(maximum[2], maximum[3]));
}

struct MinPayload {
    min_value: f32,
    min_index: u32,
};

fn argmin(acc: MinPayload, x: MinPayload) -> MinPayload {
    if acc.min_value < x.min_value {
        return acc;
    } else {
        return x;
    }
}

@compute @workgroup_size(128, 1, 1)
fn quantize(in: Input) {
    let step = NF4_BLOCK_SIZE / 8u;
    let bti = in.uid.x + (BLOCK_SIZE * in.b.x) * in.uid.y + (BLOCK_SIZE * in.b.x * in.b.y) * in.uid.z;

    if in.tid.x <= 16u {
        q[in.tid.x] = quant[in.tid.x >> 2u][in.tid.x & 3u];
    }
    workgroupBarrier();

    let amp = 1.0 / absmax[bti / step];
    let v = input[bti];
    var x: array<vec4<f32>, 2>;
    x[0] = unpack4x16float(v.xy) * amp;
    x[1] = unpack4x16float(v.zw) * amp;

    var y = 0u;
    for (var i = 0u; i < 8u; i += 1u) {
        var mp: MinPayload;
        mp.min_index = 0u;
        mp.min_value = 2.0;
        let xx = x[i >> 2u][i & 3u];

        for (var j = 0u; j < 16u; j += 1u) {
            let qq = q[j];
            var mx: MinPayload;
            mx.min_index = j;
            mx.min_value = abs(qq - xx);
            mp = argmin(mp, mx);
        }
        y |= mp.min_index << (i * 4u);
    }

    output[bti] = y;
}
