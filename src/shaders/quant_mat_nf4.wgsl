@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C / 2, R]
@group(0) @binding(1) var<uniform> quant: array<vec4<f32>, 4>; 

@group(0) @binding(2) var<storage, read> input: array<vec4<u32>>;           // (R, C)

@group(0) @binding(3) var<storage, read_write> absmax: array<f32>;          // (R, C / S)
@group(0) @binding(4) var<storage, read_write> output: array<u32>;          // (R, C / 2)

const BLOCK_SIZE: u32 = 128u;
const NF4_BLOCK_SIZE: u32 = 64u;

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn unpack_input(packed: vec4<u32>) -> array<vec4<f32>, 2> {
    var x: array<vec4<f32>, 2>;
    x[0] = unpack4x16float(packed.xy);
    x[1] = unpack4x16float(packed.zw);
    return x;
}

@compute @workgroup_size(128, 1, 1)
fn compute_absmax(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let stride = NF4_BLOCK_SIZE / 8u;

    var maximum = vec4<f32>(0.0);
    for (var i = 0u; i < stride; i += 1u) {
        let x = unpack_input(input[index * stride + i]);
        maximum = max(abs(x[0]), maximum);
        maximum = max(abs(x[1]), maximum);
    }
    absmax[index] = max(max(maximum[0], maximum[1]), max(maximum[2], maximum[3]));
}

@compute @workgroup_size(128, 1, 1)
fn quantize(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let stride = NF4_BLOCK_SIZE / 8u;

    let amp = absmax[index / stride];
    var x = unpack_input(input[index]);
    x[0] /= amp;
    x[1] /= amp;

    var y = 0u;
    for (var i = 0u; i < 8u; i += 1u) {
        var min_dist = 2.0;
        var min_index = 0u;
        let xx = x[i >> 2u][i & 3u];
        for (var j = 0u; j < 16u; j += 1u) {
            let qq = quant[j >> 2u][j & 3u];
            if abs(qq - xx) <= min_dist {
                min_dist = abs(qq - xx);
                min_index = j;
            }
        }
        y |= min_index << (i * 4u);
    }

    output[index] = y;
}
