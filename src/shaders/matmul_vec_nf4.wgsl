struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
};

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, R]
@group(0) @binding(1) var<uniform> source: View;                            // [R, T, B]
@group(0) @binding(2) var<uniform> destination: View;                       // [R, T, B]
@group(0) @binding(3) var<uniform> quantile: array<vec4<f32>, 4>;

@group(0) @binding(4) var<storage, read> matrix: array<u32>;                // (R, C)
@group(0) @binding(5) var<storage, read> absmax: array<f32>;

@group(0) @binding(6) var<storage, read> input: array<vec4<u32>>;           // (B, T, C)
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, R)

const BLOCK_SIZE: u32 = 128u;

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;

fn compute_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x / 4u;
    let offset = view.offset.x / 4u;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn unpack(packed: u32, amp: f32) -> array<vec4<f32>, 2> {
    var unpacked: array<vec4<f32>, 2>;
    var index: u32;

    index = (packed >> 0u) & 0xfu; unpacked[0][0] = quantile[index >> 2u][index & 3u];
    index = (packed >> 4u) & 0xfu; unpacked[0][1] = quantile[index >> 2u][index & 3u];
    index = (packed >> 8u) & 0xfu; unpacked[0][2] = quantile[index >> 2u][index & 3u];
    index = (packed >> 12u) & 0xfu; unpacked[0][3] = quantile[index >> 2u][index & 3u];
    index = (packed >> 16u) & 0xfu; unpacked[1][0] = quantile[index >> 2u][index & 3u];
    index = (packed >> 20u) & 0xfu; unpacked[1][1] = quantile[index >> 2u][index & 3u];
    index = (packed >> 24u) & 0xfu; unpacked[1][2] = quantile[index >> 2u][index & 3u];
    index = (packed >> 28u) & 0xfu; unpacked[1][3] = quantile[index >> 2u][index & 3u];

    unpacked[0] = unpacked[0] * amp;
    unpacked[1] = unpacked[1] * amp;
    return unpacked;
}

fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(128, 1, 1)
fn matmul(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape.x / 8u;
    let index = invocation_id.x % BLOCK_SIZE;
    let channel = invocation_id.x / BLOCK_SIZE;   // 1 channel: 4 rows in matrix
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = compute_index(source, batch, token, 0u);
    let cb = channel * 4u * stride;

    var local_sum = vec4<f32>(0.0);
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let bti = bb + i;
        var ci = cb + i;

        // read 8 elements from the input
        let x = input[bti];

        // read 4 rows from the matrix, each with 4 unpacked floats, forming a 4x4 sub-block
        var m: mat4x4<f32>;

        m[0] = unpack4x16float(matrix[ci]); ci += stride;
        m[1] = unpack4x16float(matrix[ci]); ci += stride;
        m[2] = unpack4x16float(matrix[ci]); ci += stride;
        m[3] = unpack4x16float(matrix[ci]);
        local_sum += transpose(m) * x;
    }
    sketch[index] = local_sum;
    workgroupBarrier();

    reduce_sum(index, 64u);
    reduce_sum(index, 32u);
    reduce_sum(index, 16u);
    reduce_sum(index, 8u);
    reduce_sum(index, 4u);
    reduce_sum(index, 2u);
    reduce_sum(index, 1u);

    if index == 0u {
        let btc = compute_index(destination, batch, token, channel);
        output[btc] = sketch[0];
    }
}