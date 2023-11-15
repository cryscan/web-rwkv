struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
};

@group(0) @binding(1) var<uniform> source: View;                            // [R, T, B]
@group(0) @binding(2) var<uniform> destination: View;                       // [R, T, B]
@group(0) @binding(3) var<uniform> quant: array<vec4<f32>, 4>;

@group(0) @binding(4) var<storage, read> matrix: array<u32>;                // (R, C)
@group(0) @binding(5) var<storage, read> absmax: array<u32>;

@group(0) @binding(6) var<storage, read> input: array<vec4<u32>>;           // (B, T, C)
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, R)

const BLOCK_SIZE: u32 = 128u;
const NF4_BLOCK_SIZE: u32 = 64u;

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;

fn compute_index(view: View, batch: u32, token: u32, index: u32, step: u32) -> u32 {
    let stride = view.stride.x / step;
    let offset = view.offset.x / step;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn unpack_absmax(index: u32) -> f32 {
    let i = index / (NF4_BLOCK_SIZE / 8u);              // 1 block of absmax: NF4_BLOCK_SIZE / 8u entries in matrix
    return unpack2x16float(absmax[i >> 1u])[i & 1u];
}

fn unpack_matrix(index: u32) -> array<vec4<f32>, 2> {
    let packed = matrix[index];
    var i: u32;
    var x: array<vec4<f32>, 2>;
    i = (packed >> 0u) & 0xfu; x[0][0] = quant[i >> 2u][i & 3u];
    i = (packed >> 4u) & 0xfu; x[0][1] = quant[i >> 2u][i & 3u];
    i = (packed >> 8u) & 0xfu; x[0][2] = quant[i >> 2u][i & 3u];
    i = (packed >> 12u) & 0xfu; x[0][3] = quant[i >> 2u][i & 3u];
    i = (packed >> 16u) & 0xfu; x[1][0] = quant[i >> 2u][i & 3u];
    i = (packed >> 20u) & 0xfu; x[1][1] = quant[i >> 2u][i & 3u];
    i = (packed >> 24u) & 0xfu; x[1][2] = quant[i >> 2u][i & 3u];
    i = (packed >> 28u) & 0xfu; x[1][3] = quant[i >> 2u][i & 3u];

    let amp = unpack_absmax(index);
    x[0] *= amp;
    x[1] *= amp;

    return x;
}

fn unpack_input(packed: vec4<u32>) -> array<vec4<f32>, 2> {
    var x: array<vec4<f32>, 2>;
    x[0] = unpack4x16float(packed.xy);
    x[1] = unpack4x16float(packed.zw);
    return x;
}

fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(128, 1, 1)
fn matmul(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = source.stride.x / 8u;
    let index = invocation_id.x % BLOCK_SIZE;
    let channel = invocation_id.x / BLOCK_SIZE;     // 1 channel: 4 rows in matrix
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = compute_index(source, batch, token, 0u, 8u);
    let cb = channel * 4u * stride;

    var local_sum = vec4<f32>(0.0);
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let bti = bb + i;
        var ci = cb + i;

        // read 4 rows from the matrix, each with 4x2 unpacked floats, forming 2 4x4 sub-blocks
        let m_0 = unpack_matrix(ci); ci += stride;
        let m_1 = unpack_matrix(ci); ci += stride;
        let m_2 = unpack_matrix(ci); ci += stride;
        let m_3 = unpack_matrix(ci);

        // read 8 elements from the input
        let x = unpack_input(input[bti]);

        let m = mat4x4<f32>(m_0[0], m_1[0], m_2[0], m_3[0]);
        let n = mat4x4<f32>(m_0[1], m_1[1], m_2[1], m_3[1]);

        local_sum += transpose(m) * x[0];
        local_sum += transpose(n) * x[1];
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
        let btc = compute_index(destination, batch, token, channel, 4u);
        output[btc] = sketch[0];
    }
}