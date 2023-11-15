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

fn unpack_matrix(index: u32, u: ptr<function, vec4<f32>>, v: ptr<function, vec4<f32>>) {
    let packed = matrix[index];
    let amp = unpack_absmax(index);

    var x: u32;
    x = (packed >> 0u) & 0xfu; (*u)[0] = quant[x >> 2u][x & 3u];
    x = (packed >> 4u) & 0xfu; (*u)[1] = quant[x >> 2u][x & 3u];
    x = (packed >> 8u) & 0xfu; (*u)[2] = quant[x >> 2u][x & 3u];
    x = (packed >> 12u) & 0xfu; (*u)[3] = quant[x >> 2u][x & 3u];
    x = (packed >> 16u) & 0xfu; (*v)[0] = quant[x >> 2u][x & 3u];
    x = (packed >> 20u) & 0xfu; (*v)[1] = quant[x >> 2u][x & 3u];
    x = (packed >> 24u) & 0xfu; (*v)[2] = quant[x >> 2u][x & 3u];
    x = (packed >> 28u) & 0xfu; (*v)[3] = quant[x >> 2u][x & 3u];

    *u *= amp;
    *v *= amp;
}

fn unpack_input(packed: vec4<u32>, u: ptr<function, vec4<f32>>, v: ptr<function, vec4<f32>>) {
    *u = unpack4x16float(packed.xy);
    *v = unpack4x16float(packed.zw);
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
        var m_0: vec4<f32>;
        var m_1: vec4<f32>;
        var m_2: vec4<f32>;
        var m_3: vec4<f32>;
        var n_0: vec4<f32>;
        var n_1: vec4<f32>;
        var n_2: vec4<f32>;
        var n_3: vec4<f32>;
        unpack_matrix(ci, &m_0, &n_0); ci += stride;
        unpack_matrix(ci, &m_1, &n_1); ci += stride;
        unpack_matrix(ci, &m_2, &n_2); ci += stride;
        unpack_matrix(ci, &m_3, &n_3);

        // read 8 elements from the input
        var x: vec4<f32>;
        var y: vec4<f32>;
        unpack_input(input[bti], &x, &y);

        let m = mat4x4<f32>(m_0, m_1, m_2, m_3);
        let n = mat4x4<f32>(n_0, n_1, n_2, n_3);

        local_sum += transpose(m) * x;
        local_sum += transpose(n) * y;
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