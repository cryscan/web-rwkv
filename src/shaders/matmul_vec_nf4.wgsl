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
const QUANT = array<f32, 16>(
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.28444138,
    -0.18477343,
    -0.091050036,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
);

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;

fn compute_index(view: View, batch: u32, token: u32, index: u32, step: u32) -> u32 {
    let stride = view.stride.x / step;
    let offset = view.offset.x / step;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack_absmax(index: u32) -> f32 {
    let i = index / (NF4_BLOCK_SIZE / 8u);              // 1 block of absmax: NF4_BLOCK_SIZE / 8u entries in matrix
    return unpack2x16float(absmax[i >> 1u])[i & 1u];
}

fn unpack_matrix_0(v: u32) -> vec4<f32> {
    var quant = QUANT;
    // x[0] = quant[(v >> 0u) & 0xfu];
    // x[1] = quant[(v >> 4u) & 0xfu];
    // x[2] = quant[(v >> 8u) & 0xfu];
    // x[3] = quant[(v >> 12u) & 0xfu];
    // x[0] = quant[(v >> 2u) & 3u][(v >> 0u) & 3u];
    // x[1] = quant[(v >> 6u) & 3u][(v >> 4u) & 3u];
    // x[2] = quant[(v >> 10u) & 3u][(v >> 8u) & 3u];
    // x[3] = quant[(v >> 14u) & 3u][(v >> 12u) & 3u];
    return vec4<f32>(
        quant[(v & 0x0000000fu)],
        quant[(v & 0x000000f0u) >> 4u],
        quant[(v & 0x00000f00u) >> 8u],
        quant[(v & 0x0000f000u) >> 12u]
    );
}

fn unpack_matrix_1(v: u32) -> vec4<f32> {
    var quant = QUANT;
    // x[0] = quant[(v >> 16u) & 0xfu];
    // x[1] = quant[(v >> 20u) & 0xfu];
    // x[2] = quant[(v >> 24u) & 0xfu];
    // x[3] = quant[(v >> 28u) & 0xfu];
    // x[0] = quant[(v >> 18u) & 3u][(v >> 16u) & 3u];
    // x[1] = quant[(v >> 22u) & 3u][(v >> 20u) & 3u];
    // x[2] = quant[(v >> 26u) & 3u][(v >> 24u) & 3u];
    // x[3] = quant[(v >> 30u) & 3u][(v >> 28u) & 3u];
    return vec4<f32>(
        quant[(v & 0x000f0000u) >> 16u],
        quant[(v & 0x00f00000u) >> 20u],
        quant[(v & 0x0f000000u) >> 24u],
        quant[(v & 0xf0000000u) >> 28u]
    );
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
        // read 4 rows from the matrix, each with 4x2 unpacked floats, forming 2 4x4 sub-blocks
        var ci = cb + i;
        var v: vec4<u32>;
        var a: vec4<f32>;
        v[0] = matrix[ci]; a[0] = unpack_absmax(ci); ci += stride;
        v[1] = matrix[ci]; a[1] = unpack_absmax(ci); ci += stride;
        v[2] = matrix[ci]; a[2] = unpack_absmax(ci); ci += stride;
        v[3] = matrix[ci]; a[3] = unpack_absmax(ci);

        // read 8 elements from the input
        let x = input[bb + i];

        var m: mat4x4<f32>;
        m[0] = unpack_matrix_0(v[0]);
        m[1] = unpack_matrix_0(v[1]);
        m[2] = unpack_matrix_0(v[2]);
        m[3] = unpack_matrix_0(v[3]);
        var s = transpose(m) * unpack4x16float(x.xy);

        m[0] = unpack_matrix_1(v[0]);
        m[1] = unpack_matrix_1(v[1]);
        m[2] = unpack_matrix_1(v[2]);
        m[3] = unpack_matrix_1(v[3]);
        s += transpose(m) * unpack4x16float(x.zw);

        local_sum = fma(s, a, local_sum);
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