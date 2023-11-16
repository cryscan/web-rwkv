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

fn unpack_matrix_0(packed: u32) -> vec4<f32> {
    var x: vec4<f32>;
    x[0] = quant[(packed >> (2u)) & 3u][(packed >> (0u)) & 3u];
    x[1] = quant[(packed >> (6u)) & 3u][(packed >> (4u)) & 3u];
    x[2] = quant[(packed >> (10u)) & 3u][(packed >> (8u)) & 3u];
    x[3] = quant[(packed >> (14u)) & 3u][(packed >> (12u)) & 3u];
    return x;
}

fn unpack_matrix_1(packed: u32) -> vec4<f32> {
    var x: vec4<f32>;
    x[0] = quant[(packed >> (18u)) & 3u][(packed >> (16u)) & 3u];
    x[1] = quant[(packed >> (22u)) & 3u][(packed >> (20u)) & 3u];
    x[2] = quant[(packed >> (26u)) & 3u][(packed >> (24u)) & 3u];
    x[3] = quant[(packed >> (30u)) & 3u][(packed >> (28u)) & 3u];
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
        var v: u32;
        var a: vec4<f32>;
        var m: mat4x4<f32>;
        var n: mat4x4<f32>;
        v = matrix[ci]; a[0] = unpack_absmax(ci); m[0] = unpack_matrix_0(v); n[0] = unpack_matrix_1(v); ci += stride;
        v = matrix[ci]; a[1] = unpack_absmax(ci); m[1] = unpack_matrix_0(v); n[1] = unpack_matrix_1(v); ci += stride;
        v = matrix[ci]; a[2] = unpack_absmax(ci); m[2] = unpack_matrix_0(v); n[2] = unpack_matrix_1(v); ci += stride;
        v = matrix[ci]; a[3] = unpack_absmax(ci); m[3] = unpack_matrix_0(v); n[3] = unpack_matrix_1(v);

        // read 8 elements from the input
        let packed = input[bti];
        let x = unpack4x16float(packed.xy);
        let y = unpack4x16float(packed.zw);

        local_sum += a * (transpose(m) * x + transpose(n) * y);
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