struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, R, B]
@group(0) @binding(1) var<uniform> source: View;                            // [R, T, B]
@group(0) @binding(2) var<uniform> destination: View;                       // [R, T, B]
@group(0) @binding(3) var<uniform> quant: array<vec4<f32>, 4u>;

@group(0) @binding(4) var<storage, read> matrix: array<u32>;                // (B, R, C)
@group(0) @binding(5) var<storage, read> absmax: array<u32>;

#ifdef IN_FP16
@group(0) @binding(6) var<storage, read> input: array<vec4<u32>>;           // (B, T, C)
#else
@group(0) @binding(6) var<storage, read> input: array<mat2x4<f32>>;         // (B, T, C)
#endif
#ifdef OUT_FP16
@group(0) @binding(7) var<storage, read_write> output: array<vec2<u32>>;    // (B, T, R)
#else
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, R)
#endif

const NF4_BLOCK_STEP: u32 = NF4_BLOCK_SIZE / 8u;

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> q: array<vec4<f32>, 4u>;

fn compute_index(view: View, batch: u32, token: u32, index: u32, step: u32) -> u32 {
    let stride = view.stride.x >> step;
    let offset = vec3<u32>(view.offset.zy, view.offset.x >> step);
    return dot(vec3<u32>(batch, token, index) + offset, vec3<u32>(view.stride.y * stride, stride, 1u));
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn unpack_absmax(index: u32) -> f32 {
    let i = index / NF4_BLOCK_STEP; // 1 block of absmax: NF4_BLOCK_SIZE / 8u entries in matrix
    return unpack2x16float(absmax[i >> 1u])[i & 1u];
}

fn unpack_matrix_0(v: u32) -> vec4<f32> {
    let i = vec4<u32>(
        (v & 0x0000000fu),
        (v & 0x000000f0u) >> 4u,
        (v & 0x00000f00u) >> 8u,
        (v & 0x0000f000u) >> 12u,
    );
    return vec4<f32>(
        q[i.x >> 2u][i.x & 3u],
        q[i.y >> 2u][i.y & 3u],
        q[i.z >> 2u][i.z & 3u],
        q[i.w >> 2u][i.w & 3u],
    );
}

fn unpack_matrix_1(v: u32) -> vec4<f32> {
    let i = vec4<u32>(
        (v & 0x000f0000u) >> 16u,
        (v & 0x00f00000u) >> 20u,
        (v & 0x0f000000u) >> 24u,
        (v & 0xf0000000u) >> 28u,
    );
    return vec4<f32>(
        q[i.x >> 2u][i.x & 3u],
        q[i.y >> 2u][i.y & 3u],
        q[i.z >> 2u][i.z & 3u],
        q[i.w >> 2u][i.w & 3u],
    );
}

fn squared_relu(x: vec4<f32>) -> vec4<f32> {
    let p = max(x, vec4<f32>(0.0));
    return p * p;
}

fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn matmul(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = source.stride.x / 8u;
    let index = invocation_id.x % BLOCK_SIZE;
    let channel = invocation_id.x / BLOCK_SIZE;     // 1 channel: 4 rows in matrix
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = compute_index(source, batch, token, 0u, 3u);
    let cb = batch * shape.y * stride + channel * 4u * stride;

    if index == 0u {
        q = quant;
    }

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
        m = transpose(m);
#ifdef IN_FP16
        local_sum = fma(m * unpack4x16float(x.xy), a, local_sum);
#else
        local_sum = fma(m * x[0], a, local_sum);
#endif

        m[0] = unpack_matrix_1(v[0]);
        m[1] = unpack_matrix_1(v[1]);
        m[2] = unpack_matrix_1(v[2]);
        m[3] = unpack_matrix_1(v[3]);
        m = transpose(m);
#ifdef IN_FP16
        local_sum = fma(m * unpack4x16float(x.zw), a, local_sum);
#else
        local_sum = fma(m * x[1], a, local_sum);
#endif
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
        let btc = compute_index(destination, batch, token, channel, 2u);
#ifdef ACT_SQUARED_RELU
        let out = squared_relu(sketch[0]);
#else
#ifdef ACT_TANH
        let out = tanh(sketch[0]);
#else
        let out = sketch[0];
#endif
#endif
#ifdef OUT_FP16
        output[btc] = pack4x16float(out);
#else
        output[btc] = out;
#endif
    }
}