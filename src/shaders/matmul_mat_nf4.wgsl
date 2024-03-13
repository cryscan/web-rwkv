struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

struct Input {
    @builtin(workgroup_id) bid: vec3<u32>,
    @builtin(global_invocation_id) uid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
};

@group(0) @binding(0) var<uniform> va: View;                                // [K, M, B]
@group(0) @binding(1) var<uniform> vb: View;                                // [K, N, B]
@group(0) @binding(2) var<uniform> destination: View;                       // [M, N, B]
@group(0) @binding(3) var<uniform> quant: array<vec4<f32>, 4u>;

@group(0) @binding(4) var<storage, read> absmax: array<u32>;
@group(0) @binding(5) var<storage, read> xa: array<u32>;                    // (B, M, K)
#ifdef IN_FP16
@group(0) @binding(6) var<storage, read> xb: array<vec4<u32>>;              // (B, N, K)
#else
@group(0) @binding(6) var<storage, read> xb: array<mat2x4<f32>>;            // (B, N, K)
#endif
#ifdef OUT_FP16
@group(0) @binding(7) var<storage, read_write> output: array<vec2<u32>>;    // (B, N, M)
#else
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;    // (B, N, M)
#endif

const TILE_SIZE: u32 = BLOCK_SIZE * 4u;
const NF4_BLOCK_STEP: u32 = NF4_BLOCK_SIZE / 8u;

var<workgroup> sa: array<array<u32, BLOCK_SIZE>, TILE_SIZE>;
#ifdef IN_FP16
var<workgroup> sb: array<array<vec4<u32>, BLOCK_SIZE>, TILE_SIZE>;
#else
var<workgroup> sb: array<array<mat2x4<f32>, BLOCK_SIZE>, TILE_SIZE>;
#endif
var<workgroup> q: array<vec4<f32>, 4u>;

fn compute_index(view: View, batch: u32, token: u32, index: u32, step: u32) -> u32 {
    let stride = view.stride.x / step;
    let offset = vec3<u32>(view.offset.zy, view.offset.x / step);
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

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn matmul(in: Input) {
    let b = in.bid.xy * TILE_SIZE;
    let u = in.uid.xy * 4u;
    let t = in.tid.xy * 4u;
    let ra = vec2<u32>(va.shape.x / 4u, va.shape.y);
    let rb = vec2<u32>(vb.shape.x / 8u, vb.shape.y);
    let stride = min(ra.x, rb.x);

    if in.index == 0u {
        q = quant;
    }

    var local_sum: mat4x4<f32>;
    for (var k = 0u; k < stride; k += BLOCK_SIZE) {
        // load 8x4 rows from each of the matrix, each with 8x8 columns
        // also, load 32 rows / 4 columes of absmax
        for (var j = in.tid.y; j < TILE_SIZE; j += BLOCK_SIZE) {
            let i = in.tid.x;
            let x = k + i;
            var y = b.x + j;
            if all(vec2<u32>(x, y) < ra) {
                sa[j][i] = xa[compute_index(va, in.uid.z, y, x, 4u)];
            } else {
                sa[j][i] = 0x77777777u;
            }

            y = b.y + j;
            if all(vec2<u32>(x, y) < rb) {
                sb[j][i] = xb[compute_index(vb, in.uid.z, y, x, 8u)];
            } else {
#ifdef IN_FP16
                sb[j][i] = vec4<u32>(0u);
#else
                sb[j][i] = mat2x4<f32>();
#endif
            }
        }
        workgroupBarrier();

        // each thread multiplies and sums up 4x4 blocks along the reduced dimension
        if all(u < vec2<u32>(ra.y, rb.y)) {
            var i = compute_index(va, in.uid.z, u.x, k, 4u);
            var a: vec4<f32>;
            a[0] = unpack_absmax(i); i += stride;
            a[1] = unpack_absmax(i); i += stride;
            a[2] = unpack_absmax(i); i += stride;
            a[3] = unpack_absmax(i);

            for (var x = 0u; x < BLOCK_SIZE; x += 1u) {
                if k + x >= stride {
                    break;
                }
                let la = vec4<u32>(
                    sa[t.x][x],
                    sa[t.x + 1u][x],
                    sa[t.x + 2u][x],
                    sa[t.x + 3u][x],
                );

                var aa = mat4x4<f32>(
                    a[0] * unpack_matrix_0(la[0]),
                    a[1] * unpack_matrix_0(la[1]),
                    a[2] * unpack_matrix_0(la[2]),
                    a[3] * unpack_matrix_0(la[3]),
                );
#ifdef IN_FP16
                var bb = mat4x4<f32>(
                    unpack4x16float(sb[t.y][x].xy),
                    unpack4x16float(sb[t.y + 1u][x].xy),
                    unpack4x16float(sb[t.y + 2u][x].xy),
                    unpack4x16float(sb[t.y + 3u][x].xy),
                );
#else
                var bb = mat4x4<f32>(
                    sb[t.y][x][0],
                    sb[t.y + 1u][x][0],
                    sb[t.y + 2u][x][0],
                    sb[t.y + 3u][x][0],
                );
#endif
                local_sum += transpose(aa) * bb;

                aa = mat4x4<f32>(
                    a[0] * unpack_matrix_1(la[0]),
                    a[1] * unpack_matrix_1(la[1]),
                    a[2] * unpack_matrix_1(la[2]),
                    a[3] * unpack_matrix_1(la[3]),
                );
#ifdef IN_FP16
                bb = mat4x4<f32>(
                    unpack4x16float(sb[t.y][x].zw),
                    unpack4x16float(sb[t.y + 1u][x].zw),
                    unpack4x16float(sb[t.y + 2u][x].zw),
                    unpack4x16float(sb[t.y + 3u][x].zw),
                );
#else
                bb = mat4x4<f32>(
                    sb[t.y][x][1],
                    sb[t.y + 1u][x][1],
                    sb[t.y + 2u][x][1],
                    sb[t.y + 3u][x][1],
                );
#endif
                local_sum += transpose(aa) * bb;
            }
        }
        workgroupBarrier();
    }

    if all(u < vec2<u32>(ra.y, rb.y)) {
#ifdef ACT_SQUARED_RELU
        local_sum[0] = squared_relu(local_sum[0]);
        local_sum[1] = squared_relu(local_sum[1]);
        local_sum[2] = squared_relu(local_sum[2]);
        local_sum[3] = squared_relu(local_sum[3]);
#else
#ifdef ACT_TANH
        local_sum[0] = tanh(local_sum[0]);
        local_sum[1] = tanh(local_sum[1]);
        local_sum[2] = tanh(local_sum[2]);
        local_sum[3] = tanh(local_sum[3]);
#endif
#endif
#ifdef OUT_FP16
        output[compute_index(destination, in.uid.z, u.y + 0u, in.uid.x, 4u)] = pack4x16float(local_sum[0]);
        output[compute_index(destination, in.uid.z, u.y + 1u, in.uid.x, 4u)] = pack4x16float(local_sum[1]);
        output[compute_index(destination, in.uid.z, u.y + 2u, in.uid.x, 4u)] = pack4x16float(local_sum[2]);
        output[compute_index(destination, in.uid.z, u.y + 3u, in.uid.x, 4u)] = pack4x16float(local_sum[3]);
#else
        output[compute_index(destination, in.uid.z, u.y + 0u, in.uid.x, 4u)] = local_sum[0];
        output[compute_index(destination, in.uid.z, u.y + 1u, in.uid.x, 4u)] = local_sum[1];
        output[compute_index(destination, in.uid.z, u.y + 2u, in.uid.x, 4u)] = local_sum[2];
        output[compute_index(destination, in.uid.z, u.y + 3u, in.uid.x, 4u)] = local_sum[3];
#endif
    }
}