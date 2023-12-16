struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
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

@group(0) @binding(4) var<storage, read> absmax: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> xa: array<u32>;                    // (B, M, K)
@group(0) @binding(6) var<storage, read> xb: array<vec4<u32>>;              // (B, N, K)
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;    // (B, N, M)

const NF4_BLOCK_SIZE: u32 = 64u;

var<workgroup> sa: array<array<u32, 32u>, 32u>;
var<workgroup> sb: array<array<vec4<u32>, 32u>, 32u>;
var<workgroup> q: array<vec4<f32>, 4u>;

fn compute_index(view: View, z: u32, y: u32, x: u32, step: u32) -> u32 {
    let stride = view.stride.x / step;
    let offset = view.offset.x / step;
    return ((view.offset.z + z) * view.stride.y + view.offset.y + y) * stride + offset + x;
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
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

@compute @workgroup_size(8, 8, 1)
fn matmul(in: Input) {
    let b = in.bid.xy * 32u;
    let u = in.uid.xy * 4u;
    let t = in.tid.xy * 4u;
    let ra = vec2<u32>(va.shape.x / 4u, va.shape.y);
    let rb = vec2<u32>(vb.shape.x / 8u, vb.shape.y);
    let stride = min(ra.x, rb.x);
    let i = in.index & 31u;

    if in.index == 0u {
        q = quant;
    }
    workgroupBarrier();

    var local_sum: mat4x4<f32>;
    for (var k = 0u; k < stride; k += 32u) {
        // load 8x4 rows from each of the matrix, each with 32x8 columns
        // also, load 32 rows / 4 columes of absmax
        for (var j = 0u; j < 32u; j += 1u) {
            if in.index < 32u {
                let x = k + i;
                let y = b.x + j;
                if all(vec2<u32>(x, y) < ra) {
                    sa[j][i] = xa[compute_index(va, in.uid.z, y, x, 4u)];
                } else {
                    sa[j][i] = 0x77777777u;
                }
            } else {
                let x = k + i;
                let y = b.y + j;
                if all(vec2<u32>(x, y) < rb) {
                    sb[j][i] = xb[compute_index(vb, in.uid.z, y, x, 8u)];
                } else {
                    sb[j][i] = vec4<u32>(0u);
                }
            }
        }
        workgroupBarrier();

        // each thread multiplies and sums up 4x4 blocks along the reduced dimension
        if all(u < vec2<u32>(ra.y, rb.y)) {
            let ai = compute_index(va, in.uid.z, u.x, k, 4u);
            let a = array<vec4<f32>, 4>(
                unpack4x16float(absmax[ai / (NF4_BLOCK_SIZE / 2u)]),
                unpack4x16float(absmax[(ai + stride) / (NF4_BLOCK_SIZE / 2u)]),
                unpack4x16float(absmax[(ai + stride * 2u) / (NF4_BLOCK_SIZE / 2u)]),
                unpack4x16float(absmax[(ai + stride * 3u) / (NF4_BLOCK_SIZE / 2u)]),
            );

            for (var x = 0u; x < 32u; x += 1u) {
                let ssa = vec4<u32>(
                    sa[t.x][x],
                    sa[t.x + 1u][x],
                    sa[t.x + 2u][x],
                    sa[t.x + 3u][x],
                );

                var aa = mat4x4<f32>(
                    a[0][x / (NF4_BLOCK_SIZE / 8u)] * unpack_matrix_0(ssa[0]),
                    a[1][x / (NF4_BLOCK_SIZE / 8u)] * unpack_matrix_0(ssa[1]),
                    a[2][x / (NF4_BLOCK_SIZE / 8u)] * unpack_matrix_0(ssa[2]),
                    a[3][x / (NF4_BLOCK_SIZE / 8u)] * unpack_matrix_0(ssa[3]),
                );
                var bb = mat4x4<f32>(
                    unpack4x16float(sb[t.y][x].xy),
                    unpack4x16float(sb[t.y + 1u][x].xy),
                    unpack4x16float(sb[t.y + 2u][x].xy),
                    unpack4x16float(sb[t.y + 3u][x].xy),
                );
                local_sum += transpose(aa) * bb;

                aa = mat4x4<f32>(
                    a[0][x / (NF4_BLOCK_SIZE / 8u)] * unpack_matrix_1(ssa[0]),
                    a[1][x / (NF4_BLOCK_SIZE / 8u)] * unpack_matrix_1(ssa[1]),
                    a[2][x / (NF4_BLOCK_SIZE / 8u)] * unpack_matrix_1(ssa[2]),
                    a[3][x / (NF4_BLOCK_SIZE / 8u)] * unpack_matrix_1(ssa[3]),
                );
                bb = mat4x4<f32>(
                    unpack4x16float(sb[t.y][x].zw),
                    unpack4x16float(sb[t.y + 1u][x].zw),
                    unpack4x16float(sb[t.y + 2u][x].zw),
                    unpack4x16float(sb[t.y + 3u][x].zw),
                );
                local_sum += transpose(aa) * bb;
            }
        }
        workgroupBarrier();
    }

    if all(u < vec2<u32>(ra.y, rb.y)) {
        output[compute_index(destination, in.uid.z, u.y, in.uid.x, 4u)] = local_sum[0];
        output[compute_index(destination, in.uid.z, u.y + 1u, in.uid.x, 4u)] = local_sum[1];
        output[compute_index(destination, in.uid.z, u.y + 2u, in.uid.x, 4u)] = local_sum[2];
        output[compute_index(destination, in.uid.z, u.y + 3u, in.uid.x, 4u)] = local_sum[3];
    }
}