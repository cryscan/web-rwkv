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

@group(0) @binding(3) var<storage, read> xa: array<vec2<u32>>;              // (B, M, K)
#ifdef IN_FP16
@group(0) @binding(4) var<storage, read> xb: array<vec2<u32>>;              // (B, N, K)
#else
@group(0) @binding(4) var<storage, read> xb: array<vec4<f32>>;              // (B, N, K)
#endif
#ifdef OUT_FP16
@group(0) @binding(5) var<storage, read_write> output: array<vec2<u32>>;    // (B, N, M)
#else
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;    // (B, N, M)
#endif

const TILE_SIZE: u32 = BLOCK_SIZE * 4u;

var<workgroup> sa: array<array<vec2<u32>, BLOCK_SIZE>, TILE_SIZE>;
#ifdef IN_FP16
var<workgroup> sb: array<array<vec2<u32>, BLOCK_SIZE>, TILE_SIZE>;
#else
var<workgroup> sb: array<array<vec4<f32>, BLOCK_SIZE>, TILE_SIZE>;
#endif

fn compute_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x >> 2u;
    let offset = vec3<u32>(view.offset.zy, view.offset.x >> 2u);
    return dot(vec3<u32>(batch, token, index) + offset, vec3<u32>(view.stride.y * stride, stride, 1u));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
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
    let rb = vec2<u32>(vb.shape.x / 4u, vb.shape.y);
    let stride = min(ra.x, rb.x);

    var local_sum: mat4x4<f32>;
    for (var k = 0u; k < stride; k += BLOCK_SIZE) {
        // load 8x4 rows from each of the matrix, each with 8x4 columns
        for (var j = in.tid.y; j < TILE_SIZE; j += BLOCK_SIZE) {
            let i = in.tid.x;
            let x = k + i;
            var y = b.x + j;
            if all(vec2<u32>(x, y) < ra) {
                sa[j][i] = xa[compute_index(va, in.uid.z, y, x)];
            } else {
                sa[j][i] = vec2<u32>(0u);
            }

            y = b.y + j;
            if all(vec2<u32>(x, y) < rb) {
                sb[j][i] = xb[compute_index(vb, in.uid.z, y, x)];
            } else {
#ifdef IN_FP16
                sb[j][i] = vec2<u32>(0u);
#else
                sb[j][i] = vec4<f32>(0.0);
#endif
            }
        }
        workgroupBarrier();

        // each thread multiplies and sums up 4x4 blocks along the reduced dimension
        if all(u < vec2<u32>(ra.y, rb.y)) {
            for (var x = 0u; x < BLOCK_SIZE; x += 1u) {
                if k + x >= stride {
                    break;
                }
                let aa = mat4x4<f32>(
                    unpack4x16float(sa[t.x][x]),
                    unpack4x16float(sa[t.x + 1u][x]),
                    unpack4x16float(sa[t.x + 2u][x]),
                    unpack4x16float(sa[t.x + 3u][x]),
                );
#ifdef IN_FP16
                let bb = mat4x4<f32>(
                    unpack4x16float(sb[t.y][x]),
                    unpack4x16float(sb[t.y + 1u][x]),
                    unpack4x16float(sb[t.y + 2u][x]),
                    unpack4x16float(sb[t.y + 3u][x]),
                );
#else
                let bb = mat4x4<f32>(
                    sb[t.y][x],
                    sb[t.y + 1u][x],
                    sb[t.y + 2u][x],
                    sb[t.y + 3u][x],
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
#endif
#ifdef ACT_TANH
        local_sum[0] = tanh(local_sum[0]);
        local_sum[1] = tanh(local_sum[1]);
        local_sum[2] = tanh(local_sum[2]);
        local_sum[3] = tanh(local_sum[3]);
#endif
#ifdef OUT_FP16
        output[compute_index(destination, in.uid.z, u.y + 0u, in.uid.x)] = pack4x16float(local_sum[0]);
        output[compute_index(destination, in.uid.z, u.y + 1u, in.uid.x)] = pack4x16float(local_sum[1]);
        output[compute_index(destination, in.uid.z, u.y + 2u, in.uid.x)] = pack4x16float(local_sum[2]);
        output[compute_index(destination, in.uid.z, u.y + 3u, in.uid.x)] = pack4x16float(local_sum[3]);
#else
        output[compute_index(destination, in.uid.z, u.y + 0u, in.uid.x)] = local_sum[0];
        output[compute_index(destination, in.uid.z, u.y + 1u, in.uid.x)] = local_sum[1];
        output[compute_index(destination, in.uid.z, u.y + 2u, in.uid.x)] = local_sum[2];
        output[compute_index(destination, in.uid.z, u.y + 3u, in.uid.x)] = local_sum[3];
#endif
    }
}