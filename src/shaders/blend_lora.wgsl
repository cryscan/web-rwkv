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
@group(0) @binding(3) var<uniform> factor: vec4<f32>;

@group(0) @binding(4) var<storage, read> xa: array<vec2<u32>>;              // (B, M, K)
@group(0) @binding(5) var<storage, read> xb: array<vec2<u32>>;              // (B, N, K)
@group(0) @binding(6) var<storage, read_write> output: array<vec2<u32>>;    // (B, N, M)

var<workgroup> sa: array<array<vec2<u32>, 32u>, 32u>;
var<workgroup> sb: array<array<vec2<u32>, 32u>, 32u>;

fn compute_index(view: View, z: u32, y: u32, x: u32) -> u32 {
    let stride = view.stride.x / 4u;
    let offset = view.offset.x / 4u;
    return ((view.offset.z + z) * view.stride.y + view.offset.y + y) * stride + offset + x;
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn blend(v: vec4<f32>, z: u32, y: u32, x: u32) {
    let u = unpack4x16float(output[compute_index(destination, z, y, x)]);
    output[compute_index(destination, z, y, x)] = pack4x16float(factor.x * v + factor.y * u);
}

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn blend_lora(in: Input) {
    let b = in.bid.xy * 32u;
    let u = in.uid.xy * 4u;
    let t = in.tid.xy * 4u;
    let ra = vec2<u32>(va.shape.x / 4u, va.shape.y);
    let rb = vec2<u32>(vb.shape.x / 4u, vb.shape.y);
    let stride = min(ra.x, rb.x);
    let i = in.index & 31u;

    var local_sum: mat4x4<f32>;
    for (var k = 0u; k < stride; k += 32u) {
        // load 8x4 rows from each of the matrix, each with 32x4 columns
        var x = k + i;
        for (var j = 0u; j < 32u; j += 1u) {
            if in.index < 32u {
                let y = b.x + j;
                if all(vec2<u32>(x, y) < ra) {
                    sa[j][i] = xa[compute_index(va, in.uid.z, y, x)];
                } else {
                    sa[j][i] = vec2<u32>(0u);
                }
            } else {
                let y = b.y + j;
                if all(vec2<u32>(x, y) < rb) {
                    sb[j][i] = xb[compute_index(vb, in.uid.z, y, x)];
                } else {
                    sb[j][i] = vec2<u32>(0u);
                }
            }
        }
        workgroupBarrier();

        // each thread multiplies and sums up 4x4 blocks along the reduced dimension
        if all(u < vec2<u32>(ra.y, rb.y)) {
            let reduce = min(32u, stride - k);
            for (x = 0u; x < reduce; x += 1u) {
                let aa = mat4x4<f32>(
                    unpack4x16float(sa[t.x][x]),
                    unpack4x16float(sa[t.x + 1u][x]),
                    unpack4x16float(sa[t.x + 2u][x]),
                    unpack4x16float(sa[t.x + 3u][x])
                );
                let bb = mat4x4<f32>(
                    unpack4x16float(sb[t.y][x]),
                    unpack4x16float(sb[t.y + 1u][x]),
                    unpack4x16float(sb[t.y + 2u][x]),
                    unpack4x16float(sb[t.y + 3u][x])
                );
                local_sum += transpose(aa) * bb;
            }
        }
        workgroupBarrier();
    }

    if all(u < vec2<u32>(ra.y, rb.y)) {
        blend(local_sum[0], in.uid.z, u.y, in.uid.x);
        blend(local_sum[1], in.uid.z, u.y + 1u, in.uid.x);
        blend(local_sum[2], in.uid.z, u.y + 2u, in.uid.x);
        blend(local_sum[3], in.uid.z, u.y + 3u, in.uid.x);
    }
}