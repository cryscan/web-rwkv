struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
};

@group(0) @binding(0) var<uniform> va: View;                                // [K, M, B]
@group(0) @binding(1) var<uniform> vb: View;                                // [K, N, B]
@group(0) @binding(2) var<uniform> destination: View;                       // [M, N, B]

@group(0) @binding(3) var<storage, read> xa: array<vec2<u32>>;              // (B, M, K)
@group(0) @binding(4) var<storage, read> xb: array<vec2<u32>>;              // (B, N, K)
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;    // (B, N, M)

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

@compute @workgroup_size(8, 8, 1)
fn matmul(
    @builtin(workgroup_id) bid: vec3<u32>,
    @builtin(global_invocation_id) uid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(local_invocation_index) index: u32
) {
    let b = bid.xy * 32u;
    let u = uid.xy * 4u;
    let t = tid.xy * 4u;
    let ra = vec2<u32>(va.shape.x / 4u, va.shape.y);
    let rb = vec2<u32>(vb.shape.x / 4u, vb.shape.y);
    let stride = min(ra.x, rb.x);

    var local_sum: mat4x4<f32>;
    for (var k = 0u; k < stride; k += 32u) {
        // load 8x4 rows from each of the matrix, each with 32x4 columns
        // use the first 32 threads to load from the 1st matrix, and use the last 32 threads to load from the 2nd
        let i = index & 0x1fu;
        for (var y = 0u; y < 32u; y += 1u) {
            if index < 32u {
                let ua = vec2<u32>(k + i, b.x + y);
                if all(ua < ra) {
                    sa[y][i] = xa[compute_index(va, uid.z, ua.y, ua.x)];
                } else {
                    sa[y][i] = vec2<u32>(0u);
                }
            } else {
                let ub = vec2<u32>(k + i, b.y + y);
                if all(ub < rb) {
                    sb[y][i] = xb[compute_index(vb, uid.z, ub.y, ub.x)];
                } else {
                    sb[y][i] = vec2<u32>(0u);
                }
            }
        }
        workgroupBarrier();

        // each thread multiplies and sums up 4x4 blocks along the reduced dimension
        if u.x < va.shape.y && u.y < vb.shape.y {
            for (var x = 0u; x < 32u; x += 1u) {
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

    if uid.x < stride && u.y < destination.shape.y {
        output[compute_index(destination, uid.z, u.y, uid.x)] = local_sum[0];
        output[compute_index(destination, uid.z, u.y + 1u, uid.x)] = local_sum[1];
        output[compute_index(destination, uid.z, u.y + 2u, uid.x)] = local_sum[2];
        output[compute_index(destination, uid.z, u.y + 3u, uid.x)] = local_sum[3];
    }
}