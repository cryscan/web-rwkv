@group(0) @binding(0) var<uniform> dims: vec4<u32>;                         // [C, R]
@group(0) @binding(1) var<uniform> shape: vec4<u32>;                        // [R, T, B]

@group(0) @binding(2) var<storage, read> matrix: array<u32>;                // (R, C)
@group(0) @binding(3) var<storage, read> mx: array<vec2<u32>>;              // (C)
@group(0) @binding(4) var<storage, read> rx: array<vec2<u32>>;              // (C)
@group(0) @binding(5) var<storage, read> my: array<vec2<u32>>;              // (R)
@group(0) @binding(6) var<storage, read> ry: array<vec2<u32>>;              // (R)

@group(0) @binding(7) var<storage, read> input: array<vec4<f32>>;           // (B, T, C)
@group(0) @binding(8) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, R)

const BLOCK_SIZE: u32 = 128u;

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(128, 1, 1)
fn matmul(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = dims.xy / 4u;
    let index = invocation_id.x % BLOCK_SIZE;
    let channel = invocation_id.x / BLOCK_SIZE;   // 1 channel: 4 rows in matrix
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if token >= shape[1] || batch >= shape[2] {
        return;
    }

    let bb = (batch * shape[1] + token) * stride.x;
    let cb = channel * 4u * stride.x;

    let myc = unpack4x16float(my[channel]);
    let ryc = unpack4x16float(ry[channel]);

    var local_sum = vec4<f32>(0.0);
    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let bti = bb + i;
        var ci = cb + i;

        // read 4 elements from the input
        let x = input[bti];

        let mxi = unpack4x16float(mx[i]);
        let rxi = unpack4x16float(rx[i]);

        // read 4 rows from the matrix, each with 4 unpacked floats, forming a 4x4 sub-block
        var m: mat4x4<f32>;

        m[0] = fma(unpack4x8unorm(matrix[ci]), ryc[0] * rxi, myc[0] + mxi); ci += stride.x;
        m[1] = fma(unpack4x8unorm(matrix[ci]), ryc[1] * rxi, myc[1] + mxi); ci += stride.x;
        m[2] = fma(unpack4x8unorm(matrix[ci]), ryc[2] * rxi, myc[2] + mxi); ci += stride.x;
        m[3] = fma(unpack4x8unorm(matrix[ci]), ryc[3] * rxi, myc[3] + mxi);
        local_sum += transpose(m) * x;
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
        output[(batch * shape[1] + token) * stride.y + channel] = sketch[0];
    }
}