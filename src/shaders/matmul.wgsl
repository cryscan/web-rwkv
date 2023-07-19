@group(0) @binding(0) var<uniform> dims: vec2<u32>;                         // [C, R]
@group(0) @binding(1) var<storage, read> matrix: array<vec2<u32>>;          // (R, C)
@group(0) @binding(2) var<storage, read> input: array<vec4<f32>>;           // (T, C)
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;    // (T, R)

const BLOCK_SIZE: u32 = 256u;

var<workgroup> local_sum: array<vec4<f32>, BLOCK_SIZE>;

fn reduce_step_barrier(index: u32, stride: u32) {
    if index < stride {
        local_sum[index] += local_sum[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(256, 1, 1)
fn matmul(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let channel = invocation_id.y;      // 1 channel: 4 rows in matrix
    let token = invocation_id.z;
    let stride = dims / 4u;

    local_sum[index] = vec4<f32>(0.0);
    for (var i = index; i < stride.x; i += BLOCK_SIZE) {
        let ti = token * stride.x + i;
        var ci = channel * 4u * stride.x + i;

        // read 4 elements from the input
        let x = input[ti];

        // read 4 rows from the matrix, each with 4 unpacked floats, forming a 4x4 sub-block
        var data: vec2<u32>;
        var m: mat4x4<f32>;

        data = matrix[ci]; m[0] = vec4<f32>(unpack2x16float(data.x), unpack2x16float(data.y)); ci += stride.x;
        data = matrix[ci]; m[1] = vec4<f32>(unpack2x16float(data.x), unpack2x16float(data.y)); ci += stride.x;
        data = matrix[ci]; m[2] = vec4<f32>(unpack2x16float(data.x), unpack2x16float(data.y)); ci += stride.x;
        data = matrix[ci]; m[3] = vec4<f32>(unpack2x16float(data.x), unpack2x16float(data.y));
        local_sum[index] += transpose(m) * x;
    }
    workgroupBarrier();

    reduce_step_barrier(index, 128u);
    reduce_step_barrier(index, 64u);
    reduce_step_barrier(index, 32u);

    if index < 32u {
        local_sum[index] += local_sum[index + 16u];
        local_sum[index] += local_sum[index + 8u];
        local_sum[index] += local_sum[index + 4u];
        local_sum[index] += local_sum[index + 2u];
        local_sum[index] += local_sum[index + 1u];
    }

    if index == 0u {
        output[token * stride.y + channel] = local_sum[0];
    }
}