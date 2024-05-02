struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, R, B]
@group(0) @binding(1) var<uniform> source: View;                            // [R, T, B]
@group(0) @binding(2) var<uniform> destination: View;                       // [R, T, B]

@group(0) @binding(3) var<storage, read> matrix: array<vec2<u32>>;          // (B, R, C)
#ifdef IN_FP16
@group(0) @binding(4) var<storage, read> input: array<vec2<u32>>;           // (B, T, C)
#else
@group(0) @binding(4) var<storage, read> input: array<vec4<f32>>;           // (B, T, C)
#endif
#ifdef OUT_FP16
@group(0) @binding(5) var<storage, read_write> output: array<vec2<u32>>;    // (B, T, R)
#else
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, R)
#endif

const NUM_SUBGROUPS: u32 = BLOCK_SIZE / MIN_SUBGROUP_SIZE;

var<workgroup> sketch: array<vec4<f32>, NUM_SUBGROUPS>;

fn compute_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x >> 2u;
    let offset = vec3<u32>(view.offset.zy, view.offset.x >> 2u);
    return dot(vec3<u32>(batch, token, index) + offset, vec3<u32>(view.stride.y * stride, stride, 1u));
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
}

fn squared_relu(x: vec4<f32>) -> vec4<f32> {
    let p = max(x, vec4<f32>(0.0));
    return p * p;
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn matmul(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32
) {
    let stride = shape.x / 4u;
    let index = invocation_id.x % BLOCK_SIZE;
    let channel = invocation_id.x / BLOCK_SIZE;     // 1 channel: 4 rows in matrix
    let token = invocation_id.y;
    let batch = invocation_id.z;

    // let bb = (batch * destination.shape[1] + token) * stride.x;
    let bb = compute_index(source, batch, token, 0u);
    let cb = batch * shape.y * stride + channel * 4u * stride;

    var local_sum = vec4<f32>(0.0);
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let bti = bb + i;
        var ci = cb + i;

        // read 4 elements from the input
#ifdef IN_FP16
        let x = unpack4x16float(input[bti]);
#else
        let x = input[bti];
#endif

        // read 4 rows from the matrix, each with 4 unpacked floats, forming a 4x4 sub-block
        var m: mat4x4<f32>;

        m[0] = unpack4x16float(matrix[ci]); ci += stride;
        m[1] = unpack4x16float(matrix[ci]); ci += stride;
        m[2] = unpack4x16float(matrix[ci]); ci += stride;
        m[3] = unpack4x16float(matrix[ci]);
        local_sum += transpose(m) * x;
    }
    // for (var step = subgroup_size >> 1u; step > 0u; step >>= 1u) {
    //     local_sum += subgroupShuffleDown(local_sum, step);
    // }
    local_sum = subgroupAdd(local_sum);

    if subgroup_invocation_id == 0u { sketch[subgroup_id] = local_sum; }
    workgroupBarrier();

#ifdef SUBGROUP_SIZE_32_32
    reduce_sum(index, 2u);
    reduce_sum(index, 1u);
#else
#ifdef SUBGROUP_SIZE_32_64
    if subgroup_size == 32u { reduce_sum(index, 2u); }
    reduce_sum(index, 1u);
#else
#ifdef SUBGROUP_SIZE_64_64
    reduce_sum(index, 1u);
#else
    for (var step = num_subgroups >> 1u; step > 0u; step >>= 1u) {
        if index < step {
            sketch[index] += sketch[index + step];
        }
        workgroupBarrier();
    }
#endif
#endif
#endif

    if index == 0u {
        let btc = compute_index(destination, batch, token, channel);
        var out = sketch[0];
#ifdef ACT_SQUARED_RELU
        out = squared_relu(out);
#endif
#ifdef ACT_TANH
        out = tanh(out);
#endif
#ifdef OUT_FP16
        output[btc] = pack4x16float(out);
#else
        output[btc] = out;
#endif
    }
}