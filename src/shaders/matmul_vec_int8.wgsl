struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
};

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, R, B]
@group(0) @binding(1) var<uniform> source: View;                            // [R, T, B]
@group(0) @binding(2) var<uniform> destination: View;                       // [R, T, B]

@group(0) @binding(3) var<storage, read> matrix: array<u32>;                // (B, R, C)
@group(0) @binding(4) var<storage, read> mx: array<vec4<f32>>;              // (B, C)
@group(0) @binding(5) var<storage, read> rx: array<vec4<f32>>;              // (B, C)
@group(0) @binding(6) var<storage, read> my: array<vec4<f32>>;              // (B, R)
@group(0) @binding(7) var<storage, read> ry: array<vec4<f32>>;              // (B, R)

#ifdef IN_FP16
@group(0) @binding(8) var<storage, read> input: array<vec2<u32>>;           // (B, T, C)
#else
@group(0) @binding(8) var<storage, read> input: array<vec4<f32>>;           // (B, T, C)
#endif
#ifdef OUT_FP16
@group(0) @binding(9) var<storage, read_write> output: array<vec2<u32>>;    // (B, T, R)
#else
@group(0) @binding(9) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, R)
#endif

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;

fn compute_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x / 4u;
    let offset = view.offset.x / 4u;
    return ((view.offset.z + batch) * view.stride.y + view.offset.y + token) * stride + offset + index;
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
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
    let stride = shape.x / 4u;
    let index = invocation_id.x % BLOCK_SIZE;
    let channel = invocation_id.x / BLOCK_SIZE;     // 1 channel: 4 rows in matrix
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = compute_index(source, batch, token, 0u);
    let cb = batch * shape.y * stride + channel * 4u * stride;

    // let myc = unpack4x16float(my[channel]);
    // let ryc = unpack4x16float(ry[channel]);
    let myc = my[batch * shape.y + channel];
    let ryc = ry[batch * shape.y + channel];

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

        // let mxi = unpack4x16float(mx[i]);
        // let rxi = unpack4x16float(rx[i]);
        let mxi = mx[batch * stride + i];
        let rxi = rx[batch * stride + i];

        // read 4 rows from the matrix, each with 4 unpacked floats, forming a 4x4 sub-block
        var m: mat4x4<f32>;

        m[0] = fma(unpack4x8unorm(matrix[ci]), ryc[0] * rxi, myc[0] + mxi); ci += stride;
        m[1] = fma(unpack4x8unorm(matrix[ci]), ryc[1] * rxi, myc[1] + mxi); ci += stride;
        m[2] = fma(unpack4x8unorm(matrix[ci]), ryc[2] * rxi, myc[2] + mxi); ci += stride;
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
        let btc = compute_index(destination, batch, token, channel);
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