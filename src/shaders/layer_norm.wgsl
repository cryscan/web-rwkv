@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

@group(0) @binding(1) var<storage, read> w: array<vec2<u32>>;               // (C)
@group(0) @binding(2) var<storage, read> b: array<vec2<u32>>;               // (C)
#ifdef FP16
@group(0) @binding(3) var<storage, read_write> x: array<vec2<u32>>;         // (B, T, C)
#else
@group(0) @binding(3) var<storage, read_write> x: array<vec4<f32>>;         // (B, T, C)
#endif
#ifdef STATS
@group(0) @binding(4) var<storage, read_write> s: array<vec4<f32>>;         // (B, T, 4)
#endif

var<workgroup> mu: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> m2: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> count: array<vec4<u32>, BLOCK_SIZE>;

var<workgroup> mean: f32;
var<workgroup> dev: f32;

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn reduce_step(index: u32, stride: u32) {
    if index < stride {
        let mu_1 = mu[index];
        let mu_2 = mu[index + stride];
        let count_1 = count[index];
        let count_2 = count[index + stride];

        let delta = mu_2 - mu_1;
        let total = count_1 + count_2;
        count[index] = total;

        mu[index] = select(vec4<f32>(0.0), (mu_1 * vec4<f32>(count_1) + mu_2 * vec4<f32>(count_2)) / vec4<f32>(total), total > vec4<u32>(0u));
        m2[index] = select(vec4<f32>(0.0), m2[index] + m2[index + stride] + delta * delta * vec4<f32>(count_1 * count_2) / vec4<f32>(total), total > vec4<u32>(0u));
    }
    workgroupBarrier();
}

fn load_x(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(x[index]);
#else
    return x[index];
#endif
}

fn store_x(index: u32, value: vec4<f32>) {
#ifdef FP16
    x[index] = pack4x16float(value);
#else
    x[index] = value;
#endif
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn layer_norm(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = (batch * shape[1] + token) * stride;
#ifdef GROUP_NORM
    let h = token * stride;
#endif

    var _mu: vec4<f32>;
    var _m2: vec4<f32>;
    var _count: vec4<u32>;
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = load_x(bb + i);
        let delta = value - _mu;
        _count += 1u;
        _mu += delta / vec4<f32>(_count);
        _m2 += delta * (value - _mu);
    }
    count[index] = _count;
    mu[index] = _mu;
    m2[index] = _m2;
    workgroupBarrier();

    reduce_step(index, 64u);
    reduce_step(index, 32u);
    reduce_step(index, 16u);
    reduce_step(index, 8u);
    reduce_step(index, 4u);
    reduce_step(index, 2u);
    reduce_step(index, 1u);

    if index == 0u {
        let _mu = mu[0];
        let _count = vec4<f32>(count[0]);
        mean = dot(_mu, _count / f32(shape[0]));

        let delta = _mu - mean;
        let _m2 = dot(m2[0], vec4<f32>(1.0)) + dot(delta * delta, _count);
        let _var = _m2 / f32(shape[0]) + EPS;
        dev = inverseSqrt(_var);

#ifdef STATS
        s[batch * shape[1] + token] = vec4<f32>(mean, dev, _var, 0.0);
#endif
    }
    workgroupBarrier();

    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = (load_x(bb + i) - mean) * dev;
#ifdef GROUP_NORM
        store_x(bb + i, fma(value, unpack4x16float(w[h + i]), unpack4x16float(b[h + i])));
#else
        store_x(bb + i, fma(value, unpack4x16float(w[i]), unpack4x16float(b[i])));
#endif
    }
}
