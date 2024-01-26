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
var<workgroup> deviation: f32;

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

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn layer_norm(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = (batch * shape[1] + token) * stride;

    for (var i = index; i < stride; i += BLOCK_SIZE) {
#ifdef FP16
        let value = unpack4x16float(x[bb + i]);
#else   
        let value = x[bb + i];
#endif
        let delta = value - mu[index];
        let _count = count[index] + 1u;
        let _mu = mu[index] + delta / vec4<f32>(_count);

        count[index] = _count;
        mu[index] = _mu;
        m2[index] += delta * (value - _mu);
    }
    workgroupBarrier();

    reduce_step(index, 64u);
    reduce_step(index, 32u);
    reduce_step(index, 16u);
    reduce_step(index, 8u);
    reduce_step(index, 4u);
    reduce_step(index, 2u);
    reduce_step(index, 1u);

    if index == 0u {
        let _count = vec4<f32>(count[0]);
        mean = dot(mu[0], _count / f32(shape[0]));

        let _delta = mu[0] - mean;
        let _m2 = dot(m2[0], vec4<f32>(1.0)) + dot(_delta * _delta, _count);
        deviation = inverseSqrt(_m2 / f32(shape[0]) + EPS);
    }
    workgroupBarrier();

    for (var i = index; i < stride; i += BLOCK_SIZE) {
#ifdef FP16
        let value = (unpack4x16float(x[bb + i]) - mean) * deviation;
        x[bb + i] = pack4x16float(fma(value, unpack4x16float(w[i]), unpack4x16float(b[i])));
#else
        let value = (x[bb + i] - mean) * deviation;
        x[bb + i] = fma(value, unpack4x16float(w[i]), unpack4x16float(b[i]));
#endif
    }

#ifdef STATS
    if index == 0u {
        s[batch * shape[1] + token] = vec4<f32>(mean, deviation, 0.0, 0.0);
    }
#endif
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn group_norm(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let head = invocation_id.y;
    let token = invocation_id.z;

    let h = head * stride;
    let th = (token * shape[1] + head) * stride;

    for (var i = index; i < stride; i += BLOCK_SIZE) {
#ifdef FP16
        let value = unpack4x16float(x[th + i]);
#else
        let value = x[th + i];
#endif
        let delta = value - mu[index];
        let _count = count[index] + 1u;
        let _mu = mu[index] + delta / vec4<f32>(_count);

        count[index] = _count;
        mu[index] = _mu;
        m2[index] += delta * (value - _mu);
    }
    workgroupBarrier();

    reduce_step(index, 16u);
    reduce_step(index, 8u);
    reduce_step(index, 4u);
    reduce_step(index, 2u);
    reduce_step(index, 1u);

    if index == 0u {
        let _count = vec4<f32>(count[0]);
        mean = dot(mu[0], _count / f32(shape[0]));

        let _delta = mu[0] - mean;
        let _m2 = dot(m2[0], vec4<f32>(1.0)) + dot(_delta * _delta, _count);
        deviation = inverseSqrt(_m2 / f32(shape[0]) + EPS);
    }
    workgroupBarrier();

    for (var i = index; i < stride; i += BLOCK_SIZE) {
#ifdef FP16
        let value = (unpack4x16float(x[th + i]) - mean) * deviation;
        x[th + i] = pack4x16float(fma(value, unpack4x16float(w[h + i]), unpack4x16float(b[h + i])));
#else
        let value = (x[th + i] - mean) * deviation;
        x[th + i] = fma(value, unpack4x16float(w[h + i]), unpack4x16float(b[h + i]));
#endif
    }
}