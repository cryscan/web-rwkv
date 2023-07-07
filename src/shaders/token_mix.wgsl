@group(0) @binding(0) var<uniform> num_emb: u32;

@group(1) @binding(0) var<uniform> num_tokens: u32;

@group(1) @binding(1) var<storage, read> time_decay: array<vec4<f32>>;      // (C)
@group(1) @binding(2) var<storage, read> time_first: array<vec4<f32>>;      // (C)

@group(1) @binding(3) var<storage, read> x: array<vec4<f32>>;               // (T, C)
@group(1) @binding(4) var<storage, read> k: array<vec4<f32>>;               // (T, C)
@group(1) @binding(5) var<storage, read> v: array<vec4<f32>>;               // (T, C)
@group(1) @binding(6) var<storage, read> r: array<vec4<f32>>;               // (T, C)

@group(1) @binding(7) var<storage, read_write> a: array<vec4<f32>>;         // (C)
@group(1) @binding(8) var<storage, read_write> b: array<vec4<f32>>;         // (C)
@group(1) @binding(9) var<storage, read_write> p: array<vec4<f32>>;         // (C)

@group(1) @binding(10) var<storage, read_write> sx: array<vec4<f32>>;       // (C)
@group(1) @binding(11) var<storage, read_write> output: array<vec4<f32>>;   // (T, C)

const BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(256, 1, 1)
fn token_mix(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = invocation_id.x;
    let stride = num_emb / 4u;

    for (var t = 0u; t < num_tokens; t += 1u) {
        let ti = t * stride + index;

        let kk = k[ti];
        let vv = v[ti];

        let aa = a[index];
        let bb = b[index];
        let pp = p[index];
        var ww = time_first[index] + kk;
        var q = max(pp, ww);
        var e1 = exp(pp - q);
        var e2 = exp(ww - q);

        let rr = 1.0 / (1.0 + exp(-r[ti]));
        output[ti] = rr * (e1 * aa + e2 * vv) / (e1 * bb + e2);

        ww = time_decay[index] + pp;
        q = max(ww, kk);
        e1 = exp(ww - q);
        e2 = exp(kk - q);
        a[index] = e1 * aa + e2 * vv;
        b[index] = e1 * bb + e2;
        p[index] = q;
    }

    sx[index] = x[(num_tokens - 1u) * stride + index];
}