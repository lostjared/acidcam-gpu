#version 450

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D input_image;

const float PQ_M1 = 2610.0 / 16384.0;
const float PQ_M2 = 2523.0 / 32.0;
const float PQ_C1 = 3424.0 / 4096.0;
const float PQ_C2 = 2413.0 / 128.0;
const float PQ_C3 = 2392.0 / 128.0;
const float HLG_A = 0.17883277;
const float HLG_B = 0.28466892;
const float HLG_C = 0.55991073;

vec3 pq_decode(vec3 encoded) {
    vec3 power_value = pow(clamp(encoded, 0.0, 1.0), vec3(1.0 / PQ_M2));
    vec3 numerator = max(power_value - vec3(PQ_C1), vec3(0.0));
    vec3 denominator = max(vec3(PQ_C2) - vec3(PQ_C3) * power_value,
                           vec3(1.0e-6));
    return pow(numerator / denominator, vec3(1.0 / PQ_M1));
}

vec3 pq_encode(vec3 linear_value) {
    vec3 power_value = pow(max(linear_value, vec3(0.0)), vec3(PQ_M1));
    vec3 encoded = pow((vec3(PQ_C1) + vec3(PQ_C2) * power_value) /
                           (vec3(1.0) + vec3(PQ_C3) * power_value),
                       vec3(PQ_M2));
    return clamp(encoded, 0.0, 1.0);
}

vec3 hlg_decode(vec3 encoded) {
    bvec3 low = lessThanEqual(encoded, vec3(0.5));
    vec3 low_value = encoded * encoded / 3.0;
    vec3 high_value =
        (exp((encoded - vec3(HLG_C)) / HLG_A) + vec3(HLG_B)) / 12.0;
    return mix(high_value, low_value, low);
}

vec3 hlg_encode(vec3 linear_value) {
    linear_value = max(linear_value, vec3(0.0));
    bvec3 low = lessThanEqual(linear_value, vec3(1.0 / 12.0));
    vec3 low_value = sqrt(3.0 * linear_value);
    vec3 high_value =
        HLG_A * log(max(12.0 * linear_value - vec3(HLG_B), vec3(1.0e-6))) +
        vec3(HLG_C);
    return clamp(mix(high_value, low_value, low), 0.0, 1.0);
}

void main() {
    vec4 source = texture(input_image, frag_tex_coord);
#if defined(ACMXVK_HDR_PQ_DECODE)
    color = vec4(pq_decode(source.rgb), source.a);
#elif defined(ACMXVK_HDR_PQ_ENCODE)
    color = vec4(pq_encode(source.rgb), source.a);
#elif defined(ACMXVK_HDR_HLG_DECODE)
    color = vec4(hlg_decode(source.rgb), source.a);
#elif defined(ACMXVK_HDR_HLG_ENCODE)
    color = vec4(hlg_encode(source.rgb), source.a);
#else
#error "An ACMXVK HDR transfer operation must be selected"
#endif
}
