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
    vec3 p = pow(clamp(encoded, 0.0, 1.0), vec3(1.0 / PQ_M2));
    return pow(max(p - PQ_C1, 0.0) / max(PQ_C2 - PQ_C3 * p, 1.0e-6),
               vec3(1.0 / PQ_M1));
}

vec3 hlg_decode(vec3 encoded) {
    bvec3 low = lessThanEqual(encoded, vec3(0.5));
    vec3 low_value = encoded * encoded / 3.0;
    vec3 high_value = (exp((encoded - HLG_C) / HLG_A) + HLG_B) / 12.0;
    return mix(high_value, low_value, low);
}

vec3 linear_bt2020_to_bt709(vec3 value) {
    return mat3(1.660491, -0.124550, -0.018151,
                -0.587641, 1.132900, -0.100579,
                -0.072850, -0.008349, 1.118730) * value;
}

vec3 aces_tone_map(vec3 value) {
    return clamp((value * (2.51 * value + 0.03)) /
                     (value * (2.43 * value + 0.59) + 0.14),
                 0.0, 1.0);
}

vec3 srgb_encode(vec3 linear_value) {
    linear_value = clamp(linear_value, 0.0, 1.0);
    bvec3 low = lessThanEqual(linear_value, vec3(0.0031308));
    vec3 low_value = 12.92 * linear_value;
    vec3 high_value = 1.055 * pow(linear_value, vec3(1.0 / 2.4)) - 0.055;
    return mix(high_value, low_value, low);
}

void main() {
    vec4 source = texture(input_image, frag_tex_coord);
#if defined(ACMXVK_HDR_PREVIEW_PQ)
    vec3 scene_linear = pq_decode(source.rgb) * (10000.0 / 203.0);
#elif defined(ACMXVK_HDR_PREVIEW_HLG)
    vec3 scene_linear = hlg_decode(source.rgb) * (1000.0 / 203.0);
#else
#error "An ACMXVK HDR preview transfer must be selected"
#endif
    vec3 display_linear = linear_bt2020_to_bt709(scene_linear);
    color = vec4(srgb_encode(aces_tone_map(max(display_linear, 0.0))),
                 source.a);
}
