#version 450

// Increment 5G audio-band reference shader. ACMXVK maps live RtAudio data
// into SpriteExtended while preserving the existing ABI:
//   u1.y = amp, u1.z = iamp, u2.z = iSampleRate,
//   u2.w = amp_peak, u3.z = amp_rms, u3.w = amp_smooth,
//   audio_bands.xyz = amp_low, amp_mid, amp_high.

layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D samp;

layout(set = 0, binding = 1) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
    vec4 custom_uniforms[16];
    vec4 audio_bands;
} ext;

#define amp ext.u1.y
#define iamp ext.u1.z
#define iSampleRate ext.u2.z
#define amp_peak ext.u2.w
#define amp_rms ext.u3.z
#define amp_smooth ext.u3.w
#define amp_low ext.audio_bands.x
#define amp_mid ext.audio_bands.y
#define amp_high ext.audio_bands.z

void main() {
    vec2 center = tc - vec2(0.5);
    float pulse = clamp(amp_smooth * 3.0 + amp_peak * 0.75 + amp_low * 0.3,
                        0.0, 0.4);
    vec2 sample_uv = clamp(center * (1.0 - pulse) + vec2(0.5), vec2(0.0),
                           vec2(1.0));
    vec3 source = texture(samp, sample_uv).rgb;

    float frequency_color = clamp(iamp / max(iSampleRate * 0.5, 1.0), 0.0, 1.0);
    vec3 band_tint = clamp(vec3(amp_low, amp_mid, amp_high), vec3(0.0), vec3(1.0));
    vec3 tint = vec3(1.0 + amp * 2.0, 1.0 + amp_rms,
                     1.0 + frequency_color * 0.75) + band_tint * 0.65;
    color = vec4(clamp(source * tint, vec3(0.0), vec3(1.0)), 1.0);
}
