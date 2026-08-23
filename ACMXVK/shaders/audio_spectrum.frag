#version 450

// Increment 5F current-frame FFT spectrum reference shader.
// MXVK exposes 256 R32_SFLOAT bins as a 1-D sampler at binding 3.

layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D samp;

layout(set = 0, binding = 1) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
} ext;

layout(set = 0, binding = 3) uniform sampler1D spectrum;

void main() {
    vec3 source = texture(samp, tc).rgb;
    float frequency = tc.x * tc.x;
    float energy = texture(spectrum, frequency).r;
    float bar_height = clamp(sqrt(max(energy, 0.0)) * 3.0, 0.0, 1.0);
    float bar = step(1.0 - bar_height, tc.y);
    vec3 spectrum_color = mix(vec3(0.1, 0.5, 1.0), vec3(1.0, 0.2, 0.1), tc.x);
    color = vec4(mix(source * 0.55, spectrum_color, bar * 0.9), 1.0);
}
