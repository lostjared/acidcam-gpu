#version 450

// Increment 6B MIDI reference shader. ACMX2 MIDI Slider 1 maps to the first
// custom uniform declared in library.json.

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
} ext;

#define slider1 ext.custom_uniforms[0].x

void main() {
    vec4 source = texture(samp, tc);
    float brightness = mix(0.2, 2.0, clamp(slider1, 0.0, 1.0));
    color = vec4(source.rgb * brightness, source.a);
}
