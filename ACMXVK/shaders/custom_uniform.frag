#version 450

// Increment 5C custom-uniform reference shader. library.json values are packed
// in declaration order. The first value is custom_uniforms[0].x.

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

#define square_size ext.custom_uniforms[0].x

void main() {
    vec2 resolution = max(ext.u0.zw, vec2(1.0));
    float block_size = clamp(square_size, 1.0, max(resolution.x, resolution.y));
    vec2 pixel = floor(tc * resolution / block_size) * block_size;
    vec2 sample_uv = clamp((pixel + vec2(block_size * 0.5)) / resolution,
                           vec2(0.0), vec2(1.0));
    color = texture(samp, sample_uv);
}
