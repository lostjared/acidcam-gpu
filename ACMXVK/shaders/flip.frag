#version 450

layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D samp;

void main() {
    color = texture(samp, vec2(tc.x, 1.0 - tc.y));
}
