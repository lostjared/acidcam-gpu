#version 450

layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D background_image;
layout(set = 0, binding = 2) uniform sampler2DArray foreground_history;

void main() {
    vec4 background = texture(background_image, tc);
    vec4 foreground = texture(foreground_history, vec3(tc, 0.0));
    color = vec4(mix(background.rgb, foreground.rgb, foreground.a), 1.0);
}
