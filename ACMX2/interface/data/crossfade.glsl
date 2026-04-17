#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    color = mix(prev, curr, fade_alpha);
}
