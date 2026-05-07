#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Iris open: curr expands as a circle from the centre.
    vec2 c = tc - 0.5;
    float r = length(c);
    float maxR = length(vec2(0.5, 0.5));
    float radius = fade_alpha * maxR;
    float w = 0.02;
    float t = smoothstep(radius - w, radius + w, r);
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    color = mix(curr, prev, t);
}
