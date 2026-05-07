#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Diagonal wipe along x+y.
    float diag = (tc.x + tc.y) * 0.5;
    float edge = fade_alpha;
    float w = 0.05;
    float t = smoothstep(edge - w, edge + w, diag);
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    color = mix(curr, prev, t);
}
