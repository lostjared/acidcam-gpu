#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    float edge = 0.08;
    float threshold = fade_alpha * (1.0 + edge) - edge;
    float t = smoothstep(threshold, threshold + edge, tc.x);
    color = mix(prev, curr, t);
}
