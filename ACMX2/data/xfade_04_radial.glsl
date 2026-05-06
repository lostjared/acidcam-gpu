#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    vec2 d = tc - vec2(0.5);
    float r = length(d) / 0.7071;
    float edge = 0.1;
    float threshold = fade_alpha * (1.0 + edge) - edge;
    float t = 1.0 - smoothstep(threshold, threshold + edge, r);
    color = mix(prev, curr, t);
}
