#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    float dim = 1.0 - 2.0 * abs(fade_alpha - 0.5);
    vec4 mid = mix(prev, curr, fade_alpha);
    color = vec4(mid.rgb * (1.0 - dim), mid.a);
}
