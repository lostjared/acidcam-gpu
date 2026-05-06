#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    vec2 grid = vec2(20.0, 15.0);
    vec2 g = fract(tc * grid) - 0.5;
    float d = (abs(g.x) + abs(g.y)) * 2.0;
    float edge = 0.08;
    float t = smoothstep(fade_alpha - edge, fade_alpha + edge, 1.0 - d);
    color = mix(prev, curr, t);
}
