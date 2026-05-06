#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    float bump = sin(fade_alpha * 3.14159);
    float angle = bump * 6.2831853;
    vec2 c = tc - 0.5;
    float r = length(c);
    float a = atan(c.y, c.x) + angle * (1.0 - r);
    vec2 uv = vec2(cos(a), sin(a)) * r + 0.5;
    uv = clamp(uv, 0.0, 1.0);
    vec4 curr = texture(samp, uv);
    vec4 prev = texture(prev_samp, uv);
    color = mix(prev, curr, fade_alpha);
}
