#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    vec4 mid = mix(prev, curr, fade_alpha);
    float bump = sin(fade_alpha * 3.14159);
    color = vec4(mix(mid.rgb, vec3(1.0) - mid.rgb, bump), mid.a);
}
