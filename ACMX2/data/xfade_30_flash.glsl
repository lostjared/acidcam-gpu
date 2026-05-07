#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(54.7811, 21.4451))) * 19847.7531);
}

void main() {
    // Quick white flash mid-transition.
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    float flash = pow(max(0.0, 1.0 - 4.0 * abs(fade_alpha - 0.5)), 1.5);
    float n = rand(floor(tc * 600.0));
    vec3 base = (fade_alpha < 0.5) ? prev.rgb : curr.rgb;
    color = vec4(mix(base, vec3(1.0) * (0.7 + 0.3 * n), flash), 1.0);
}
