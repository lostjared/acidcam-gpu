#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Chromatic aberration crossfade: RGB channels split outward then heal.
    float bump = sin(fade_alpha * 3.14159);
    float off = 0.03 * bump;
    vec2 dir = normalize(tc - 0.5 + 1e-5);
    vec4 curr = vec4(
        texture(samp, tc + dir * off).r,
        texture(samp, tc).g,
        texture(samp, tc - dir * off).b,
        1.0);
    vec4 prev = vec4(
        texture(prev_samp, tc + dir * off).r,
        texture(prev_samp, tc).g,
        texture(prev_samp, tc - dir * off).b,
        1.0);
    color = mix(prev, curr, fade_alpha);
}
