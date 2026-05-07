#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Horizontal sine wave distortion.
    float bump = sin(fade_alpha * 3.14159);
    float dx = sin(tc.y * 30.0 + fade_alpha * 6.2831) * 0.04 * bump;
    vec2 uv = vec2(clamp(tc.x + dx, 0.0, 1.0), tc.y);
    vec4 curr = texture(samp, uv);
    vec4 prev = texture(prev_samp, uv);
    color = mix(prev, curr, fade_alpha);
}
