#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Concentric ripple distortion.
    float bump = sin(fade_alpha * 3.14159);
    vec2 c = tc - 0.5;
    float r = length(c);
    float wave = sin(r * 40.0 - fade_alpha * 12.0) * 0.03 * bump;
    vec2 uv = tc + normalize(c + 1e-5) * wave;
    uv = clamp(uv, 0.0, 1.0);
    vec4 curr = texture(samp, uv);
    vec4 prev = texture(prev_samp, uv);
    color = mix(prev, curr, fade_alpha);
}
