#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    float bump = sin(fade_alpha * 3.14159);
    float bandY = floor(tc.y * 80.0);
    float jitter = (rand(vec2(bandY, 1.0)) * 2.0 - 1.0) * 0.04 * bump;
    vec2 uv = vec2(clamp(tc.x + jitter, 0.0, 1.0), tc.y);
    float ca = 0.012 * bump;
    vec4 curr = vec4(
        texture(samp, uv + vec2(ca, 0.0)).r,
        texture(samp, uv).g,
        texture(samp, uv - vec2(ca, 0.0)).b,
        1.0);
    vec4 prev = vec4(
        texture(prev_samp, uv + vec2(ca, 0.0)).r,
        texture(prev_samp, uv).g,
        texture(prev_samp, uv - vec2(ca, 0.0)).b,
        1.0);
    color = mix(prev, curr, fade_alpha);
}
