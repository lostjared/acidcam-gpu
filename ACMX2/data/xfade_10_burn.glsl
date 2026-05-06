#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = fract(sin(dot(i, vec2(12.9898, 78.233))) * 43758.5453);
    float b = fract(sin(dot(i + vec2(1.0, 0.0), vec2(12.9898, 78.233))) * 43758.5453);
    float c = fract(sin(dot(i + vec2(0.0, 1.0), vec2(12.9898, 78.233))) * 43758.5453);
    float d = fract(sin(dot(i + vec2(1.0, 1.0), vec2(12.9898, 78.233))) * 43758.5453);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

void main() {
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    float n = noise(tc * 8.0);
    float edge = 0.12;
    float t = smoothstep(n - edge, n + edge, fade_alpha);
    vec3 ember = vec3(1.0, 0.45, 0.05);
    float glow = (1.0 - abs(2.0 * t - 1.0)) * (1.0 - smoothstep(0.0, 0.2, abs(n - fade_alpha)));
    vec4 mixed = mix(prev, curr, t);
    mixed.rgb += ember * glow * 0.6;
    color = mixed;
}
