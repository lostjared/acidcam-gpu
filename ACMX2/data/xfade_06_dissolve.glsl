#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    float n = hash(floor(tc * vec2(640.0, 480.0)));
    float edge = 0.05;
    float t = smoothstep(n - edge, n + edge, fade_alpha);
    color = mix(prev, curr, t);
}
