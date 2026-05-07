#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(91.345, 47.853))) * 38291.337);
}

void main() {
    // Per-pixel random dissolve (fine-grained noise reveal).
    float n = rand(floor(tc * 1024.0));
    float w = 0.05;
    float t = smoothstep(n - w, n + w, fade_alpha);
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    color = mix(prev, curr, t);
}
