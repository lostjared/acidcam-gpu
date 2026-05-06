#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    float bump = sin(fade_alpha * 3.14159);
    float blocks = mix(256.0, 8.0, bump);
    vec2 cell = floor(tc * blocks) / blocks + (0.5 / blocks);
    vec4 curr = texture(samp, cell);
    vec4 prev = texture(prev_samp, cell);
    color = mix(prev, curr, fade_alpha);
}
