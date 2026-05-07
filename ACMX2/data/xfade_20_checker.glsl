#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Checkerboard reveal.
    vec2 cell = floor(tc * 16.0);
    float parity = mod(cell.x + cell.y, 2.0);
    float threshold = mix(fade_alpha * 0.5, 0.5 + fade_alpha * 0.5, parity);
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    color = (fade_alpha > threshold) ? curr : prev;
}
