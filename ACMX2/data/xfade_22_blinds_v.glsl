#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Vertical blinds.
    float band = floor(tc.x * 12.0);
    float offset = fract(band * 0.211);
    float t = step(offset, fade_alpha);
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    color = mix(prev, curr, t);
}
