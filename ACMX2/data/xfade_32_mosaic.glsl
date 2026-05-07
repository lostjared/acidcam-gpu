#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Mosaic grow: pixelation peaks mid-transition.
    float bump = sin(fade_alpha * 3.14159);
    float cells = mix(400.0, 16.0, bump);
    vec2 uv = (floor(tc * cells) + 0.5) / cells;
    vec4 curr = texture(samp, uv);
    vec4 prev = texture(prev_samp, uv);
    color = mix(prev, curr, fade_alpha);
}
