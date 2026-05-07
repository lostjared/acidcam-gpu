#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(73.156, 13.737))) * 27341.91);
}

void main() {
    // Mosaic explode: cells offset outward from centre then settle.
    float bump = sin(fade_alpha * 3.14159);
    float cells = 80.0;
    vec2 cell = floor(tc * cells);
    vec2 cellCentre = (cell + 0.5) / cells;
    vec2 dir = normalize(cellCentre - 0.5 + 1e-5);
    float jitter = rand(cell) * 0.06 * bump;
    vec2 uv = clamp(tc + dir * jitter, 0.0, 1.0);
    vec4 curr = texture(samp, uv);
    vec4 prev = texture(prev_samp, uv);
    color = mix(prev, curr, fade_alpha);
}
