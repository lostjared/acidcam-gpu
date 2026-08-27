#version 450
layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D samp;
layout(set = 0, binding = 2) uniform sampler2DArray prev_samp;
layout(push_constant) uniform SpritePushConstants {
    float screenWidth;
    float screenHeight;
    float spritePosX;
    float spritePosY;
    float spriteSizeW;
    float spriteSizeH;
    float effectsOn;
    float rotationDegrees;
    vec4 params;
} pc;

#define fade_alpha pc.params.x

vec4 sample_previous(vec2 uv) {
    return texture(prev_samp, vec3(uv, 0.0));
}

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
    vec4 prev = sample_previous(uv);
    color = mix(prev, curr, fade_alpha);
}
