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
void main() {
    // Mosaic grow: pixelation peaks mid-transition.
    float bump = sin(fade_alpha * 3.14159);
    float cells = mix(400.0, 16.0, bump);
    vec2 uv = (floor(tc * cells) + 0.5) / cells;
    vec4 curr = texture(samp, uv);
    vec4 prev = sample_previous(uv);
    color = mix(prev, curr, fade_alpha);
}
