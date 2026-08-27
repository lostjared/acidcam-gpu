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
    // Checkerboard reveal.
    vec2 cell = floor(tc * 16.0);
    float parity = mod(cell.x + cell.y, 2.0);
    float threshold = mix(fade_alpha * 0.5, 0.5 + fade_alpha * 0.5, parity);
    vec4 curr = texture(samp, tc);
    vec4 prev = sample_previous(tc);
    color = (fade_alpha > threshold) ? curr : prev;
}
