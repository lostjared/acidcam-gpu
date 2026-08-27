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
    // Iris open: curr expands as a circle from the centre.
    vec2 c = tc - 0.5;
    float r = length(c);
    float maxR = length(vec2(0.5, 0.5));
    float radius = fade_alpha * maxR;
    float w = 0.02;
    float t = smoothstep(radius - w, radius + w, r);
    vec4 curr = texture(samp, tc);
    vec4 prev = sample_previous(tc);
    color = mix(curr, prev, t);
}
