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
    // curr slides in from the right, prev slides out to the left.
    vec2 uvP = vec2(tc.x + fade_alpha, tc.y);
    vec2 uvC = vec2(tc.x - (1.0 - fade_alpha), tc.y);
    if (tc.x < fade_alpha) {
        color = texture(samp, uvC);
    } else {
        color = sample_previous(uvP);
    }
}
