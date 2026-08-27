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
    // Vertical shutter close on prev, then opens to reveal curr.
    float dist = abs(tc.y - 0.5) * 2.0;
    // Gap shrinks to 0 at fade_alpha=0.5 then reopens revealing curr.
    float threshold = abs(fade_alpha - 0.5) * 2.0;
    vec4 curr = texture(samp, tc);
    vec4 prev = sample_previous(tc);
    if (dist > threshold) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        color = (fade_alpha < 0.5) ? prev : curr;
    }
}
