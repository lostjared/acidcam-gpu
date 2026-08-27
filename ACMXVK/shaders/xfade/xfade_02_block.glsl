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
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 gridSize = vec2(40.0, 30.0);
    vec2 blockId = floor(tc * gridSize);
    vec2 blockOffset = vec2(
        rand(blockId) * 2.0 - 1.0,
        rand(blockId + vec2(1.33, 2.71)) * 2.0 - 1.0
    );
    float intensity = 4.0 * fade_alpha * (1.0 - fade_alpha);
    float maxTearDistance = 0.15;
    vec2 uvDisplaced = clamp(tc + (blockOffset * intensity * maxTearDistance), 0.0, 1.0);
    vec4 curr = texture(samp, uvDisplaced);
    vec4 prev = sample_previous(uvDisplaced);
    float threshold = fade_alpha + (blockOffset.x * 0.2);
    color = (threshold < 0.5) ? prev : curr;
}
