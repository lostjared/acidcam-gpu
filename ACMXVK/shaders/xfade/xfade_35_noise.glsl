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
    return fract(sin(dot(co.xy, vec2(91.345, 47.853))) * 38291.337);
}

void main() {
    // Per-pixel random dissolve (fine-grained noise reveal).
    float n = rand(floor(tc * 1024.0));
    float w = 0.05;
    float t = smoothstep(n - w, n + w, fade_alpha);
    vec4 curr = texture(samp, tc);
    vec4 prev = sample_previous(tc);
    color = mix(prev, curr, t);
}
