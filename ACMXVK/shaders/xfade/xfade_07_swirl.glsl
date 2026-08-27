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
    float bump = sin(fade_alpha * 3.14159);
    float angle = bump * 6.2831853;
    vec2 c = tc - 0.5;
    float r = length(c);
    float a = atan(c.y, c.x) + angle * (1.0 - r);
    vec2 uv = vec2(cos(a), sin(a)) * r + 0.5;
    uv = clamp(uv, 0.0, 1.0);
    vec4 curr = texture(samp, uv);
    vec4 prev = sample_previous(uv);
    color = mix(prev, curr, fade_alpha);
}
