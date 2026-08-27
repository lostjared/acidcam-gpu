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
    // Chromatic aberration crossfade: RGB channels split outward then heal.
    float bump = sin(fade_alpha * 3.14159);
    float off = 0.03 * bump;
    vec2 dir = normalize(tc - 0.5 + 1e-5);
    vec4 curr = vec4(
        texture(samp, tc + dir * off).r,
        texture(samp, tc).g,
        texture(samp, tc - dir * off).b,
        1.0);
    vec4 prev = vec4(
        sample_previous(tc + dir * off).r,
        sample_previous(tc).g,
        sample_previous(tc - dir * off).b,
        1.0);
    color = mix(prev, curr, fade_alpha);
}
