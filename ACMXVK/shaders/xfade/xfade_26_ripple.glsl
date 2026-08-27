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
    // Concentric ripple distortion.
    float bump = sin(fade_alpha * 3.14159);
    vec2 c = tc - 0.5;
    float r = length(c);
    float wave = sin(r * 40.0 - fade_alpha * 12.0) * 0.03 * bump;
    vec2 uv = tc + normalize(c + 1e-5) * wave;
    uv = clamp(uv, 0.0, 1.0);
    vec4 curr = texture(samp, uv);
    vec4 prev = sample_previous(uv);
    color = mix(prev, curr, fade_alpha);
}
