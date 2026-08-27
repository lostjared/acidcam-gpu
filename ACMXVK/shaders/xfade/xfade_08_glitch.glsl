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
    float bump = sin(fade_alpha * 3.14159);
    float bandY = floor(tc.y * 80.0);
    float jitter = (rand(vec2(bandY, 1.0)) * 2.0 - 1.0) * 0.04 * bump;
    vec2 uv = vec2(clamp(tc.x + jitter, 0.0, 1.0), tc.y);
    float ca = 0.012 * bump;
    vec4 curr = vec4(
        texture(samp, uv + vec2(ca, 0.0)).r,
        texture(samp, uv).g,
        texture(samp, uv - vec2(ca, 0.0)).b,
        1.0);
    vec4 prev = vec4(
        sample_previous(uv + vec2(ca, 0.0)).r,
        sample_previous(uv).g,
        sample_previous(uv - vec2(ca, 0.0)).b,
        1.0);
    color = mix(prev, curr, fade_alpha);
}
