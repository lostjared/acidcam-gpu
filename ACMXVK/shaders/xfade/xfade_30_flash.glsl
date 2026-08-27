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
    return fract(sin(dot(co.xy, vec2(54.7811, 21.4451))) * 19847.7531);
}

void main() {
    // Quick white flash mid-transition.
    vec4 curr = texture(samp, tc);
    vec4 prev = sample_previous(tc);
    float flash = pow(max(0.0, 1.0 - 4.0 * abs(fade_alpha - 0.5)), 1.5);
    float n = rand(floor(tc * 600.0));
    vec3 base = (fade_alpha < 0.5) ? prev.rgb : curr.rgb;
    color = vec4(mix(base, vec3(1.0) * (0.7 + 0.3 * n), flash), 1.0);
}
