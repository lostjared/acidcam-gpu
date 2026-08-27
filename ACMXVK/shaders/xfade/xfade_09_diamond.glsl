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
    vec4 curr = texture(samp, tc);
    vec4 prev = sample_previous(tc);
    vec2 grid = vec2(20.0, 15.0);
    vec2 g = fract(tc * grid) - 0.5;
    float d = (abs(g.x) + abs(g.y)) * 2.0;
    float edge = 0.08;
    float t = smoothstep(fade_alpha - edge, fade_alpha + edge, 1.0 - d);
    color = mix(prev, curr, t);
}
