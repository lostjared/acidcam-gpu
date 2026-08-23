#version 450

// MXVK conversion of shaders_repo/shaders/E/echo_cache.glsl.
// The original separate uniforms are mapped to MXVK's descriptor ABI:
//   binding 0: current RGBA frame
//   binding 1: SpriteExtended UBO (u3.x stores history_head)
//   binding 2: eight-layer RGBA history texture array

layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D samp;

layout(set = 0, binding = 1) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
} ext;

layout(set = 0, binding = 2) uniform sampler2DArray history;

layout(push_constant) uniform SpritePushConstants {
    float screen_width;
    float screen_height;
    float sprite_pos_x;
    float sprite_pos_y;
    float sprite_size_w;
    float sprite_size_h;
    float effects_on;
    float rotation_degrees;
    vec4 params;
} pc;

const int HISTORY_SIZE = 8;

int history_layer(int index) {
    int history_head = int(ext.u3.x + 0.5);
    return (history_head + clamp(index, 0, HISTORY_SIZE - 1)) % HISTORY_SIZE;
}

vec4 sample_cache(int index, vec2 uv) {
    return texture(history, vec3(uv, float(history_layer(index))));
}

void main() {
    color = texture(samp, tc);
    if (pc.effects_on < 0.5) {
        return;
    }

    vec2 offset = vec2(0.01);
    for (int index = 0; index <= 6; ++index) {
        color = mix(color, sample_cache(index, tc + offset), 0.5);
        offset += vec2(0.02, 0.01);
    }
    color = clamp(color, vec4(0.0), vec4(1.0));
}
