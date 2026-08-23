#version 450

// MXVK conversion of shaders_repo/shaders/E/echo_cache.glsl.
// The original separate uniforms are mapped to MXVK's descriptor ABI:
//   binding 0: current RGBA frame
//   binding 1: SpriteExtended UBO (u3.x stores history_head)
//   binding 2: runtime-sized RGBA history texture array

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

const int MAX_HISTORY_SIZE = 64;

int history_size() {
    return clamp(int(ext.u3.y + 0.5), 1, MAX_HISTORY_SIZE);
}

int history_layer(int index) {
    int history_head = int(ext.u3.x + 0.5);
    int layer_count = history_size();
    return (history_head + clamp(index, 0, layer_count - 1)) % layer_count;
}

vec4 sample_cache(int index, vec2 uv) {
    return texture(history, vec3(uv, float(history_layer(index))));
}

void main() {
    color = texture(samp, tc);
    if (pc.effects_on < 0.5) {
        return;
    }

    int sample_count = max(history_size() - 1, 1);
    vec2 offset = vec2(0.01);
    for (int index = 0; index < MAX_HISTORY_SIZE; ++index) {
        if (index >= sample_count) {
            break;
        }
        color = mix(color, sample_cache(index, tc + offset), 0.5);
        offset += vec2(0.02, 0.01);
    }
    color = clamp(color, vec4(0.0), vec4(1.0));
}
