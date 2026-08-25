#version 450

layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;

// Binding 0: Previous stage output
layout(set = 0, binding = 0) uniform sampler2D input_image;

// Binding 1: MXVK Engine State
layout(set = 0, binding = 1, std140) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
    vec4 custom_uniforms[16];
    vec4 audio_bands;
    vec4 audio_history;
} ext;

// Push Constants: Available only in Fragment stage
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

// ACMXVK / MX2 Compatibility Aliases
#define iResolution max(ext.u0.zw, vec2(1.0))
#define iTime ext.u2.y
#define iTimeDelta ext.u1.x
#define iFrame ext.u2.x
#define amp_low ext.audio_bands.x
#define amp_mid ext.audio_bands.y
#define amp_high ext.audio_bands.z

void main() {
    // Respect the Spacebar bypass toggle
    if (pc.effects_on < 0.5) {
        color = texture(input_image, tc);
        return;
    }

    // Read input pixel
    vec4 source_color = texture(input_image, tc);

    // --- GLITCH / ART LOGIC HERE ---
    vec3 final_color = source_color.rgb;
    // -------------------------------

    color = vec4(final_color, source_color.a);
}
