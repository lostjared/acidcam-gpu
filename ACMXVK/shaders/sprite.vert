#version 450

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec2 out_uv;

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

void main() {
    vec2 sprite_size = vec2(pc.spriteSizeW, pc.spriteSizeH);
    vec2 center = vec2(pc.spritePosX, pc.spritePosY) + sprite_size * 0.5;
    vec2 local_pos = (in_pos - vec2(0.5)) * sprite_size;
    float rotation = radians(pc.rotationDegrees);
    float rotation_sin = sin(rotation);
    float rotation_cos = cos(rotation);
    vec2 rotated_pos = vec2(
        local_pos.x * rotation_cos - local_pos.y * rotation_sin,
        local_pos.x * rotation_sin + local_pos.y * rotation_cos);
    vec2 pixel_pos = center + rotated_pos;
    vec2 ndc = vec2(
        (pixel_pos.x / max(pc.screenWidth, 1.0)) * 2.0 - 1.0,
        (pixel_pos.y / max(pc.screenHeight, 1.0)) * 2.0 - 1.0);

    gl_Position = vec4(ndc, 0.0, 1.0);
    out_uv = in_uv;
}
