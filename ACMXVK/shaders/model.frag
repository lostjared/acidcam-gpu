#version 450

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 1) in vec3 frag_normal;

layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D source_texture;

void main() {
    // ACMX2's bypass path presents the source texture without adding model
    // lighting. Keep that behavior so toggling effects does not darken the
    // inward-facing skybox.
    color = texture(source_texture, frag_tex_coord);
}
