#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_tex_coord;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec2 frag_tex_coord;
layout(location = 1) out vec3 frag_normal;

layout(set = 0, binding = 1) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 effects;
} uniforms;

void main() {
    mat3 normal_matrix = transpose(inverse(mat3(uniforms.model)));
    frag_normal = normalize(normal_matrix * in_normal);
    frag_tex_coord = in_tex_coord;
    gl_Position = uniforms.projection * uniforms.view * uniforms.model *
                  vec4(in_position, 1.0);
}
