#version 450

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 1) in vec3 frag_normal;

layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D source_texture;

void main() {
    vec4 source = texture(source_texture, frag_tex_coord);
    vec3 normal = normalize(frag_normal);
    vec3 light_direction = normalize(vec3(0.25, 0.35, 1.0));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float lighting = 0.65 + 0.35 * diffuse;
    color = vec4(source.rgb * lighting, source.a);
}
