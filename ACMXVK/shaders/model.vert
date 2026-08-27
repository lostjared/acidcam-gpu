#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_tex_coord;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec2 frag_tex_coord;
layout(location = 1) out vec3 frag_normal;

// Binding 1 is reserved for the same fragment-uniform block used by ACMXVK's
// 2D shaders.  MXVK places the model transforms at binding 2 when extended
// model fragment uniforms are enabled.
layout(set = 0, binding = 2) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    // xyz: ACMX2 wave amplitudes; w: wave phase. All zero when disabled.
    vec4 effects;
} uniforms;

void main() {
    vec3 position = in_position;
    mat3 deformation_jacobian = mat3(1.0);
    vec3 wave_amplitude = uniforms.effects.xyz;
    float wave_phase = uniforms.effects.w;
    if (any(greaterThan(wave_amplitude, vec3(0.0)))) {
        const float wave_frequency = 2.0;

        float x_factor = wave_amplitude.x * wave_frequency *
                         cos(position.y * wave_frequency + wave_phase);
        vec3 x_derivative = vec3(1.0, x_factor, 0.0);
        position.x += wave_amplitude.x *
                      sin(position.y * wave_frequency + wave_phase);

        float y_factor = wave_amplitude.y * wave_frequency *
                         cos(position.x * wave_frequency + wave_phase + 120.0);
        vec3 y_derivative = vec3(0.0, 1.0, 0.0) +
                            y_factor * x_derivative;
        position.y += wave_amplitude.y *
                      sin(position.x * wave_frequency + wave_phase + 120.0);

        float z_factor = wave_amplitude.z * wave_frequency *
                         cos(position.y * wave_frequency + wave_phase + 240.0);
        vec3 z_derivative = vec3(0.0, 0.0, 1.0) +
                            z_factor * y_derivative;
        position.z += wave_amplitude.z *
                      sin(position.y * wave_frequency + wave_phase + 240.0);

        deformation_jacobian = mat3(
            vec3(x_derivative.x, y_derivative.x, z_derivative.x),
            vec3(x_derivative.y, y_derivative.y, z_derivative.y),
            vec3(x_derivative.z, y_derivative.z, z_derivative.z));
    }

    mat3 normal_matrix = transpose(inverse(mat3(uniforms.model)));
    vec3 deformed_normal =
        transpose(inverse(deformation_jacobian)) * in_normal;
    frag_normal = normalize(normal_matrix * deformed_normal);
    frag_tex_coord = in_tex_coord;
    gl_Position = uniforms.projection * uniforms.view * uniforms.model *
                  vec4(position, 1.0);
}
