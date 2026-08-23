#version 450

// Increment 5H FFT history reference shader. Binding 4 contains a circular
// sampler1DArray; audio_history.x is its newest physical layer and
// audio_history.y is the allocated layer count.

layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D samp;

layout(set = 0, binding = 1) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
    vec4 custom_uniforms[16];
    vec4 audio_bands;
    vec4 audio_history;
} ext;

layout(set = 0, binding = 3) uniform sampler1D spectrum;
layout(set = 0, binding = 4) uniform sampler1DArray spectrum_history;

int history_layer(int age, int count, int head) {
    return (head - (age % count) + count) % count;
}

void main() {
    vec3 source = texture(samp, tc).rgb;
    int count = max(int(ext.audio_history.y + 0.5), 1);
    int head = clamp(int(ext.audio_history.x + 0.5), 0, count - 1);
    int age = clamp(int((1.0 - tc.y) * float(count)), 0, count - 1);
    int layer = history_layer(age, count, head);

    float frequency = tc.x * tc.x;
    float history_energy = texture(spectrum_history,
                                   vec2(frequency, float(layer))).r;
    float live_energy = texture(spectrum, frequency).r;
    float intensity = clamp(sqrt(max(history_energy, 0.0)) * 4.0, 0.0, 1.0);
    float live_line = smoothstep(0.025, 0.0, abs(tc.y - 0.985));

    float age_mix = float(age) / float(max(count - 1, 1));
    vec3 history_color = mix(vec3(1.0, 0.25, 0.08),
                             vec3(0.08, 0.3, 1.0), age_mix);
    vec3 result = mix(source * 0.25, history_color, intensity * 0.95);
    result += vec3(clamp(sqrt(max(live_energy, 0.0)) * 2.0, 0.0, 1.0)) *
              live_line;
    color = vec4(clamp(result, vec3(0.0), vec3(1.0)), 1.0);
}
