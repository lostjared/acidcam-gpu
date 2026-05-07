#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Luma-driven dissolve: bright areas of prev disappear first.
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    float luma = dot(prev.rgb, vec3(0.299, 0.587, 0.114));
    float w = 0.1;
    float t = smoothstep(luma - w, luma + w, fade_alpha);
    color = mix(prev, curr, t);
}
