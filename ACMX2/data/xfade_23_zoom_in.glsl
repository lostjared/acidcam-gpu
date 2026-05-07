#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // curr zooms in from a small centred quad.
    float scale = mix(0.05, 1.0, fade_alpha);
    vec2 uv = (tc - 0.5) / scale + 0.5;
    vec4 prev = texture(prev_samp, tc);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        color = prev;
    } else {
        vec4 curr = texture(samp, uv);
        color = mix(prev, curr, fade_alpha);
    }
}
