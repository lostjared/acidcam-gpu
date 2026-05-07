#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // prev zooms out (shrinks) while curr fades in underneath.
    float scale = mix(1.0, 0.05, fade_alpha);
    vec2 uv = (tc - 0.5) / scale + 0.5;
    vec4 curr = texture(samp, tc);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        color = curr;
    } else {
        vec4 prev = texture(prev_samp, uv);
        color = mix(prev, curr, fade_alpha);
    }
}
