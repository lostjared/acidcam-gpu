#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    vec2 uvP = vec2(tc.x, tc.y + fade_alpha);
    vec2 uvC = vec2(tc.x, tc.y - (1.0 - fade_alpha));
    if (tc.y < fade_alpha) {
        color = texture(samp, uvC);
    } else {
        color = texture(prev_samp, uvP);
    }
}
