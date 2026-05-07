#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // curr slides in from the right, prev slides out to the left.
    vec2 uvP = vec2(tc.x + fade_alpha, tc.y);
    vec2 uvC = vec2(tc.x - (1.0 - fade_alpha), tc.y);
    if (tc.x < fade_alpha) {
        color = texture(samp, uvC);
    } else {
        color = texture(prev_samp, uvP);
    }
}
