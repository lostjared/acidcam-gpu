#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Vertical shutter close on prev, then opens to reveal curr.
    float dist = abs(tc.y - 0.5) * 2.0;
    // Gap shrinks to 0 at fade_alpha=0.5 then reopens revealing curr.
    float threshold = abs(fade_alpha - 0.5) * 2.0;
    vec4 curr = texture(samp, tc);
    vec4 prev = texture(prev_samp, tc);
    if (dist > threshold) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        color = (fade_alpha < 0.5) ? prev : curr;
    }
}
