#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;
void main() {
    // Rotation reveal: rotate sample around centre.
    float bump = sin(fade_alpha * 3.14159);
    float angle = bump * 1.5708; // up to 90 degrees mid-transition
    vec2 c = tc - 0.5;
    float ca = cos(angle);
    float sa = sin(angle);
    vec2 uv = vec2(c.x * ca - c.y * sa, c.x * sa + c.y * ca) + 0.5;
    uv = clamp(uv, 0.0, 1.0);
    vec4 curr = texture(samp, uv);
    vec4 prev = texture(prev_samp, uv);
    color = mix(prev, curr, fade_alpha);
}
