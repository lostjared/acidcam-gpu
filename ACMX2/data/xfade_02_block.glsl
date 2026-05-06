#version 330 core
in vec2 tc;
out vec4 color;
uniform sampler2D samp;
uniform sampler2D prev_samp;
uniform float fade_alpha;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 gridSize = vec2(40.0, 30.0);
    vec2 blockId = floor(tc * gridSize);
    vec2 blockOffset = vec2(
        rand(blockId) * 2.0 - 1.0,
        rand(blockId + vec2(1.33, 2.71)) * 2.0 - 1.0
    );
    float intensity = 4.0 * fade_alpha * (1.0 - fade_alpha);
    float maxTearDistance = 0.15;
    vec2 uvDisplaced = clamp(tc + (blockOffset * intensity * maxTearDistance), 0.0, 1.0);
    vec4 curr = texture(samp, uvDisplaced);
    vec4 prev = texture(prev_samp, uvDisplaced);
    float threshold = fade_alpha + (blockOffset.x * 0.2);
    color = (threshold < 0.5) ? prev : curr;
}
