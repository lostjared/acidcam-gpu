# ACMXVK

ACMXVK is an in-progress Vulkan port of the ACMX2 real-time video shader
engine. The goal is to preserve ACMX2's workflow and behavior while replacing
the MX2/OpenGL rendering path with the installed
[MXVK](https://github.com/lostjared/MXVK) engine and Vulkan SPIR-V shaders.

The port is currently at **Increment 5D**. It is usable for video, camera, and
still-image shader processing, but it is not yet a complete replacement for
ACMX2.

## Translation progress

| Area | Status | Notes |
| --- | --- | --- |
| Standalone CMake project | Complete | Builds against installed MXVK and MXWrite and provides an `uninstall` target. |
| Window and Vulkan lifecycle | Complete | MXVK owns the window, device, swapchain, rendering, screenshots, and validation integration. |
| Video, camera, and image input | Complete | OpenCV-backed capture supports files, camera devices, and still images. |
| Basic shader playback | Complete | Loads Vulkan fragment shaders compiled to `.spv`. |
| Shader libraries | Complete | Prefers `library.json` and falls back to `index.txt`; supports nested paths and object or string entries. |
| Shader selection | Complete | Supports selection by index or filename and keyboard switching. |
| Multipass and playlists | Implemented | Includes named playlist nodes, multipass chains, sequential autopilot, and random autopilot. |
| Frame history/texture cache | Implemented | Uses a Vulkan `sampler2DArray` ring buffer with configurable size and write delay. |
| Custom library uniforms | Implemented | Up to 64 validated floats from `library.json`, with repeatable `--uniform name=value` overrides. |
| Video recording | Implemented | MXWrite supports software or hardware encoders, encoder options, no-drop mode, duration and size limits, and optional audio copying. |
| PNG output | Implemented | Supports full PNG sequences and periodic generated frames. |
| Rotation and final-output flip | Implemented | Applies input rotation and optional final display/recording flip. |
| Runtime playback controls | Implemented | Supports video pause, rendering freeze, shader-time toggle/stepping/speed, and fullscreen switching. |
| ACMX2 GLSL compatibility | Partial | Existing GLSL effects must be translated to the MXVK Vulkan descriptor ABI and compiled to SPIR-V. |
| Audio-reactive shader data | Not yet ported | Audio spectrum, amplitude, and history resources from ACMX2 remain future work. |
| MIDI controls | Not yet ported | ACMX2 MIDI uniform control is not present yet. |
| CUDA filters and DNN effects | Not yet ported | The current pipeline uses MXVK/OpenCV input and Vulkan shader passes. |
| 3D model pipeline | Not yet ported | ACMX2 model rendering remains outside the current increment. |
| Qt interface integration | Not yet ported | ACMXVK currently provides the command-line renderer only. |

## Requirements

- A C++20 compiler and CMake 3.20 or newer
- Vulkan SDK 1.4 with `glslc`
- A current system installation of MXVK built with `-DVALIDATION=ON -DCV=ON`
- MXWrite from the MXVK source tree
- SDL3, SDL3_ttf, Vulkan, OpenCV, PNG, ZLIB, glm, and FFmpeg development files
- Optional SDL3_mixer, JPEG, and CUDA dependencies when enabled by the installed MXVK package

The Vulkan environment used for this project can be loaded with:

```bash
source ~/vulkan.sh
```

## Build

From the `acidcam-gpu` repository root:

```bash
source ~/vulkan.sh
cmake -S ACMXVK -B build/acmxvk -DVALIDATION=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/acmxvk -j
./build/acmxvk/acmxvk --help
```

To install or uninstall using the selected CMake prefix:

```bash
cmake --install build/acmxvk
cmake --build build/acmxvk --target uninstall
```

Increment 5D requires the matching MXVK custom-uniform changes to be installed
before ACMXVK is compiled against the system package.

## Examples

Process a video with a SPIR-V shader library:

```bash
./build/acmxvk/acmxvk \
    --input input.mp4 \
    --shaders /path/to/spv-library \
    --shader-file effect.spv
```

Render a still image for five seconds and encode it with a software encoder:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --shaders /path/to/spv-library \
    --duration 5 \
    --encode-codec software \
    --no-drop \
    --output output.mp4
```

Enable an eight-layer frame-history cache:

```bash
./build/acmxvk/acmxvk \
    --input input.mp4 \
    --shaders /path/to/spv-library \
    --texture-cache \
    --texture-cache-size 8 \
    --cache-delay 1
```

Override a custom float declared by `library.json`:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --shaders /path/to/spv-library \
    --uniform square_size=64 \
    --duration 2 \
    --output output.mp4
```

## SPIR-V shader-library format

ACMXVK accepts `library.json` entries as strings or objects containing a
`file` field:

```json
{
    "version": 1,
    "custom_uniforms": {
        "square_size": {
            "minimum": 1.0,
            "maximum": 128.0,
            "step": 1.0,
            "value": 32.0
        }
    },
    "shaders": [
        "basic.spv",
        { "file": "history/echo.spv" }
    ]
}
```

When `library.json` is absent, `index.txt` is read as one relative `.spv` path
per line. Absolute paths, parent-directory traversal, files outside the library,
and non-SPIR-V entries are rejected.

Custom uniforms are packed in manifest declaration order. A Vulkan shader can
append the custom array to MXVK's binding-1 block:

```glsl
layout(set = 0, binding = 1) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
    vec4 custom_uniforms[16];
} ext;

#define square_size ext.custom_uniforms[0].x
```

Uniform number `N` is stored in
`ext.custom_uniforms[N / 4][N % 4]`. See
`shaders/custom_uniform.frag` for a working reference shader.

The remaining Vulkan bindings are:

- Set 0, binding 0: current RGBA frame as `sampler2D`
- Set 0, binding 1: mouse, frame state, resolution, time, history metadata, and custom floats
- Set 0, binding 2: optional RGBA history as `sampler2DArray`

## Runtime controls

- Up/Down: change the shader or playlist node
- Shift+Up/Down: change the final shader while using a playlist
- P: toggle playlist mode
- P without a playlist: pause or resume video input
- L: freeze or resume both input and shader animation
- T: enable or disable shader-time advancement
- U/I: step shader time forward or backward by 0.05
- Page Up/Page Down: increase or decrease shader-time speed
- F: toggle fullscreen
- M: toggle the configured multipass chain
- J: toggle random autopilot
- Y: toggle sequential autopilot
- Space: bypass or enable shader effects
- F10: capture a screenshot when `--enable-screenshot` is active
- Escape: quit

Run `acmxvk --help` for the complete command-line reference.

## Validation and current testing

Development builds are tested with the Vulkan SDK selected by `~/vulkan.sh`
and with validation enabled in both MXVK and ACMXVK. The current increment has
been exercised with shader-library loading, multipass rendering, configurable
history caches, MXWrite encoding, and custom-uniform rendering. The known
duplicate vkBasalt implicit-layer warning is external to ACMXVK.

## Development note

I have been using the **Codex CLI from OpenAI** as an
engineering aid while porting ACMX2 to MXVK. Codex has assisted with incremental
code translation, CMake integration, shader conversion, debugging, Vulkan
validation testing, and documentation. Project direction, testing decisions,
and maintenance remain under the project owner's control.
