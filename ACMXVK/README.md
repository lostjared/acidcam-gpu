# ACMXVK

ACMXVK is an in-progress Vulkan port of the ACMX2 real-time video shader
engine. The goal is to preserve ACMX2's workflow and behavior while replacing
the MX2/OpenGL rendering path with the installed
[MXVK](https://github.com/lostjared/MXVK) engine and Vulkan SPIR-V shaders.

The port is currently at **Increment 5O**. It is usable for video, camera, and
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
| Audio-reactive shader data | Implemented | RtAudio capture, an FFmpeg-decoded media file, or an M3U/M3U8 playlist can drive amplitude, frequency, peak, RMS, smoothed amplitude, low/mid/high bands, a current-frame FFT, and configurable FFT history. Live and file audio support output pass-through and AAC muxing; file audio also supports repeat and stop-at-EOF behavior. |
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
- Optional RtAudio development files when building with `-DAUDIO=ON`

The Vulkan environment used for this project can be loaded with:

```bash
source ~/vulkan.sh
```

## Build

From the `acidcam-gpu` repository root:

```bash
source ~/vulkan.sh
cmake -S ACMXVK -B build/acmxvk -DVALIDATION=ON -DAUDIO=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/acmxvk -j
./build/acmxvk/acmxvk --help
```

To install or uninstall using the selected CMake prefix:

```bash
cmake --install build/acmxvk
cmake --build build/acmxvk --target uninstall
```

Audio support is optional and remains disabled when `-DAUDIO=ON` is omitted.
Increment 5H added the MXVK spectrum-history descriptor and UBO suffix, so that
matching MXVK version must be installed before compiling ACMXVK with
`-DAUDIO=ON`. Increment 5I changes only ACMXVK and does not require another MXVK
reinstall. Increments 5J through 5O also change only ACMXVK.

### Apple Silicon and MoltenVK

On an Apple Silicon Mac, source the Vulkan SDK environment and configure an
arm64 build with MoltenVK enabled:

```bash
source ~/vulkan.sh
cmake -S ACMXVK -B build/acmxvk-macos \
    -DACMXVK_USE_MOLTENVK=ON \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_PREFIX_PATH="$VULKAN_SDK;/opt/homebrew;/usr/local" \
    -DVALIDATION=ON \
    -DAUDIO=OFF \
    -DCMAKE_BUILD_TYPE=Debug
cmake --build build/acmxvk-macos -j
./build/acmxvk-macos/acmxvk --help
```

`ACMXVK_USE_MOLTENVK` defaults to `ON` on Apple platforms. The CMake project
uses the Vulkan SDK loader, which discovers MoltenVK as its macOS Vulkan driver.
It intentionally does not link `libMoltenVK` directly because SDL, volk, and
validation layers must use the same Vulkan loader dispatch path.
Explicitly setting `CMAKE_OSX_ARCHITECTURES` remains useful when CMake is run
from a translated shell or when switching between arm64 and universal builds.

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

Preview the included shader with live audio reactivity:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/audio_reactive.frag.spv \
    --enable-audio \
    --audio-input default
```

Use an audio or video file as the reactive source without opening a microphone:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/audio_reactive.frag.spv \
    --audio-file soundtrack.mp3
```

Display the current 256-bin FFT spectrum over an image:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/audio_spectrum.frag.spv \
    --enable-audio \
    --audio-input default
```

Display an eight-frame FFT waterfall over an image:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/audio_history.frag.spv \
    --enable-audio \
    --audio-input default \
    --enable-audio-buffers 8
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
    vec4 audio_bands;
    vec4 audio_history;
} ext;

#define square_size ext.custom_uniforms[0].x
```

Uniform number `N` is stored in
`ext.custom_uniforms[N / 4][N % 4]`. See
`shaders/custom_uniform.frag` for a working reference shader.

The remaining Vulkan bindings are:

- Set 0, binding 0: current RGBA frame as `sampler2D`
- Set 0, binding 1: mouse, frame state, resolution, time, history metadata, custom floats, and audio bands
- Set 0, binding 2: optional RGBA history as `sampler2DArray`
- Set 0, binding 3: current 256-bin audio FFT as an R32 `sampler1D`
- Set 0, binding 4: optional 256-bin FFT history as an R32 `sampler1DArray`

With an `AUDIO=ON` build, `--enable-audio` uses live RtAudio input while
`--audio-file <media>` uses the first audio stream decoded by FFmpeg. It also
accepts an M3U or M3U8 playlist. Both paths map the same audio metrics into the
binding-1 block:

| ACMX2 name | MXVK field | Meaning |
| --- | --- | --- |
| `amp` | `ext.u1.y` | Mean absolute amplitude |
| `iamp` | `ext.u1.z` | Zero-crossing frequency estimate in Hz |
| `iSampleRate` | `ext.u2.z` | Active input sample rate |
| `amp_peak` | `ext.u2.w` | Sensitivity-scaled peak level |
| `amp_rms` | `ext.u3.z` | Sensitivity-scaled RMS level |
| `amp_smooth` | `ext.u3.w` | Sensitivity-scaled smoothed amplitude |
| `amp_low` | `ext.audio_bands.x` | Sensitivity-scaled energy below 300 Hz |
| `amp_mid` | `ext.audio_bands.y` | Sensitivity-scaled energy from 300 through 3000 Hz |
| `amp_high` | `ext.audio_bands.z` | Sensitivity-scaled energy above 3000 Hz |

See `shaders/audio_reactive.frag` for a working shader. Audio values remain
zero when capture is disabled or an input device cannot be opened.

File audio is decoded up front to mono 44.1 kHz floating-point samples. It is a
silent analysis source by default and advances according to ACMXVK's output
frame rate. Add `--pass-through` to play it through RtAudio's default output;
`--audio-output <index>` selects another output listed by `--list-devices`.
During pass-through, the output device becomes the master audio clock and the
shader analysis follows its playback position. This keeps visual reactivity
aligned even when rendering runs faster or slower than the requested FPS.

When an encoded `--output` is open, file audio is automatically encoded as
mono AAC at 192 kbps and muxed into the completed video. This is implemented
directly with the linked FFmpeg libraries; ACMXVK does not invoke the `ffmpeg`
command-line program. The original encoded video remains untouched until the
temporary mux output has been finalized successfully. MP4, MOV, and other
AAC-compatible containers are supported; if a selected container rejects AAC,
ACMXVK reports the mux failure, removes the temporary file, and preserves the
video-only recording.

When live `--enable-audio` input and an encoded `--output` are active, ACMXVK
also records the microphone automatically. Capture begins immediately before
the first frame is submitted to MXWrite and stops before the writer closes, so
device and shader initialization do not add leading audio. Input channels are
downmixed to mono, resampled to 44.1 kHz when needed, encoded as AAC at 192
kbps, and muxed through the same linked-library path as file audio. No WAV file
or external process is required. `--copy-audio` takes precedence and disables
live-input recording so the selected video input's original audio is copied
instead.

Add `--pass-through` to monitor live input through RtAudio's default output, or
select an output from `--list-devices` with `--audio-output <index>`. Live
monitoring uses one full-duplex stream, so the samples sent to the output are
the same samples used for shader analysis and recording. ACMXVK selects a
sample rate supported by both devices and duplicates mono input into a stereo
output when needed. Headphones are recommended to avoid acoustic feedback.

Without repeat, the muxed result is limited to the shorter of the recorded
video and decoded audio. With `--audio-repeat`, the complete file or playlist
is repeated to the recorded video duration. `--audio-trunc` stops recording at
the file-audio EOF; this also permits a still-image recording without an
explicit `--duration`. At end-of-stream without truncation, preview continues
with zero-valued audio metrics. When repeat and truncation are both supplied,
repeat keeps the source active, matching ACMX2 behavior.

For example, loop a song with audible pass-through:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/audio_reactive.frag.spv \
    --audio-file song.mp3 \
    --pass-through \
    --audio-repeat
```

Replace `--audio-repeat` with `--audio-trunc` to close the application at the
end of the song.

Record five seconds of processed video and mux repeated file audio into it:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/audio_reactive.frag.spv \
    --audio-file song.mp3 \
    --audio-repeat \
    --duration 5 \
    --output output.mp4
```

Record processed video with synchronized live microphone audio:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/audio_reactive.frag.spv \
    --enable-audio \
    --audio-input default \
    --duration 5 \
    --output live-output.mp4 \
    --enable-vsync
```

Monitor that microphone through the default output while recording:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/audio_reactive.frag.spv \
    --enable-audio \
    --audio-input default \
    --pass-through \
    --audio-output default \
    --duration 5 \
    --output monitored-live-output.mp4 \
    --enable-vsync
```

M3U and M3U8 playlists are read in order. Blank lines, `#EXTM3U`, `#EXTINF`,
and other comment lines are ignored; relative entries are resolved against the
playlist directory. Unusable tracks are reported and skipped as long as at
least one entry can be decoded. `--audio-repeat` restarts the complete playlist,
and `--audio-trunc` exits only after its final usable track.

```m3u
#EXTM3U
#EXTINF:-1,First track
music/first.flac
#EXTINF:-1,Second track
music/second.mp3
```

`audio_bands` and `audio_history` are appended after `custom_uniforms[16]`.
This preserves every existing field and custom-float offset; older shaders may
stop their uniform block at `u3`, `custom_uniforms`, or `audio_bands` without
declaring the newer suffixes.

The FFT uses a 512-sample Hann window and exposes 256 linear-frequency bins.
Coordinate `0.0` is DC and `1.0` approaches the Nyquist frequency. Declare it
in a Vulkan shader as:

```glsl
layout(set = 0, binding = 3) uniform sampler1D spectrum;
float energy = texture(spectrum, frequency).r;
```

See `shaders/audio_spectrum.frag` for a complete visualization. The descriptor
can be used alongside binding-2 frame history in the same MXVK extended layout.

`--enable-audio-buffers N` allocates binding 4 as a circular FFT-history array.
The requested depth is clamped to the Vulkan device's maximum image-array-layer
count. `ext.audio_history.x` contains the physical layer holding the newest
spectrum, `ext.audio_history.y` contains the allocated layer count, and
`ext.audio_history.z` contains the number of bins. A shader can sample an age
without copying layers:

```glsl
layout(set = 0, binding = 4) uniform sampler1DArray spectrum_history;
int count = max(int(ext.audio_history.y + 0.5), 1);
int head = int(ext.audio_history.x + 0.5);
int age = 3;
int layer = (head - (age % count) + count) % count;
float old_energy = texture(spectrum_history, vec2(0.08, float(layer))).r;
```

See `shaders/audio_history.frag` for a current-versus-history waterfall test.

## Runtime controls

- Up/Down: change the shader or playlist node
- Shift+Up/Down: change the final shader while using a playlist
- P: toggle playlist mode
- P without a playlist: pause or resume video input
- L: freeze or resume both input and shader animation
- T: enable or disable shader-time advancement
- U/I: step shader time forward or backward by 0.05
- Page Up/Page Down: increase or decrease shader-time speed
- Insert/Delete: increase or decrease live audio sensitivity
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
history caches, MXWrite encoding, custom-uniform rendering, optional live audio
metrics, FFmpeg-decoded file reactivity, routed-tone FFT visualization, and FFT
spectrum history. The known duplicate vkBasalt
implicit-layer warning is external to ACMXVK.

## Development note

I have been using the **Codex CLI from OpenAI** as an
engineering aid while porting ACMX2 to MXVK. Codex has assisted with incremental
code translation, CMake integration, shader conversion, debugging, Vulkan
validation testing, and documentation. Project direction, testing decisions,
and maintenance remain under the project owner's control.
