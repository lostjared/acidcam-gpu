# ACMXVK

ACMXVK is an in-progress Vulkan port of the ACMX2 real-time video shader
engine. The goal is to preserve ACMX2's workflow and behavior while replacing
the MX2/OpenGL rendering path with the installed
[MXVK](https://github.com/lostjared/MXVK) engine and Vulkan SPIR-V shaders.

The port is currently at **Increment 7E**. It is usable for video, camera, and
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
| Frame history/texture cache | Implemented | Uses a Vulkan `sampler2DArray` ring buffer with configurable size and write delay. CUDA-filter builds place the post-filter image in history through direct CUDA/Vulkan layered-image interop. |
| Custom library uniforms | Implemented | Up to 64 validated floats from `library.json`, with repeatable `--uniform name=value` overrides. |
| Video recording | Implemented | MXWrite supports software or hardware encoders, encoder options, no-drop mode, duration and size limits, and optional audio copying. |
| PNG output | Implemented | Supports full PNG sequences and periodic generated frames. |
| Rotation and final-output flip | Implemented | Applies input rotation and optional final display/recording flip. |
| Runtime playback controls | Implemented | Supports video pause, rendering freeze, shader-time toggle/stepping/speed, and fullscreen switching. |
| ACMX2 GLSL compatibility | Partial | Existing GLSL effects must be translated to the MXVK Vulkan descriptor ABI and compiled to SPIR-V. |
| Audio-reactive shader data | Implemented | RtAudio capture, an FFmpeg-decoded media file, or an M3U/M3U8 playlist can drive amplitude, frequency, peak, RMS, smoothed amplitude, low/mid/high bands, a current-frame FFT, and configurable FFT history. Live and file audio support configurable shader warmup, adjustable-gain output pass-through, and AAC muxing; live recording has independent gain, while file audio also supports repeat and stop-at-EOF behavior. |
| MIDI controls | Partial | Optional RtMidi support handles input enumeration, a bounded callback queue, live monitoring, ACMX2 MIDI Map `.midi_cfg` files, Slider 1–4 custom uniforms, and ACMXVK-equivalent playback actions. Paired knobs use ACMX2's centered, velocity-sensitive repeat behavior. |
| CUDA filters | Partial | Optional `acidcam-gpu` integration accepts filter chains and temporal-buffer sizes, keeps video/camera RGBA and input rotation resident on the GPU through filtering and Vulkan upload/history, and supports ACMX2-compatible Left/Right selection from the keyboard or MIDI maps. |
| DNN effects | Not yet ported | OpenCV DNN segmentation, edge detection, and generic ONNX processing remain outside the current increment. |
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
- Optional RtMidi development files when building with `-DMIDI=ON`
- Optional CUDA Toolkit, CUDA-enabled OpenCV and MXVK, and an installed
  `acidcam-gpu` CMake package when building with `-DWITH_CUDA=ON`

The Vulkan environment used for this project can be loaded with:

```bash
source ~/vulkan.sh
```

## Build

From the `acidcam-gpu` repository root:

```bash
source ~/vulkan.sh
cmake -S ACMXVK -B build/acmxvk -DVALIDATION=ON -DAUDIO=ON -DMIDI=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/acmxvk -j
./build/acmxvk/acmxvk --help
```

To install or uninstall using the selected CMake prefix:

```bash
cmake --install build/acmxvk
cmake --build build/acmxvk --target uninstall
```

Audio and MIDI support are optional and remain disabled when their CMake
options are omitted.
Increment 5H added the MXVK spectrum-history descriptor and UBO suffix, so that
matching MXVK version must be installed before compiling ACMXVK with
`-DAUDIO=ON`. Increment 5I changes only ACMXVK and does not require another MXVK
reinstall. Increments 5J through 5R, 6A through 6C, and 7A through 7B also
change only ACMXVK. Increment 7A requires the existing MXVK and `acidcam-gpu`
installations to have CUDA support. Increment 7C adds MXVK's layered CUDA
history upload API, so MXVK must be rebuilt and reinstalled before building
ACMXVK 7C with `-DWITH_CUDA=ON`. Increment 7D adds a CUDA `GpuMat` input overload
to acidcam-gpu's temporal frame buffer, so acidcam-gpu must be rebuilt and
reinstalled before building ACMXVK 7D with `-DWITH_CUDA=ON`; MXVK does not need
another reinstall for 7D. Increment 7E changes only ACMXVK, so neither MXVK nor
acidcam-gpu needs another reinstall.

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
Use `--pass-through-gain <0.0-4.0>` when the input device does not expose a
system level control. Gain affects only monitored output: shader metrics and
recorded or muxed audio retain the original samples. Values above `1.0` amplify
the monitor signal, and samples are clamped to the floating-point audio range
to prevent overflow.

Use `--record-gain <0.0-2.0>` to amplify quiet live-input samples before they
are encoded into the final video's AAC stream. Recording gain is independent
from `--pass-through-gain` and `--sense`: it does not change headphone volume
or shader response. Unity gain remains the default, and amplified samples are
clamped to the floating-point audio range before encoding.

Audio-reactive shader values ramp from zero at startup at a default rate of
`0.5` per second, reaching full strength in about two seconds. Set
`--audio-warm-rate <rate>` to change the slope, or use
`--audio-warm-rate 0` to disable warmup. The envelope scales amplitude,
peak/RMS/bands, the current FFT, and FFT history. It does not alter monitoring,
recording, frequency estimation, or the reported sample rate.

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
    --audio-warm-rate 0.5 \
    --pass-through \
    --audio-output default \
    --pass-through-gain 2.0 \
    --record-gain 1.5 \
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

### MIDI input and mappings

Configure with `-DMIDI=ON` to enable RtMidi. List available input ports with:

```bash
./build/acmxvk/acmxvk --list-midi
```

Open a port and print its incoming byte messages while ACMXVK runs:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/audio_reactive.frag.spv \
    --midi-device 0 \
    --midi-monitor
```

When `--midi-device` is omitted, MIDI monitoring or mapping opens port zero.
RtMidi's callback feeds a bounded, thread-safe queue drained by the render loop;
if more than 256 unprocessed messages arrive, ACMXVK retains the newest messages
and reports the drop count.

Increment 6B reads the same `.midi_cfg` format written by
`ACMX2/interface/midi-map`. Use that tool to capture controller messages, save
the map, and load it with `--midi-map`. Slider 1–4 actions (`600:601` through
`606:607`) update custom uniforms named `slider1` through `slider4`. Values are
normalized from MIDI's 0–127 range into each uniform's `minimum` and `maximum`
range from `library.json`. Supported ACMXVK-equivalent actions include shader
selection, bypass, playlist/pause, freeze, time stepping and speed, multipass,
audio sensitivity, autopilot, and the MXVK screenshot key.

The build shader directory now contains a small `library.json` and
`midi_slider.frag.spv` test library. The included nanoKONTROL2 example maps the
channel-1 CC 20 messages shown above to Slider 1:

```bash
./build/acmxvk/acmxvk \
    --graphic jared-ai.png \
    --shaders ./build/acmxvk/shaders \
    --midi-device 1 \
    --midi-map ./ACMXVK/midi-examples/nanokontrol2-cc20-slider1.midi_cfg \
    --midi-monitor \
    --enable-vsync
```

Moving CC 20 changes image brightness and prints both the raw message and the
mapped `slider1` value. To map a CC directly without a `.midi_cfg` file, repeat
`--midi-cc [channel:]CC=uniform`; an omitted channel matches all channels:

```bash
./build/acmxvk/acmxvk \
    --graphic jared-ai.png \
    --shaders ./build/acmxvk/shaders \
    --midi-device 1 \
    --midi-cc 1:20=slider1
```

Increment 6C matches ACMX2's paired-knob behavior. The most recent CC value is
held until the controller sends neutral value `64`; distance from `64` controls
the action rate, ranging from about once every 16 frames near center to every
frame at either extreme. The included CC21 example controls shader-time speed:

```bash
./build/acmxvk/acmxvk \
    --graphic jared-ai.png \
    --shaders ./build/acmxvk/shaders \
    --midi-device 1 \
    --midi-map ./ACMXVK/midi-examples/centered-cc21-time-speed.midi_cfg \
    --midi-monitor \
    --enable-vsync
```

This centered mode is intended for spring-return or relative knobs. Use a
Slider 1–4 mapping or `--midi-cc` for an absolute knob or fader. When a map is
loaded, ACMXVK reports how many mappings are active and how many refer to ACMX2
features that have not yet been ported; `--midi-monitor` also names individual
unavailable action-code pairs.

### CUDA filters

Increment 7A adds the first `acidcam-gpu` processing path. Configure a separate
CUDA build on Linux with an NVIDIA GPU:

```bash
source ~/vulkan.sh
cmake -S ACMXVK -B build/acmxvk-cuda \
    -DWITH_CUDA=ON \
    -DVALIDATION=ON \
    -DCMAKE_BUILD_TYPE=Debug
cmake --build build/acmxvk-cuda -j
```

Confirm support and inspect the available devices and filters:

```bash
./build/acmxvk-cuda/acmxvk --check-cuda
./build/acmxvk-cuda/acmxvk --list-cuda-devices
./build/acmxvk-cuda/acmxvk --list-filters
```

Apply one filter to a video before its Vulkan shader pipeline:

```bash
./build/acmxvk-cuda/acmxvk \
    --input input.mp4 \
    --gpu-filter 3 \
    --gpu-buffer 10 \
    --cuda-device 0 \
    --fragment ./build/acmxvk-cuda/shaders/audio_reactive.frag.spv \
    --enable-vsync
```

`--gpu-filter` accepts a comma-separated chain such as `1,7,23`. Filter indices
must be in the range printed by `--list-filters`, and `--gpu-buffer` accepts
4–32 temporal frames. Video and camera inputs are filtered for every captured
frame. A still image is filtered when its source texture is initialized and
reprocessed immediately when Left/Right changes the active filter.

The filter engine uploads source RGBA into acidcam-gpu's temporal CUDA buffer,
runs the selected chain, and gives MXVK the resulting CUDA `GpuMat`; MXVK then
copies device-to-device into its Vulkan image. There is no post-filter CPU
download. Vulkan fragment and multipass shaders operate on the filtered image.
In Increment 7B, Left/Right selects the previous or next filter with wraparound.
This matches ACMX2: selecting a filter at runtime replaces a startup filter
chain with that single filter. MIDI Map actions 262 (Right) and 263 (Left) drive
the same selection when MIDI support and a GPU filter are enabled.

Increment 7C writes the filtered CUDA output into the Vulkan frame-history
array, including its initial fill, delayed ring-buffer updates, and still-image
filter changes. MXVK imports the layered Vulkan image into CUDA and copies the
selected frame device-to-device. If layered external-memory import is not
available on a driver, ACMXVK reports it once and falls back to host staging for
history only; the primary sprite upload remains device-to-device. On a
multi-GPU system, select the CUDA device corresponding to the Vulkan GPU. CUDA
filters are unavailable in MoltenVK builds.

Increment 7D removes the CPU RGBA handoff for unrotated video and camera input.
MXVK returns a resident CUDA `GpuMat`, acidcam-gpu copies it device-to-device
into its temporal buffer, and the filtered result continues directly into the
Vulkan sprite and optional history array. This path is selected automatically
when `--gpu-filter` is active.

Increment 7E keeps `--rotate clockwise`, `--rotate 180`, and
`--rotate counterclockwise` on that resident path. Exact CUDA transpose and
flip operations run on MXVK's capture stream before acidcam-gpu consumes the
frame, avoiding the former CPU rotation and upload. Still graphics continue to
begin from their host image because they are decoded only once.

## Runtime controls

- Up/Down: change the shader or playlist node
- Shift+Up/Down: change the final shader while using a playlist
- Left/Right: select the previous or next CUDA filter
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
spectrum history. Increments 7B through 7E were additionally tested with a
CUDA+MIDI build, live Left/Right filter changes, filtered Vulkan frame history,
resident `GpuMat` video input, and CUDA-resident clockwise, 180-degree, and
counterclockwise rotation on an NVIDIA RTX 2070. The known duplicate vkBasalt
implicit-layer warning is external to ACMXVK.

## Development note

I have been using the **Codex CLI from OpenAI** as an
engineering aid while porting ACMX2 to MXVK. Codex has assisted with incremental
code translation, CMake integration, shader conversion, debugging, Vulkan
validation testing, and documentation. Project direction, testing decisions,
and maintenance remain under the project owner's control.
