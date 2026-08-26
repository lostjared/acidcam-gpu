# ACMXVK

ACMXVK is an in-progress Vulkan port of the ACMX2 real-time video shader
engine. The goal is to preserve ACMX2's workflow and behavior while replacing
the MX2/OpenGL rendering path with the installed
[MXVK](https://github.com/lostjared/MXVK) engine and Vulkan SPIR-V shaders.

The port is currently at **Increment 8L**. It is usable for video, camera, and
still-image shader processing, but it is not yet a complete replacement for
ACMX2.

## Translation progress

| Area | Status | Notes |
| --- | --- | --- |
| Standalone CMake project | Complete | Builds against installed MXVK and MXWrite and provides an `uninstall` target. |
| Runtime resource paths | Implemented | ACMX2-compatible `-p/--path`, `ACMXVK_PATH`, and build/install fallbacks resolve data, internal shaders, shader libraries, playlists, and MIDI examples. `ACMXVK_SHADER_PATH` supplies a default SPIR-V library. |
| Window and Vulkan lifecycle | Complete | MXVK owns the window, device, swapchain, rendering, screenshots, and validation integration. |
| Video, camera, and image input | Complete | Video files prefer MXVK's FFmpeg capture with CUDA/NVDEC when available and fall back to OpenCV; an MXVK CUDA installation sends NVDEC frames directly to Vulkan independently of the acidcam-gpu build option. `--use-source-fps` provides real-time effects playback on the source-reported video clock, waiting when early and skipping decode work when late. Camera devices use ACMX2-compatible resolution, pixel-format, buffer, and FPS negotiation and report both the negotiated mode and measured delivery rate. `--maximize-fps` decouples camera acquisition from Vulkan presentation. When `--resolution` is omitted, encoded output follows the negotiated source dimensions, including a width/height swap for 90-degree input rotation; oversized previews are fitted to the usable display while preserving that aspect ratio. Still images remain OpenCV-backed. |
| Basic shader playback | Complete | Loads Vulkan fragment or compute shaders compiled to `.spv`; `--fragment` and `--compute` validate the SPIR-V stage. |
| Shader libraries | Complete | Prefers `library.json` and falls back to `index.txt`; supports nested paths and object or string entries. |
| Shader selection | Complete | Supports selection by index or filename and keyboard switching. |
| Multipass and playlists | Implemented | Includes named playlist nodes, mixed fragment/compute chains, sequential autopilot, and random autopilot. Shader stages are detected from SPIR-V entry points rather than filenames. |
| Frame history/texture cache | Implemented | Uses one shared Vulkan `sampler2DArray` ring buffer with configurable size and write delay. Fragment and compute post-processing passes can sample it at binding 2, and SPIR-V reflection enables it automatically for history-capable libraries. CUDA-filter builds place the post-filter image in history through direct CUDA/Vulkan layered-image interop. |
| Custom library uniforms | Implemented | Up to 64 validated floats from `library.json`, with repeatable `--uniform name=value` overrides. |
| Video recording | Implemented | MXWrite supports software or hardware encoders, encoder options, no-drop mode, duration and size limits, optional audio copying, source-timeline PTS, audio-clock synchronization, and pipelined Vulkan readback. |
| PNG output | Implemented | Supports full PNG sequences, periodic generated frames, and ACMX2-compatible one-shot `Z` snapshots with a configurable destination. |
| Text overlays and watermark | Implemented | Provides an ACMX2-compatible preview HUD with shader, multipass chain, decoded video position/source duration, processing elapsed time, measured FPS, audio track, CUDA filter, and autopilot status. The native title bar identifies graphics, video, or capture mode; distinguishes preview from recording; and reports recording time, frame count, and current encoded file size. Slow video processing advances the video timer by decoded frames rather than wall time. `--disable-counter` or F9 hides the HUD. The HUD and title are excluded from readback, snapshots, and recordings; explicit filter/watermark overlays remain included in output. |
| Rotation and final-output flip | Implemented | Applies input rotation and optional final display/recording flip. |
| Runtime playback controls | Implemented | Supports video pause, rendering freeze, shader locking, wall-clock or audio-reactive shader time, time stepping/speed, and fullscreen switching. |
| Input validation | Implemented | Centralized allowlists validate CLI and environment strings, paths, URLs, identifiers, encoder fields, manifests, playlists, MIDI maps, device names, and bounded live MIDI messages before use. Configuration files, lines, entry counts, numeric ranges, image dimensions, and SPIR-V binaries have explicit limits. |
| ACMX2 GLSL compatibility | Partial | Existing GLSL effects must be translated to the MXVK Vulkan descriptor ABI and compiled to SPIR-V. |
| Audio-reactive shader data | Implemented | RtAudio capture, an FFmpeg-decoded media file, an M3U/M3U8 playlist, or `--use-source-audio` with real-time source-FPS video playback can drive amplitude, frequency, peak, RMS, smoothed amplitude, low/mid/high bands, a current-frame FFT, configurable FFT history, audio-reactive shader time, and optional delta/sensitivity scaling. Source-video analysis follows the media clock even when late video frames are skipped. Live and file audio support configurable shader warmup, adjustable-gain output pass-through, and AAC muxing; live input can also be recorded independently as PCM16 WAV with adjustable gain, while file audio supports repeat and stop-at-EOF behavior. |
| MIDI controls | Partial | Optional RtMidi support handles input enumeration, a bounded callback queue, live monitoring, ACMX2 MIDI Map `.midi_cfg` files, Slider 1–4 custom uniforms, ACMXVK-equivalent playback actions, snapshots, watermark toggling, and the audio-time/delta/FFT sensitivity actions. Paired knobs use ACMX2's centered, velocity-sensitive repeat behavior. |
| CUDA filters | Partial | Optional `acidcam-gpu` integration accepts filter chains and temporal-buffer sizes, keeps NVDEC video frames, camera RGBA, and input rotation resident on the GPU through filtering and Vulkan upload/history, and supports ACMX2-compatible Left/Right selection from the keyboard or MIDI maps. |
| DNN effects | Not yet ported | OpenCV DNN segmentation, edge detection, and generic ONNX processing remain outside the current increment. |
| 3D model pipeline | Initial support | `--enable-3d` maps live video, camera, or still-image input onto MXVK's OBJ/MXMOD model renderer. Compatible fragments execute directly on model UVs; compute, history/spectrum, multipass, and playlist chains use a pre-model offscreen target whose result becomes the model texture. The camera starts at the normalized model center as a 120-degree skybox view with automatic rotation disabled. OBJ, MXMOD, and compressed MXMOD files are supported, with a bundled textured cube as the default. Mouse look/movement, automatic rotation, scale/speed controls, 2D/3D switching, recording, snapshots, and compatible MIDI-map actions are implemented. ACMX2's wave deformation remains to be ported. |
| Qt interface integration | Not yet ported | ACMXVK currently provides the command-line renderer only. |

## Requirements

- A C++20 compiler and CMake 3.20 or newer
- Vulkan SDK 1.4 with `glslc`
- MXVK 0.33.0 or newer, built with `-DVALIDATION=ON -DCV=ON`
- MXWrite from the MXVK source tree
- SDL3, SDL3_ttf, Vulkan, OpenCV, PNG, ZLIB, glm, and FFmpeg development files
- Optional SDL3_mixer, JPEG, and CUDA dependencies when enabled by the installed MXVK package
- Optional RtAudio development files when building with `-DAUDIO=ON`
- Optional RtMidi development files when building with `-DMIDI=ON`
- Optional CUDA Toolkit, CUDA-enabled OpenCV and MXVK, and an installed
  `acidcam-gpu` CMake package when building with `-DWITH_CUDA=ON`

Ensure the selected Vulkan SDK's `bin` directory is on `PATH` so CMake can
find tools such as `glslc`. If the SDK is installed outside the platform's
standard search paths, set `VULKAN_SDK` or add its prefix to
`CMAKE_PREFIX_PATH` using the setup instructions supplied with that SDK.

## Build

From the `acidcam-gpu` repository root:

```bash
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
acidcam-gpu needs another reinstall. Increment 7F also changes only ACMXVK and
uses MXVK's existing FFmpeg-capture API when that feature is present. Increment
7G changes only ACMXVK as well. Increment 7H also changes only ACMXVK and does
not require either dependency to be reinstalled. Increment 7I adds MXVK's
FFmpeg capture device-selection API, so MXVK must be rebuilt and reinstalled
before ACMXVK is rebuilt; acidcam-gpu does not need another reinstall.
Increment 7J adds MXVK's in-place FFmpeg seek API, so MXVK must again be rebuilt
and reinstalled before rebuilding ACMXVK; acidcam-gpu remains unchanged.
Increment 7K changes MXVK's internal NVDEC surface synchronization, so MXVK
must be rebuilt and reinstalled once more; acidcam-gpu remains unchanged.
Increment 7L changes only ACMXVK and does not require either dependency to be
rebuilt or reinstalled. Increments 7M through 7P also change only ACMXVK.
Increment 7Q adds MXVK's efficient capture-frame skip APIs and raises MXVK to
0.26.0, so MXVK must be rebuilt and reinstalled before rebuilding ACMXVK;
acidcam-gpu remains unchanged. Increment 7R changes only ACMXVK and does not
require MXVK or acidcam-gpu to be rebuilt. Increment 7S adds MXVK's
preview-only text queue and raises MXVK to 0.27.0, so MXVK must be rebuilt and
reinstalled before ACMXVK; acidcam-gpu remains unchanged. Increment 7T changes
only ACMXVK and does not require MXVK or acidcam-gpu to be rebuilt. Increment
7U also changes only ACMXVK. Increment 7V pipelines rendered-frame readback and
raises MXVK to 0.28.0, so MXVK must be rebuilt and reinstalled before ACMXVK;
acidcam-gpu remains unchanged. Increment 7W fixes discrete-GPU readback memory
selection and moves hardware-encoder upload work off the render thread. It
raises MXVK to 0.28.1, so MXVK must be rebuilt and reinstalled before ACMXVK.
Increment 7X adds compute post-processing pipelines and raises MXVK to 0.29.0,
so MXVK must again be rebuilt and reinstalled before ACMXVK; acidcam-gpu is
unchanged.
Increment 7Y changes only ACMXVK. It selects the video, negotiated camera, or
still-image dimensions when `--resolution` is omitted and applies input
rotation before choosing the automatic width and height. Neither MXVK nor
acidcam-gpu needs to be rebuilt or reinstalled.
Increment 7Z changes only ACMXVK. It prevents the desktop compositor from
maximizing an oversized source window to a different aspect ratio by fitting
the preview within the usable display and locking its aspect. Recording and
generated output retain the automatic source dimensions. Neither dependency
needs to be rebuilt or reinstalled.
Increment 8A adds MXVK 0.30.0's fixed native render extent. Shader passes,
watermark/output text, snapshots, and recording readback remain at the source
dimensions while a separate aspect-preserving copy is fitted into the preview
swapchain. Output text and preview-only HUD fonts are sized independently.
MXVK must be rebuilt and reinstalled; acidcam-gpu remains unchanged.
Increment 8B changes only ACMXVK. All overlay and HUD font sizing now follows
the preview window height at a smaller 1/60 scale instead of using the source
image or video height. MXVK does not need another rebuild or reinstall.
Increment 8C adds MXVK 0.31.1's non-owning shared history descriptor for
fragment and compute post-processing passes. ACMXVK reflects binding 2 from
SPIR-V, enables one input-frame history ring automatically, and supplies its
live head/layer metadata to every pass. Converted history shaders are now
included in `library.json`. Bindings 3 and 4 are reflected as well, allowing
spectrum shaders to receive zero-initialized safe descriptors even without
explicit audio-buffer flags. MXVK must be rebuilt and reinstalled;
acidcam-gpu remains unchanged.
Increment 8D changes only ACMXVK. `--use-source-audio` selects the input
video's embedded audio track for shader reactivity during `--use-source-fps`
playback. Optional pass-through uses the selected RtAudio output as the A/V
master clock; silent analysis follows the video media clock directly. Neither
MXVK nor acidcam-gpu needs to be rebuilt or reinstalled.
Increment 8E also changes only ACMXVK. A video without a decodable audio track
now continues with zero-valued audio-reactive inputs instead of aborting;
requested pass-through is disabled with a warning. An explicitly selected
`--audio-file` remains a hard error when it cannot be decoded.
Increment 8F changes only ACMXVK. Its native title bar now follows ACMX2's
graphics/video/capture mode format, explicitly reports preview or recording
state, and refreshes recording time, frame count, and MXWrite's encoded file
size twice per second. PNG sequences report recording state without a
single-file size.
Increment 8G changes only ACMXVK. Shader bypass now retains a compiled identity
post-processing pass instead of removing MXVK's source-sized offscreen chain.
This keeps its Vulkan image barriers and presentation target valid when Space
disables the selected shader, fixing the MoltenVK crash in video and camera
modes. The identity pass is also used when no visual shader is active.
Increment 8H changes only ACMXVK. `K` now toggles ACMX2-compatible shader
locking. While locked, manual shader and playlist-node selection and both
autopilot modes retain the current pipeline; autopilot resumes its existing
countdown after unlocking. The runtime HUD marks the shader as locked, and
ACMX2 MIDI-map action code 75 can toggle the same state.
Increment 8I changes only ACMXVK. It uses MXVK's `VKAbstractModel` and
`MXModel` loader to render OBJ, MXMOD, and compressed MXMOD models. Each input
frame replaces texture slot zero across the complete model draw, and the
result enters the same ordered fragment/compute pipeline used by 2D input.
The bundled `cube.obj` is selected when `--model` is omitted. MXVK 0.31.1
already provides the required public model API, so it does not need to be
rebuilt or reinstalled for this increment.
Increment 8J changes the 3D render order and camera defaults to match ACMX2.
The active compatible fragment shader is now the model fragment shader, so
the effect evaluates against the source texture in model UV space. The camera
starts at the normalized object center with yaw 270, pitch 0, distance 0, a
120-degree field of view, and automatic view rotation disabled. MXVK's model
fragment UBO now includes ACMXVK's 64 custom-uniform slots and audio-band
fields, so MXVK must be rebuilt and reinstalled for this increment.
Increment 8K adds MXVK 0.33.0's post-processing texture-consumer stage.
Compute shaders, history/spectrum shaders, `--shader-pass`, and playlist
chains now finish in an offscreen image that is sampled by the model. They no
longer appear as a flat screen-space layer over the completed 3D object.
Increment 8L changes only ACMXVK. Its 3D camera now accepts ACMX2's continuous
W/A/S/D look controls, plus/minus movement along the view direction, and 1/2
movement-sensitivity controls. MXVK and acidcam-gpu do not need to be rebuilt
or reinstalled.

### Input validation

Increment 7R routes user-controlled strings through one validation module.
Paths and visible labels accept well-formed printable UTF-8, including spaces
and international filenames, while rejecting embedded controls, malformed
encoding, bidirectional overrides, noncharacters, and oversized values.
Identifiers, encoder tokens, MIDI expressions, uniform overrides, and FFmpeg
option strings use narrower ASCII allowlists appropriate to their grammar.
Playlist URLs are restricted to `http`, `https`, `file`, `rtsp`, and `rtmp`
with basic authority and percent-escape validation.

Text configuration files are limited to 4 MiB; line-oriented index, playlist,
and MIDI-map formats also have 4096-byte line limits. Shader, playlist,
audio-playlist, and MIDI-map entry counts are bounded. User SPIR-V
files must have a valid aligned size and SPIR-V magic word before Vulkan loads
them. Command-line dimensions, frame rates, device indices, buffer counts,
durations, colors, and other allocation-sensitive numbers also have explicit
ranges. Build the regression test with the default `BUILD_TESTING=ON` and run:

```bash
ctest --test-dir build/acmxvk --output-on-failure
```

### Runtime resource paths

`-p/--path` accepts an ACMXVK resource root with this layout:

```text
resource-root/
├── data/font.ttf
├── shaders/library.json
├── shaders/*.spv
├── playlists/*.txt
└── midi-examples/*.midi_cfg
```

When none of `--shaders`, `--fragment`, or `--compute` is supplied, ACMXVK automatically
uses `shaders/library.json` (or `index.txt`) from the selected resource root.
Relative playlist and MIDI-map names are also searched beneath their matching
resource subdirectories. Media input and output arguments remain relative to
the current working directory.

Resource precedence is:

1. Explicit `-p/--path`.
2. `ACMXVK_PATH`.
3. `ACMX2_PATH` as a compatibility fallback for shared data such as fonts.
4. Installed, build-tree, and current-working-directory resources.

An explicit `--shaders`, `--fragment`, or `--compute` always wins. Otherwise,
`ACMXVK_SHADER_PATH` can name a SPIR-V library directory or its `library.json`
or `index.txt` file. `ACMX2_SHADER_PATH` is intentionally not consumed because
ACMX2 libraries contain OpenGL GLSL rather than MXVK SPIR-V.

For example:

```bash
./build/acmxvk/acmxvk \
    --path ./build/acmxvk \
    --graphic image.png \
    --shader-file midi_slider.frag.spv
```

The equivalent environment setup is:

```bash
export ACMXVK_PATH=/usr/local/share/acmxvk
export ACMXVK_SHADER_PATH=/path/to/spv-library
```

### Apple Silicon and MoltenVK

On an Apple Silicon Mac, configure the Vulkan SDK environment according to its
installation instructions, then configure an arm64 build with MoltenVK enabled:

```bash
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

Play a video with effects at its reported source rate instead of processing it
as quickly as possible:

```bash
./build/acmxvk/acmxvk \
    --input input.mp4 \
    --shaders /path/to/spv-library \
    --shader-file effect.spv \
    --use-source-fps \
    --enable-vsync
```

This is a real-time playback mode. Frames wait when processing is ahead of the
source clock. If an effect cannot render fast enough, ACMXVK skips late source
frames so the displayed video position does not drift into slow motion. The
mode cannot be combined with `--fps`, because the reported source rate is the
requested clock. `P` pause and `L` freeze suspend this clock and resume without
jumping over the paused interval. Add `--use-source-audio` with an `AUDIO=ON`
build to use the video's embedded audio track for shader reactivity. Add
`--pass-through` to hear it; `--audio-output <index>` selects the output device,
which then becomes the more accurate A/V master clock. If the video has no
decodable audio track, ACMXVK warns once, continues with zero-valued audio
inputs, and disables pass-through.

```bash
./build/acmxvk/acmxvk \
    --input input.mp4 \
    --fragment ./build/acmxvk/shaders/audio_reactive.frag.spv \
    --use-source-fps \
    --use-source-audio \
    --pass-through \
    --audio-output default
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

Take a processed PNG snapshot by pressing `Z`. The directory is created on the
first request:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/custom_uniform.frag.spv \
    --prefix ./snapshots
```

Snapshots include the timestamp, rendered resolution, and a collision-safe
sequence number in their filename. Readback is enabled for only the requested
frame unless video or PNG output already requires continuous readback. PNG
compression and disk writes run on a bounded background queue so the render
loop can continue while the snapshot is saved; queued work is drained during
shutdown.

Show the active shader/filter information with a yellow watermark. Press `E`
to toggle only the watermark while the filter information remains visible:

```bash
./build/acmxvk/acmxvk \
    --graphic image.png \
    --fragment ./build/acmxvk/shaders/custom_uniform.frag.spv \
    --display-filter \
    --use-watermark "My Channel" \
    --use-watermark-color 255,255,0
```

The source overlay font defaults to `ACMX2/data/font.ttf`, is copied into the
build resource tree, and is installed with ACMXVK. It can be replaced at
configure time with `-DACMXVK_OVERLAY_FONT=/path/to/font.ttf`.

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
        "compute/block_pixelate.comp.spv",
        { "file": "history/echo.spv" }
    ]
}
```

When `library.json` is absent, `index.txt` is read as one relative `.spv` path
per line. Absolute paths, parent-directory traversal, files outside the library,
and non-SPIR-V entries are rejected.

Each entry can contain either a fragment or compute entry point. ACMXVK reads
the execution model and compute local size directly from SPIR-V, so mixed
chains work through `--shader-pass`, `--shader-pass-files`, and playlists
without stage metadata in the manifest. Vertex and unsupported shader stages
are rejected before a pipeline is attached.

## Writing fragment and compute shaders

The repository includes `scripts/convert_shaders_to_acmxvk.pl` for bulk
conversion of ACMX2/OpenGL shader libraries. By default it reads fragment
shaders from `shaders_new`, compute shaders from `compute`, and writes a
separate `shaders_acmxvk` directory, leaving every original file untouched:

```bash
perl scripts/convert_shaders_to_acmxvk.pl
```

The output contains Vulkan-compatible `.frag` and `.comp` source files,
compiled `.spv` modules, `library.json`, and `conversion-report.txt`. Only
successfully compiled shaders are added to the manifest. Re-run with `--force`
to replace previously generated outputs, or use `--input`, `--output`,
`--compute-input`, and `--glslc` to select other locations. Converted compute
sources and modules are placed beneath `shaders_acmxvk/compute`. `--dry-run`
lists candidates without writing anything. The converter reports history-cache shaders separately
and includes them in the manifest. ACMXVK detects their binding 2 directly
from SPIR-V and enables the shared input-frame history cache automatically.

ACMXVK shaders are full-frame post-processing stages. The application renders
the input image, camera, or video first, then executes the configured pass list
in order. Each pass samples the preceding result. A fragment or compute shader
can be the first, middle, or final pass, and a chain can mix both stages. MXVK
reads the stage and compute workgroup size from the SPIR-V module; the filename
suffix is only a naming convention.

ACMXVK supplies `sprite.vert.spv` for every fragment pass. A shader library
therefore contains fragment and compute modules, not user vertex modules. The
first configured pre-pass runs first, the shader selected by `--shader-file`
runs after the pre-passes, and the internal vertical-flip pass runs last when
`--flip` is enabled.

### Image size, coordinates, color, and pass order

Every user pass operates at the current render/swapchain resolution, not
necessarily at the camera or source-file resolution. The output dimensions are
available as `ext.u0.zw`. The sampled input at binding 0 and compute output at
binding 5 have those same dimensions.

Fragment shaders receive `tc` from ACMXVK's vertex shader. It is a normalized
texture coordinate, normally in the inclusive range from `0.0` through `1.0`.
`gl_FragCoord.xy` is also available when integer pixel coordinates are more
convenient. Compute shaders do not receive `tc`; derive it from
`gl_GlobalInvocationID.xy` and `imageSize(output_image)` as shown below.

The pass images use normalized RGBA8 color. Sampled values are floating-point
RGBA in the `0.0` through `1.0` range, and storage-image writes outside that
range are clamped by the RGBA8 target. Binding-0 sampling uses normalized
coordinates, linear filtering, clamp-to-edge addressing, and no mipmaps. A
full-screen fragment effect should normally write alpha `1.0`; fragment
pipelines enable alpha blending, and intermediate attachments do not preserve
a defined destination color for partial-alpha composition. Do not rely on an
alpha below `1.0` unless the particular pass has been designed and tested for
that behavior.

### Descriptor-set bindings

All resources use descriptor set 0. A shader should declare only resources it
actually needs because optional bindings are absent from the pipeline layout
when their feature is unavailable.

| Binding | GLSL declaration | Stages | Contents and availability |
| --- | --- | --- | --- |
| 0 | `uniform sampler2D input_image;` | Fragment and compute | Always present. The image produced by the preceding stage. The name may be `samp`, `input_image`, or any other valid identifier. |
| 1 | `uniform SpriteExtended { ... } ext;` | Fragment and compute | Always present. Per-frame state, mouse, timing, audio values, and custom library uniforms. |
| 2 | `uniform sampler2DArray history;` | Fragment and compute | Optional shared RGBA input-frame history ring. `--texture-cache` enables it explicitly without adding an effect; ACMXVK also enables it automatically when a directly loaded shader or any entry in `library.json` declares set 0, binding 2. Every pass in the active fragment/compute chain shares the same ring. Use `--history-test` only when you also want ACMXVK's built-in echo demonstration applied before the selected pipeline. |
| 3 | `uniform sampler1D spectrum;` | Fragment and compute | Current 256-bin R32 floating-point FFT. ACMXVK detects this binding from SPIR-V and supplies a zero-initialized descriptor when audio support or an active audio source is unavailable. |
| 4 | `uniform sampler1DArray spectrum_history;` | Fragment and compute | Circular FFT history. ACMXVK detects this binding from SPIR-V and automatically allocates eight zero-initialized layers when `--enable-audio-buffers N` was not supplied. |
| 5 | `layout(rgba8) writeonly uniform image2D output_image;` | Compute only | Compute destination. Always required by an ACMXVK compute shader and unavailable to fragment shaders. |

Binding numbers, descriptor types, array lengths, and block-member order are
part of the ABI and must match exactly. Resource and member names are not part
of the ABI and may be renamed.

### Complete `SpriteExtended` uniform block

Binding 1 uses `std140` layout. Declaring `std140` explicitly is recommended so
the offsets are obvious and stable:

```glsl
layout(set = 0, binding = 1, std140) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
    vec4 custom_uniforms[16];
    vec4 audio_bands;
    vec4 audio_history;
} ext;
```

Every entry is a `vec4` aligned to 16 bytes. A shader may stop the declaration
after the last field it uses. For example, a shader which needs only time and
resolution may stop after `u2`. It must not remove or reorder earlier fields:
declaring `audio_bands` immediately after `u3` would read custom-uniform memory,
not the audio bands.

The complete field map is:

| Field | Offset | Components | Value |
| --- | ---: | --- | --- |
| `mouse` | 0 | `.x`, `.y` | Mouse position in window pixels. SDL supplies a top-left-origin position. |
|  |  | `.z` | `1.0` while the left mouse button is held, otherwise `0.0`. |
|  |  | `.w` | Reserved; currently `0.0`. |
| `u0` | 16 | `.x` | ACMX2-compatible animated `alpha`, advanced by `0.1` per rendered frame and reflected between `1.0` and `6.0` after its initial `0.1` value. |
|  |  | `.y` | Wall-clock elapsed seconds for the ACMX2/Shadertoy-compatible `iTime`; unlike shader time, this does not pause, scale, or reset when the selected effect changes. |
|  |  | `.z`, `.w` | Render width and height in pixels. A convenient alias is `vec2 resolution = ext.u0.zw`. |
| `u1` | 32 | `.x` | Frame delta in seconds. With `--normalized`, this is exactly `1.0 / output_fps`; otherwise it is measured wall-clock delta. |
|  |  | `.y` | `amp`: processed mean audio amplitude after sensitivity, warmup, time-speed, and optional delta scaling. Zero without active audio. |
|  |  | `.z` | `iamp`: zero-crossing frequency estimate in Hz. |
|  |  | `.w` | Instantaneous render rate, calculated as `1.0 / u1.x`. |
| `u2` | 48 | `.x` | Rendered frame counter, stored as a float. It resets when shader time is reset or a new shader/playlist node is selected. |
|  |  | `.y` | Shader time in seconds. It follows `--time-speed`, normalized time, the `T/U/I/Page Up/Page Down` controls, and audio-reactive time when enabled. |
|  |  | `.z` | `iSampleRate`: active audio sample rate in Hz; the compatibility default is `44100.0`. |
|  |  | `.w` | `amp_peak`: sensitivity- and warmup-scaled peak audio level. |
| `u3` | 64 | `.x`, `.y` | Frame-history write head and layer count. The values are supplied to every fragment and compute pass when the shared input history cache is active. |
|  |  | `.z` | `amp_rms`: sensitivity- and warmup-scaled RMS audio level. |
|  |  | `.w` | `amp_smooth`: sensitivity- and warmup-scaled smoothed audio amplitude. |
| `custom_uniforms` | 80 | 16 `vec4`s | Up to 64 user floats from `library.json`, packed in declaration order. Unused slots are `0.0`. The array occupies byte offsets 80 through 335. |
| `audio_bands` | 336 | `.x`, `.y`, `.z` | `amp_low`, `amp_mid`, and `amp_high`: scaled energy below 300 Hz, from 300 through 3000 Hz, and above 3000 Hz. |
|  |  | `.w` | Reserved; currently `0.0`. |
| `audio_history` | 352 | `.x` | Physical array layer containing the newest FFT spectrum. |
|  |  | `.y` | Number of allocated FFT-history layers. |
|  |  | `.z` | Number of bins per layer, currently `256.0`. |
|  |  | `.w` | Reserved; currently `0.0`. |

The entire `audio_history` vector is zero when FFT history is not enabled.
The audio scalar fields remain safe to declare in any build because they are
part of binding 1; they simply remain zero when no audio source is active.
Bindings 3 and 4 are separate optional descriptors and follow the availability
rules in the descriptor table.

`mouse.xy` uses output-window pixels rather than normalized coordinates. A
typical conversion is
`vec2 mouse_uv = ext.mouse.xy / max(ext.u0.zw, vec2(1.0));`. Unlike the
Shadertoy `iMouse` convention, `mouse.z` is only a pressed/not-pressed value
and `mouse.w` does not contain a click position.

Useful aliases can make a port from an ACMX2 shader easier to read:

```glsl
#define iResolution ext.u0.zw
#define alpha ext.u0.x
#define iTime ext.u0.y
#define iTimeDelta ext.u1.x
#define amp ext.u1.y
#define iamp ext.u1.z
#define iFrameRate ext.u1.w
#define iFrame ext.u2.x
#define iSampleRate ext.u2.z
#define amp_peak ext.u2.w
#define amp_rms ext.u3.z
#define amp_smooth ext.u3.w
#define amp_low ext.audio_bands.x
#define amp_mid ext.audio_bands.y
#define amp_high ext.audio_bands.z
```

### Minimal fragment shader

A fragment shader needs one normalized input coordinate, one RGBA output, and
binding 0. Binding 1 is optional when none of its values are needed:

```glsl
#version 450

layout(location = 0) in vec2 tc;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D input_image;

layout(set = 0, binding = 1, std140) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
} ext;

void main() {
    vec2 resolution = max(ext.u0.zw, vec2(1.0));
    vec2 one_pixel = 1.0 / resolution;
    float wave = sin(ext.u2.y * 2.0 + tc.y * 20.0) * 4.0;
    vec2 sample_uv = clamp(tc + vec2(wave * one_pixel.x, 0.0),
                           vec2(0.0), vec2(1.0));
    color = vec4(texture(input_image, sample_uv).rgb, 1.0);
}
```

The fragment-stage names are conventional, but their locations and types are
fixed:

| Fragment value | Meaning |
| --- | --- |
| `layout(location = 0) in vec2 tc` | Interpolated normalized coordinate from ACMXVK's full-screen vertex shader. The variable may be renamed. |
| `layout(location = 0) out vec4 color` | RGBA result for this pixel. The variable may be renamed. |
| `gl_FragCoord.xy` | Pixel-center position in framebuffer coordinates. Useful for grids and effects which must align exactly to output pixels. |
| `texture(input_image, uv)` | Filtered sample using normalized coordinates. Coordinates are clamped at the image edges. |
| `texelFetch(input_image, pixel, 0)` | Exact unfiltered integer-pixel sample from mip level zero. Clamp `pixel` before fetching. |
| `textureSize(input_image, 0)` | Integer dimensions of the preceding pass image. They normally equal `ivec2(ext.u0.zw)`. |

Fragment shaders may also declare MXVK's 48-byte push-constant block. This
block is not available to compute shaders:

```glsl
layout(push_constant) uniform SpritePushConstants {
    float screen_width;
    float screen_height;
    float sprite_pos_x;
    float sprite_pos_y;
    float sprite_size_w;
    float sprite_size_h;
    float effects_on;
    float rotation_degrees;
    vec4 params;
} pc;
```

| Push-constant field | ACMXVK value for a full-frame user pass |
| --- | --- |
| `screen_width`, `screen_height` | Current render dimensions in pixels. |
| `sprite_pos_x`, `sprite_pos_y` | `0.0, 0.0`. |
| `sprite_size_w`, `sprite_size_h` | Current render dimensions in pixels. |
| `effects_on` | `1.0` while effects are active. Pressing Space bypasses the user pipeline entirely. |
| `rotation_degrees` | `0.0` for the full-screen post-process quad. |
| `params.xyz` | Compatibility constants, currently `1.0, 1.0, 1.0`. |
| `params.w` | Shader time in seconds, equivalent to `ext.u2.y`. |

`shaders/custom_uniform.frag`, `shaders/audio_reactive.frag`,
`shaders/audio_spectrum.frag`, and `shaders/audio_history.frag` demonstrate
progressively larger parts of the fragment ABI.

### Minimal compute shader

An ACMXVK compute shader samples binding 0 and writes binding 5. Compute
pipelines have no push-constant range, so use `SpriteExtended` for time,
resolution, mouse, audio, and custom values.

```glsl
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D input_image;

layout(set = 0, binding = 1, std140) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
    vec4 custom_uniforms[16];
    vec4 audio_bands;
    vec4 audio_history;
} ext;

layout(set = 0, binding = 5, rgba8) writeonly uniform image2D output_image;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_image);

    // MXVK rounds the workgroup count up, so edge workgroups contain
    // invocations outside the image.
    if (any(greaterThanEqual(pixel, size))) {
        return;
    }

    vec2 uv = (vec2(pixel) + vec2(0.5)) / vec2(size);
    vec4 source = texture(input_image, uv);
    float pulse = 0.5 + 0.5 * sin(ext.u2.y * 2.0);
    imageStore(output_image, pixel,
               vec4(source.rgb * mix(0.5, 1.5, pulse), source.a));
}
```

The standard compute built-ins are useful for effects which operate on tiles
or share data within a workgroup:

| Compute built-in | Meaning |
| --- | --- |
| `gl_GlobalInvocationID` | Absolute unsigned invocation coordinate. `.xy` is the output pixel in the usual one-invocation-per-pixel design. |
| `gl_WorkGroupID` | Workgroup coordinate within the dispatch. |
| `gl_LocalInvocationID` | 3-D coordinate of the invocation inside its workgroup. |
| `gl_LocalInvocationIndex` | Flattened one-dimensional index inside the workgroup. |
| `gl_NumWorkGroups` | Number of dispatched workgroups in each dimension. MXVK calculates `.xy` by rounding the output size up to the declared local size; `.z` is one. |
| `gl_WorkGroupSize` | Compile-time `uvec3` matching `local_size_x`, `local_size_y`, and `local_size_z`. |

Use `texture(input_image, uv)` for filtered sampling or
`texelFetch(input_image, pixel, 0)` for an exact input texel. Use
`imageSize(output_image)` for bounds. Because `output_image` is declared
`writeonly`, read the old color from `input_image`, not with `imageLoad`.

MXVK dispatches
`ceil(width / local_size_x)` by `ceil(height / local_size_y)` by one workgroup.
Always bounds-check `gl_GlobalInvocationID.xy`, declare
`local_size_z = 1`, and write every in-range pixel. Input and output are
different ping-pong images, so reading binding 0 and writing binding 5 never
aliases the same image. MXVK inserts the required barriers before the next
fragment or compute pass. If the final pass is compute, MXVK presents its
result with an internal full-screen copy.

Use literal `local_size_x`, `local_size_y`, and `local_size_z` values as in the
example. MXVK currently reads SPIR-V `LocalSize` metadata; specialization-ID
forms such as `local_size_x_id` are not supported for dispatch sizing.

`shaders/compute_test.comp` is the complete working reference. Workgroups of
`8x8` or `16x16` are sensible starting points; performance depends on the
shader and GPU.

### Custom variables from `library.json`

Custom uniforms are packed in manifest declaration order. Uniform number `N`
is stored in `ext.custom_uniforms[N / 4][N % 4]`. For example, the first five
entries map as follows:

| Declaration index | Shader location |
| ---: | --- |
| 0 | `ext.custom_uniforms[0].x` |
| 1 | `ext.custom_uniforms[0].y` |
| 2 | `ext.custom_uniforms[0].z` |
| 3 | `ext.custom_uniforms[0].w` |
| 4 | `ext.custom_uniforms[1].x` |

Given the earlier `square_size` manifest entry, the shader can use:

```glsl
#define square_size ext.custom_uniforms[0].x
```

The manifest's `minimum`, `maximum`, `step`, and `value` fields define the
accepted range, adjustment step, and initial value. Repeat
`--uniform name=value` to override initial values. ACMX2 MIDI Slider 1 through
Slider 4 target custom uniforms named `slider1` through `slider4`; their packed
positions depend on declaration order. Keep that order stable after compiling
a shader which uses fixed array positions. See `shaders/custom_uniform.frag`
and `shaders/midi_slider.frag`.

### FFT textures

In an `AUDIO=ON` build, binding 3 contains 256 non-negative FFT magnitudes in
an `R32_SFLOAT` 1-D texture:

```glsl
layout(set = 0, binding = 3) uniform sampler1D spectrum;

float normalized_frequency = 0.25;
float magnitude = texture(spectrum, normalized_frequency).r;
```

Coordinate `0.0` is the lowest-frequency/DC end and `1.0` is the Nyquist end.
The texture uses linear filtering, so coordinates between bins interpolate.
Squaring a normalized coordinate before sampling, as in
`audio_spectrum.frag`, devotes more screen space to lower frequencies.

With `--enable-audio-buffers N`, binding 4 stores previous FFTs in a circular
`sampler1DArray`. `ext.audio_history.x` identifies the newest physical layer
and `.y` gives the allocated count:

```glsl
layout(set = 0, binding = 4) uniform sampler1DArray spectrum_history;

int history_layer(int age, int count, int newest) {
    return (newest - (age % count) + count) % count;
}

int count = max(int(ext.audio_history.y + 0.5), 1);
int newest = clamp(int(ext.audio_history.x + 0.5), 0, count - 1);
int layer = history_layer(0, count, newest);
float newest_magnitude =
    texture(spectrum_history, vec2(normalized_frequency, float(layer))).r;
```

Age zero is newest, age one is the preceding FFT, and so on. The requested
history depth may be clamped to the GPU's maximum image-array layer count.

### Compile, validate, and load a shader

Source the Vulkan SDK used by the project, compile GLSL to SPIR-V, and validate
the result before adding it to `library.json`:

```bash
glslc my_effect.frag -o my_effect.frag.spv
glslc my_effect.comp -o my_effect.comp.spv

spirv-val my_effect.frag.spv
spirv-val my_effect.comp.spv
```

For a filename without a recognized stage suffix, specify it explicitly with
`glslc -fshader-stage=fragment` or `glslc -fshader-stage=compute`. Test a single
module with `--fragment` or `--compute`; use `--shaders` after adding it to a
manifest. Validation builds report descriptor-layout, storage-image, and
synchronization mistakes at runtime.

Common shader problems are:

- omitting `set = 0` or using a binding with the wrong descriptor type;
- reordering or shortening `SpriteExtended` before a field that the shader reads;
- declaring binding 3 or 4 with a descriptor type other than the documented sampler type;
- declaring binding 2 with a type other than `sampler2DArray`;
- declaring fragment push constants in a compute shader;
- using a non-`main` entry point or specialization-ID compute local sizes;
- forgetting the compute edge bounds check or failing to write every valid output pixel;
- using a compute storage-image format other than `rgba8`;
- relying on fragment alpha below `1.0` even though the pass destination is not preserved for partial-alpha composition;
- adding a vertex module or unsupported SPIR-V stage to `library.json`.

Test the standalone compute path after reinstalling MXVK 0.29.0:

```bash
./build/acmxvk/acmxvk \
    --graphic acmx-vk/jared-ai.png \
    --compute ./build/acmxvk/shaders/compute_test.comp.spv \
    --resolution 1280x720 \
    --enable-vsync
```

The generated test library already contains the compute pass and MIDI-slider
fragment pass. Select compute as the pre-pass and fragment as the active shader:

```bash
./build/acmxvk/acmxvk \
    --graphic acmx-vk/jared-ai.png \
    --shaders ./build/acmxvk/shaders \
    --shader-pass-files 21:compute_test.comp.spv \
    --shader-file midi_slider.frag.spv \
    --resolution 1280x720 \
    --enable-vsync
```

With an `AUDIO=ON` build, `--enable-audio` uses live RtAudio input while
`--audio-file <media>` uses the first audio stream decoded by FFmpeg. It also
accepts an M3U or M3U8 playlist. Both paths map the same audio metrics into the
binding-1 block:

| ACMX2 name | MXVK field | Meaning |
| --- | --- | --- |
| `amp` | `ext.u1.y` | Processed mean amplitude after sensitivity, warmup, time-speed, and optional delta scaling |
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

The ACMX2 audio-time controls are available while a live or file audio source
is active. Press `Q` to make audio amplitude advance shader time instead of the
wall clock. `Home` toggles frame-delta scaling for both reactive time and the
`amp` value, which makes their behavior less dependent on rendering frame rate.
`End` toggles sensitivity scaling for the current FFT and FFT-history textures.
The same actions are accepted from ACMX2 MIDI Map codes 81, 268, and 269.

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

For video-file input with a reactive audio track, the recorded frame PTS now
follow the source-video timeline. Encoder queue drops therefore leave timestamp
gaps instead of shortening the result. With `--pass-through`, the audio output
device becomes the master clock: ACMXVK waits when video is early, efficiently
skips decoded frames when video is late, and submits the displayed frame at the
matching timeline PTS. Without pass-through, file-audio analysis advances by
exactly one nominal video frame and produces the same offline alignment.

```bash
./build/acmxvk/acmxvk \
    --input input.mp4 \
    --fragment ./build/acmxvk/shaders/audio_reactive.frag.spv \
    --audio-file soundtrack.mp3 \
    --pass-through \
    --output synchronized-output.mp4 \
    --enable-vsync
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

Record live microphone input to a standalone mono PCM16 WAV while processing a
video. `--record-audio` enables live audio capture automatically, and
`--record-gain` is applied only to the saved or muxed microphone samples:

```bash
./build/acmxvk/acmxvk \
    --input clip.mp4 \
    --fragment ./build/acmxvk/shaders/audio_reactive.frag.spv \
    --audio-input default \
    --record-audio microphone.wav \
    --record-gain 1.25 \
    --enable-vsync
```

The WAV closes when the video ends or ACMXVK exits. It can be written alongside
an encoded `--output`; in that mode the same captured samples can also be muxed
into the video. Standalone recording uses live input and therefore cannot be
combined with `--audio-file`.

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

See `shaders/audio_spectrum.frag` for a complete visualization. MXVK's extended
ABI permits the descriptor alongside binding-2 frame history. ACMXVK shares
the input sprite's single history array with every user post-processing pass,
avoiding a duplicate ring allocation per pass.

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

Increment 7F prefers MXVK's FFmpeg capture for video files. On a CUDA-enabled
MXVK build and a supported codec, FFmpeg decodes with NVDEC and supplies a
resident CUDA RGBA frame directly to the 7E rotation/filter/upload path. The
application reports the selected decoder at startup; software FFmpeg decoding
and OpenCV remain automatic fallbacks. Cameras continue through `VK_Capture`,
and `--repeat` reopens the same FFmpeg/NVDEC path at end-of-stream.

Increment 7G removes the acidcam-gpu-filter requirement from the resident
NVDEC path. In a `-DWITH_CUDA=ON` build, a supported video can now flow from
NVDEC through optional CUDA rotation directly into the Vulkan sprite and
history array even when `--gpu-filter` is omitted. If direct CUDA/Vulkan interop
is unavailable, ACMXVK reports the condition once and uses host staging. The
7F filtered route remains unchanged.

Increment 7H separates MXVK CUDA interop from ACMXVK's `WITH_CUDA` filter
option. When the installed MXVK package was built with CUDA, an ordinary
`-DWITH_CUDA=OFF` ACMXVK build now includes resident NVDEC upload, rotation, and
history without locating or linking acidcam-gpu. Configure `-DWITH_CUDA=ON`
only when the acidcam-gpu filter library is wanted. MoltenVK and non-CUDA MXVK
installations continue compiling the portable host path.

Increment 7I completes that separation for the CUDA command-line controls.
`--list-cuda-devices` and `--cuda-device` now work whenever the installed MXVK
package has CUDA interop, including `-DWITH_CUDA=OFF` ACMXVK builds. The selected
device is applied before rendering and forwarded to MXVK's FFmpeg capture, so
its NVDEC context and optional acidcam-gpu filters use the same CUDA device.
`--check-cuda` reports MXVK interop and acidcam-gpu filter support separately,
and capture startup identifies the active NVDEC device.

Increment 7J makes repeated FFmpeg input seek back to the first video timestamp
without destroying the decoder. `--repeat` therefore preserves the active
NVDEC device and CUDA hardware context across loops and avoids repeatedly
allocating and reopening the decoder. If an input is not seekable, ACMXVK
retains the previous close-and-reopen behavior as an automatic fallback.

Increment 7K protects each FFmpeg-owned NVDEC surface until MXVK's asynchronous
device-to-device plane copies have completed. A CUDA event marks that ownership
boundary while later color conversion, rotation, filtering, and Vulkan upload
remain queued asynchronously. This prevents FFmpeg from recycling a decode
surface while MXVK still reads it without introducing a full capture-stream
synchronization. MXVK reports the active barrier once per capture session.

Increment 7L ports ACMX2's standalone `--record-audio <wav>` workflow. Live
microphone samples are buffered by the existing real-time callback, converted
to mono PCM16 only after capture stops, and written with an explicit
little-endian RIFF/WAVE header. Recording can run independently or alongside
video encoding and continues to honor `--record-gain`.

Increment 7M ports ACMX2's remaining audio-runtime controls. `Q` selects
amplitude-driven shader time, `Home` applies frame-delta scaling to reactive
time and amplitude, and `End` applies the current audio sensitivity to FFT and
FFT-history samples. ACMX2 MIDI Map actions for all three controls use the same
event path as the keyboard.

Increment 7N ports ACMX2's `Z` PNG snapshot workflow and `-e/--prefix`
destination. A snapshot captures the final processed swapchain image, including
the configured shader passes and output flip. ACMX2 MIDI Map action 90 now
dispatches the same `Z` request. MXVK readback is enabled for one frame and then
disabled again when recording or periodic PNG generation is not active. A
four-job background queue keeps PNG compression and disk I/O off the render
thread and rejects excess requests instead of allowing unbounded frame-memory
growth.

Increment 7O ports ACMX2's recording overlays through MXVK's Vulkan text
renderer. `--display-filter` identifies the active shader, playlist node,
multipass stack, and CUDA filter chain. `--use-watermark` and
`--use-watermark-color` add configurable text immediately below that status,
and `E` or ACMX2 MIDI Map action 69 toggles the watermark at runtime. The final
MXVK compositing pass draws both overlays after the shader effect chain, keeping
the text crisp and unmodified while preserving it in one-shot PNG snapshots,
PNG sequences, periodic captures, and MXWrite video output.

Increment 7P ports ACMX2's runtime asset-root workflow. `-p/--path`,
`ACMXVK_PATH`, and the compatible `ACMX2_PATH` fallback now feed one ordered
resource resolver for fonts, internal shaders, default shader libraries,
playlists, and MIDI examples. `ACMXVK_SHADER_PATH` provides a SPIR-V-specific
shader default without accidentally consuming ACMX2's OpenGL shader tree.
CMake mirrors the installed `data/` and `shaders/` layout in the build tree.

Increment 7Q ports ACMX2's media-timeline synchronization and extends it to
live microphone muxing. Recording starts on the first valid source frame.
Video-file frames retain their nominal source PTS even when MXWrite's bounded
queue drops work, while audible file audio and muxed live input use their
hardware sample counters as the master clock. Late file-video frames are
decoded and discarded through MXVK without RGBA conversion, CUDA transfer, or
Vulkan upload. Readback is requested only when a new recording frame is due.

Increment 7R hardens every ACMXVK text-input boundary with the shared
`input_validation` module. It covers command-line arguments, relevant
environment variables, shader JSON/index entries, shader playlists, M3U audio
playlists, MIDI map files, custom-uniform and encoder expressions, device
labels, and overlay text. Bounded readers prevent oversized configuration
lines from allocating without limit, while count, numeric, decoded-image, live
MIDI-message, and SPIR-V checks protect the corresponding non-string inputs.

Increment 7S ports ACMX2's default runtime HUD. It shows the active shader,
elapsed time, measured presentation FPS, the current file-audio track, active
CUDA filter, and autopilot status. F9 toggles the HUD and `--disable-counter`
starts with it hidden. MXVK composites this preview-only queue after frame
readback, so it never appears in MXWrite video, PNG output, one-shot snapshots,
or F10 captures. Explicit `--display-filter` and watermark text continue to be
drawn before readback and remain part of saved output.

For video-file input the HUD now separates media position from processing
time. `Video: HH:MM:SS / HH:MM:SS` advances from the decoded source-frame
position and reports the container duration, while `Elapsed: HH:MM:SS` follows
the real processing clock. A high-resolution effect can therefore run slower
than real time without making the media-position counter run ahead. Camera and
still-image modes show only the elapsed processing timer. Repeating a video
resets the displayed source position after each in-place seek while recording
PTS remain continuous.

`--use-source-fps` turns that source timeline into a playback clock. ACMXVK
waits before decoding an early frame and uses MXVK's decode-only skip path to
catch up when rendering is late, avoiding unnecessary RGBA conversion and GPU
upload for discarded frames. Pause and rendering-freeze time are subtracted
from the playback clock. Without this option, video input retains ACMXVK's
offline behavior and processes frames as quickly as the machine permits.

Increment 7T aligns camera setup with ACMX2 by applying a one-frame capture
buffer, requested dimensions, MJPG (or `--use-yuv` YUYV), and the requested FPS
in a stable negotiation order. Startup output reports the dimensions, nominal
FPS, and FourCC read back from the driver and warns when they differ from the
request. The preview HUD labels render throughput separately and adds a camera
rate measured from successfully delivered frames. Significant delivery-rate
changes are also logged, making low-light automatic-exposure reductions such
as 60-to-30 FPS visible without mistaking the driver's nominal rate for actual
throughput.

Increment 7U adds `--maximize-fps` for camera input. It requires `--fps` and
runs camera acquisition on a bounded latest-frame worker while pacing the
Vulkan loop at the requested rate. If a camera supplies 30 FPS and the target
is 60 FPS, camera textures update when new frames arrive while shader time,
custom uniforms, overlays, and presentation continue at 60 FPS. Only one
unconsumed camera frame is retained, preventing latency and memory growth when
capture is faster than rendering. Recording follows the render clock in this
mode, so animated shader output can be encoded at the requested rate even when
adjacent frames use the same camera image. VSync and hardware load can still
cap the achieved presentation rate.

Increment 7V removes MXVK's same-frame recording fence wait. Each frame in
flight now has its own persistently mapped Vulkan readback buffer; ACMXVK
consumes a completed buffer when that slot's normal fence is reached and keeps
the original snapshot intent and recording PTS attached to the delayed frame.
Pending buffers are flushed before MXWrite closes, so the final submitted
frames are not lost.

Increment 7W makes the pipelined path practical on discrete GPUs. MXVK now
prefers host-cached coherent memory for readback instead of accepting the first
host-visible type, which can be an uncached PCIe mapping on NVIDIA hardware.
The repository MXWrite also queues host RGBA frames immediately and performs
conversion plus hardware-frame upload on its encoder thread. Devices without a
host-cached coherent type retain the portable coherent-memory fallback.

Increment 7X ports ACMX2-style compute image effects into the ordered Vulkan
pass chain. MXVK 0.29.0 reflects fragment versus compute entry points from
SPIR-V, uses synchronized RGBA8 storage-image ping-pong targets, and preserves
the existing binding-1 uniforms plus optional history and audio descriptors.
Fragment-only commands and manifests remain compatible.

Increment 7Y makes source-sized rendering the default. After a video or camera
is opened, ACMXVK uses the decoder or camera driver's negotiated dimensions to
size the window before MXVK creates its swapchain. Clockwise and
counterclockwise 90-degree input rotation swap those dimensions. A still image
uses its decoded dimensions in the same way. Passing `--resolution` keeps the
requested fixed window/output size; fullscreen presentation continues to use
the active display extent.

Increment 7Z separates automatic output geometry from the physical preview
window geometry. Sources that fit within 90 percent of the usable display keep
their native preview dimensions. Larger 16:9, 9:16, square, and rotated inputs
are uniformly scaled down and centered with a locked aspect ratio, preventing
the window manager from stretching them when they exceed the desktop. Video
and generated output still use the full source width and height. An explicit
`--resolution` continues to control both the window and output directly.

Increment 8A removes the remaining preview-resolution processing compromise.
MXVK renders the scene and every ordered fragment/compute pass into
source-sized offscreen images, reads the completed source-sized frame for
encoding, then scales only the presentation copy into the window with
letterboxing if necessary. Resizing or maximizing the preview therefore cannot
change shader resolution or encoded geometry.

Increment 8B sizes both output overlays and the preview-only runtime HUD from
the actual preview-window height. The 1/60 scale keeps long shader, multipass,
timer, and FPS lines readable when a high-resolution or portrait source is
fitted into a smaller window; the minimum remains 12 points.

## 3D model rendering

Enable the MXVK model renderer with `--enable-3d`. Without `--model`, ACMXVK
loads its bundled `models/cube.obj`. Supplying `--model` also enables 3D mode
and accepts Wavefront OBJ, MXMOD, or compressed MXMOD (`.mxmod.z`) input:

```bash
./build/acmxvk/acmxvk \
    --input video.mp4 \
    --use-source-fps \
    --model /path/to/model.mxmod.z \
    --fragment ./build/acmxvk/shaders/passthrough.frag.spv \
    --resolution 1280x720 \
    --enable-vsync
```

MXVK normalizes the model from its bounds and ACMXVK uploads each decoded
RGBA input frame into the model's primary Vulkan texture. The primary texture
is explicitly selected for the complete draw, so models containing multiple
submeshes receive the same live input across every surface. CUDA-enabled MXVK
builds use direct CUDA/Vulkan model-texture interop for NVDEC and filtered
frames, with host staging as a reported fallback.

For a compatible single fragment shader, ACMXVK installs that shader directly
on MXVK's model pipeline. The shader samples the live source at the mesh UVs,
which matches ACMX2's 3D path and avoids applying the effect as a flat
screen-space filter after rasterization. Compute shaders, history/spectrum
descriptor shaders, and active multipass/playlist chains use MXVK 0.33.0's
pre-model offscreen chain. Its completed image is then sampled at the model
UVs.
Output overlays, snapshots, and encoded video still operate on the completed
3D image. The preview-only HUD remains excluded from saved output.

The initial view is a skybox-style camera at the normalized model center. It
uses ACMX2's yaw 270 degrees, pitch 0, distance 0, and 120-degree field of
view. Automatic view rotation is disabled at startup.

The main 3D controls are:

- Left mouse drag: look around from the camera
- Mouse wheel: move backward or forward along the view direction
- `W` / `S`: look up or down
- `A` / `D`: look left or right
- `+` / `-`: move backward or forward along the view direction
- `1` / `2`: increase or decrease keyboard movement sensitivity
- `3`: switch between 3D model and 2D sprite rendering
- `V`: toggle automatic view rotation
- `X`: reset the centered skybox view and scale
- `[` / `]`: decrease or increase model scale
- `,` / `.`: decrease or increase automatic view-rotation speed

ACMX2 MIDI-map action codes 44, 46, 51, 86, 88, 91, and 93 drive the same
controls. Model paths are centrally validated, restricted to the supported
extensions, and limited to 1 GiB before reaching MXVK's loader.

## Runtime controls

- Up/Down: change the shader or playlist node
- Shift+Up/Down: change the final shader while using a playlist
- Left/Right: select the previous or next CUDA filter
- P: toggle playlist mode
- P without a playlist: pause or resume video input
- L: freeze or resume both input and shader animation
- T: enable or disable shader-time advancement
- Q: toggle audio-reactive shader-time advancement
- Home: toggle frame-delta scaling for reactive time and amplitude
- End: toggle sensitivity scaling for FFT and FFT-history data
- U/I: step shader time forward or backward by 0.05
- Page Up/Page Down: increase or decrease shader-time speed
- Insert/Delete: increase or decrease live audio sensitivity
- F: toggle fullscreen
- F9: toggle the preview-only runtime HUD
- E: toggle the configured watermark
- 3: toggle 2D sprite or 3D model rendering
- V: toggle automatic 3D view rotation
- X: reset the centered skybox view and model scale
- W/A/S/D: look around in the 3D view
- Plus/Minus: move backward or forward along the 3D view direction
- 1/2: increase or decrease 3D keyboard movement sensitivity
- Left mouse drag / wheel: look around or move along the view direction
- Left bracket / Right bracket: decrease or increase model scale
- Comma / Period: decrease or increase 3D view-rotation speed
- M: toggle the configured multipass chain
- J: toggle random autopilot
- K: lock or unlock shader and playlist selection
- Y: toggle sequential autopilot
- Space: bypass or enable shader effects
- Z: save a processed PNG snapshot under the `--prefix` directory
- F10: capture a screenshot when `--enable-screenshot` is active
- Escape: quit

Run `acmxvk --help` for the complete command-line reference.

## Validation and current testing

Development builds are tested with Vulkan SDK 1.4 and with validation enabled
in both MXVK and ACMXVK. The current increment has
been exercised with shader-library loading, multipass rendering, configurable
history caches, MXWrite encoding, custom-uniform rendering, optional live audio
metrics, FFmpeg-decoded file reactivity, routed-tone FFT visualization, and FFT
spectrum history. Increments 7B through 7K were additionally tested with a
CUDA+MIDI build, live Left/Right filter changes, filtered Vulkan frame history,
resident `GpuMat` video input, and CUDA-resident clockwise, 180-degree, and
counterclockwise rotation on an NVIDIA RTX 2070. Increment 7F was tested with
H.264 NVDEC feeding CUDA rotation, acidcam-gpu, Vulkan history, and repeated
playback without a host-frame handoff. Increment 7G was also tested with the
acidcam-gpu filter omitted while retaining direct NVDEC rotation, Vulkan sprite
upload, and layered history. Increment 7H verified the same resident route in a
`-DWITH_CUDA=OFF` build and separately regression-tested the optional filtered
build. Increment 7I verified explicit device 0 selection in both configurations,
including `decode=cuda:0` from MXVK and a clean direct history path. Increment
7J looped an 84-frame H.264 source into a 186-frame output while retaining one
NVDEC decoder open, direct rotation/upload/history, and clean validation. The
Increment 7K repeated the same NVDEC source through the event-protected surface
handoff with and without acidcam-gpu filters. The known duplicate vkBasalt
implicit-layer warning is external to ACMXVK. Increment 7L was tested by writing
live microphone input to a standalone PCM16 WAV, then writing the WAV alongside
an H.264/AAC output from the same capture buffer. Both files' channel,
sample-rate, duration, codec, and sample-format metadata were inspected.
Increment 7M exercised all three audio controls through SDL keyboard events
against repeating file audio and verified ACMX2 MIDI Map actions 81, 268, and
269 as active mappings under Vulkan validation. Increment 7N captured and
inspected 1280x720 RGBA PNGs with both one-shot and continuous readback, and
verified MIDI Map action 90 as an active snapshot mapping. A 3840x2091 stress
capture then processed another keyboard event between the snapshot's queue and
completion messages, and an immediate-exit test drained the pending PNG before
Vulkan teardown. Increment 7O rendered and inspected watermark-on and
watermark-off 1920x1080 processed snapshots, then inspected a 640x360 H.264
MXWrite output frame with both labels present. A second CUDA-filter encode
confirmed the `SquareBlockResize [3]` label from the live acidcam-gpu filter
state. All runs completed under Vulkan validation without project validation
errors. Increment 7P loaded its font, sprite shader, manifest, and selected
effect exclusively from an isolated `--path` tree, encoded the result, then
repeated default discovery through `ACMXVK_PATH`. Increment 7Q was compiled in
CUDA, audio, and MIDI mode and again with ACMXVK audio, MIDI, and acidcam-gpu
filters disabled. Both configurations linked against a staged MXVK 0.26.0;
their command-line and CUDA capability smoke checks completed successfully.
Increment 7R passed its malformed UTF-8, control-character, identifier,
structured-value, URL, bounded-line, and UTF-8 truncation regression suite in
both configurations. Additional CLI probes rejected control characters,
disallowed encoder punctuation, and oversized output dimensions before Vulkan
initialization. Increment 7S built against staged CUDA and non-CUDA MXVK 0.27.0
packages. Under Vulkan validation, two deterministic 30-frame H.264 recordings
with the HUD shown and hidden produced identical decoded-frame SHA-256 hashes;
an otherwise identical watermark recording produced a different hash, proving
that preview status is excluded while explicit saved overlays remain embedded.
Increment 7V built with CUDA, audio, MIDI, and validation against a staged MXVK
0.28.0 package. A CUDA/NVDEC input and NVENC output run completed without
validation errors, and its destructor flush produced all 30 expected frames in
the one-second H.264 test clip. MXVK and ACMXVK also compiled cleanly in the
non-CUDA configuration used by portable and Apple builds. Increment 7W then
repeated a 1920x1080 NVDEC/NVENC recording driven by live-audio PTS. Selecting
the RTX 2070's host-cached memory type increased delivery from 6–7 frames to 59
frames over two seconds; the resulting H.264 and AAC streams both measured
exactly two seconds, with a 29.5 FPS average video rate.
Increment 8I loaded both the bundled OBJ and MXVK's compressed
`cube.mxmod.z`, mapped a 1920x1080 still image over the complete mesh, and
encoded the source-sized 3D render through the existing Vulkan post-process
and pipelined readback path. A separate bundled-OBJ run completed with the
Vulkan SDK validation layer enabled and no project validation errors; the
duplicate vkBasalt implicit-layer warning remained external to ACMXVK.
Increment 8J then verified the centered skybox transform and direct model-UV
fragment pipeline in the portable build; the active fragment was removed from
the later screen-space chain so it was evaluated exactly once.
Increment 8K validated standalone compute selection and a mixed compute plus
selected-fragment pass chain against MXVK 0.33.0. Both produced encoded cube
interiors with the processed image following the cube faces, and Vulkan
validation reported no project errors. CUDA and non-CUDA builds and the input
validation test also passed.

## Development note

I have been using the **Codex CLI from OpenAI** as an
engineering aid while porting ACMX2 to MXVK. Codex has assisted with incremental
code translation, CMake integration, shader conversion, debugging, Vulkan
validation testing, and documentation. Project direction, testing decisions,
and maintenance remain under the project owner's control.
