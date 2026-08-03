
# ACMX2
<img width="2560" height="1440" alt="Screenshot From 2026-04-13 07-41-08" src="https://github.com/user-attachments/assets/0e0cd74f-ce6b-47e5-abfa-bc268cd74d4b" />

#  Now Works on AMD, Intel, and Apple Hardware (NVIDIA / CUDA Optional)

<img width="3360" height="2100" alt="acmx2 macos" src="https://github.com/user-attachments/assets/851527a0-978f-40ee-9edb-00b7d09b9b91" />


[Full Documentation](https://lostsidedead.biz/acmx2/docs/)

[YouTube Video Tutorial](https://youtu.be/-IDAF8MMmkg)

To regenerate local docs in a versioned folder and refresh `docs/latest`:

```bash
./build-docs.sh
```

This keeps `docs/index.html` as a stable redirect to the newest generated docs.

The command-line engine for **acidcam-gpu**. Applies GLSL shaders to live camera feeds, video files, or static images in real time, with optional CUDA-accelerated GPU filters when built on NVIDIA hardware. Supports 3D model rendering, audio reactivity, MIDI control, shader playlists, and multipass shader chains.

> **NVIDIA GPUs are not required.** ACMX2 is built around an OpenGL/SDL2 shader pipeline that runs on NVIDIA, AMD, Intel, and Apple GPUs. The CUDA GPU-filter stack is **opt-in at compile time** via `-DWITH_CUDA=ON` and only requires NVIDIA hardware + CUDA-enabled OpenCV when you choose to enable it.

---

## Features

- **Camera, video, or image input** with configurable resolution
- **Shader library** — load a single fragment shader or a full directory using
  `library.json` when present, with `index.txt` as a compatibility fallback
- **CUDA GPU filters** — apply GPU-accelerated pixel filters in addition to shaders
- **3D mode** — render shaders onto a 3D model (`.mxmod`)
- **Multipass shaders** — chain multiple shader passes in a single frame
- **Random multipass mode** — generate random 1–5 shader chains on the fly with crossfade transitions; navigate the main shader with Up/Down while in random mode
- **Shader playlists** — cycle through an ordered list of shaders
- **Audio reactivity** — shaders respond to real-time audio input (RtAudio/PulseAudio)
- **File-based audio reactivity** — drive audio-reactive shaders from an audio or video file instead of a live microphone via `--audio-file`; the audio track is automatically muxed into the output video
- **MIDI control** — map hardware knobs and buttons to shader parameters
- **Video recording** with optional audio muxing via FFmpeg
- **Up to 8K recording support** — 4K and below records as H.264; above 4K records as HEVC (H.265)
- **Recording quality controls** — preset, tune, CRF, codec mode, and realtime low-latency encoding are exposed in both CLI and Qt interface
- **Lossless HEVC/NVENC workflows** — select `hevc_nvenc`, NVENC `p1`–`p7` presets, `lossless` tuning, and additional FFmpeg-style options passed through MXWrite
- **Silent mode** — headless video processing without a window
- **HDR video pipeline** — detects BT.2020 HDR sources, processes them in linear BT.2020, and re-encodes them as HDR HEVC Main10
- **Shader cache** — precompile shader binaries for fast startup on supported
  OpenGL drivers; source compilation is always used on macOS
- **Live shader coding** — save a shader in the Qt editor while ACMX2 is running to recompile and reload only that shader without restarting the session
- **Shader code editor** — line numbers, GLSL highlighting, bracket matching,
  automatic indentation/pairing, line operations, search/replace, persistent
  font and word-wrap settings, and direct navigation to source locations
- **Find in Files** — recursively search shader sources with regular expressions
  and open any result directly at its matching line and column
- **Custom interface themes** — choose from 25 built-in light and dark
  stylesheet presets or edit and persist custom QSS
- **Qt6 GUI** available via the `interface/` subdirectory (`acmx2_interface`)
- **MIDI Map Tool** — standalone Qt6 app for creating MIDI controller mappings (`interface/midi-map/`)

## August 2026 Updates (Month to Date)

### Live Shader Coding

- Saving a shader from the built-in Qt editor now publishes a shared-memory reload request to the running ACMX2 process.
- ACMX2 recompiles only the edited shader slot, including both its 2D and 3D variants when dual mode is enabled.
- Replacement programs are compiled and initialized before they are installed. A compile, link, or uniform-setup failure leaves the currently working shader active.
- Full OpenGL compiler and linker diagnostics are written to the interface log, making edit-save-preview iteration possible without restarting the render session.
- Live reload works with both full-library launches and the interface's single-shader launch mode. Canonical path and library-index checks prevent an editor save from replacing the wrong program.
- On supported non-macOS platforms, saving also marks the binary shader cache stale so a later launch can rebuild the persistent cache from the updated source.

### macOS Shader Authoring and Editor Workflow

- Persistent shader binaries are disabled on macOS because Apple's
  Metal-backed OpenGL implementation does not support `glProgramBinary`.
  **Run from Cache** is disabled, **Rebuild Shader Cache** and the compile-health
  column are hidden, and Run Selected/Run All compile shader source each time.
- Saving from the editor on macOS sends only the live-reload request; it does
  not try to write or invalidate an unsupported binary cache.
- Editor File/Edit/View menus remain attached to each editor window instead of
  moving into the macOS global menu bar, matching the main interface behavior.
- The editor now includes line numbers, current line/column status, current-line
  and matching-bracket highlighting, auto-indent and paired delimiters,
  duplicate/move/comment/indent line tools, find/replace, Go to Line, adjustable
  font size, and persistent word wrap.
- **List > Find in Files** (`Ctrl+Shift+F`) searches `.glsl`, `.frag`, and `.vert`
  files recursively with a regular expression and optional case sensitivity.
  Results show file, line, match, and source text in a continuous uniform list;
  activating a result opens the editor with the exact match selected.
- The Custom Style Editor now includes ten additional presets: Lavender Mist,
  Rose Quartz, Sandstone, Mint & Navy, High Contrast, Cyberpunk Neon, Dracula,
  Nord Frost, Solarized, and Graphite Orange. This brings the built-in palette
  collection to 25 styles, while preserving editable QSS and the last selected
  preset through `QSettings`.

### Shader Library Manifest Updates

- ACMX2 and the interface prefer `library.json` whenever it exists and retain
  `index.txt` as a compatibility fallback.
- Loading a text-only library in the interface automatically creates
  `library.json` without modifying the original `index.txt`.
- Cache building, live reload, sorting, adding/removing shaders, and Remove
  Broken all operate on the selected manifest.
- `convert-index-to-json.pl` provides the same conversion as a standalone,
  core-Perl command-line utility with overwrite protection.

### Encoding and Output

- The Qt Session Settings dialog now exposes explicit `h264_nvenc` and `hevc_nvenc` codec choices, NVENC presets `p1` through `p7`, and NVENC tunes including `lossless`.
- A persistent **Extra FFmpeg Parameters** field forwards additional video-encoder options through ACMX2's `--encode-params` option to the repository-local MXWrite library.
- MXWrite accepts FFmpeg-style parameters such as `-profile:v rext` and `-pix_fmt yuv444p`, enabling options that are not represented by dedicated controls.
- Lossless hardware HEVC output can be configured with:

  ```bash
  acmx2 --encode-codec hevc_nvenc \
        --encode-params "-preset p6 -tune lossless -profile:v rext -pix_fmt yuv444p" \
        -o output_hardware_lossless.mkv
  ```

- MKV remains the recommended container when combining HEVC, lossless tuning, and non-default pixel formats.

### Runtime and Warmup Behavior

- Right-click shader selection from the interface now remains available during playlist playback. It changes the post/main shader without replacing the playlist node's multipass list.
- Normal Up/Down shader navigation now consistently starts a crossfade when moving between valid library entries.
- Texture-cache startup now replicates the first real source frame across every history slot instead of seeding with black frames. This applies to legacy 2D texture slots and `sampler2DArray` history mode and prevents dark startup trails.
- Build scripts and generated documentation workflows received additional portability and packaging updates.

## July 2026 Updates

### Texture and Audio History

- **Texture cache array mode:** `--texture-cache-array` exposes frame history through one `sampler2DArray history` ring with `history_head`, while `--texture-cache-size <N>` supports runtime-selected cache depths up to 64 frames.
- **Dual cache interface:** shaders can support array mode and the legacy `samp1`–`samp8`/`textures[SIZE]` path through the injected `USE_HISTORY_TEXTURE_ARRAY` macro.
- **Shader migration tooling:** `scripts/migrate_cache_samplers.pl` converts existing cache shaders to the array-based interface.
- **Audio spectrum history array:** `--enable-audio-buffers <N>` now exposes rolling FFT history through one runtime-sized `sampler1DArray spectrum_history`, limited by the GPU's array-layer capacity. `spectrum0` remains a current-frame compatibility alias.
- **Spectrum migration tooling:** `scripts/migrate_spectrum_samplers.pl` converts legacy `spectrum1`, `spectrum2`, and later sampler lookups to array-layer access.
- **Audio startup control:** `--audio-warm-rate <value>` fades audio-reactive uniforms and spectrum textures in at startup to reduce initial transients.

### Playback, Processing, and Models

- File audio/video synchronization and pass-through timing were reworked so decode, playback, rendering, and MXWrite output use a more consistent clock and drain behavior.
- File-audio recording and muxing now use explicit opt-in behavior, avoiding unintended output or audio-copy operations during preview-only sessions.
- Texture caching was extended to camera and still-image input in addition to video input, and startup warmup was hardened to keep splash/loading frames out of history.
- Shared-memory runtime controls allow the Qt interface to select the active shader and update playback settings while ACMX2 is running.
- Playlist authoring gained shuffle, concat, and clear actions, while random/sequential autopilot and post-multipass navigation expanded live-performance control.
- Long-running shaders now preserve trigonometric phase continuity when `time_f` wraps, avoiding visible jumps after extended sessions.
- Qt session dialogs preserve their last-used values; preferred camera FPS is restored after capability re-enumeration, and the metadata viewer exposes input-media details without leaving the interface.
- The shader editor now prompts before closing modified source through Escape or the window close action.
- ONNX processing gained expanded model/YAML coverage, preprocessing and smoothing improvements, and additional style/model assets.
- New shader packs, playlists, and MXMOD geometry assets expanded the included performance content.

### Encoding, Safety, and Builds

- ACMX2 switched to the repository-local MXWrite implementation and added explicit hardware/software encoder mode reporting.
- Optional `hevc_nvenc` support joined the existing H.264 paths, with automatic fallback behavior where appropriate.
- Headless/silent processing, startup pipeline reporting, and color-coded terminal messages improved long-running batch monitoring.
- HDR and color handling gained HLG-to-HDR10 workflow updates, MKV output support, and more consistent SDR TIFF/WebP snapshot behavior.
- Output safety checks now reject using the same file as both input and output instead of allowing a destructive read/write collision.
- Pcons build scripts, dependency setup helpers, and OpenCV 5 compatibility work improved portable CUDA and non-CUDA builds.

## Latest Features – May 2026

The past week has brought significant expansions to the ONNX/DNN model system and output capabilities:

### ONNX Model Library Expansion

- **FP16 Optimization**: DNN inference now uses half-precision floating-point for faster computation on hardware that supports it, maintaining visual quality while improving performance.
- **New Pre-Trained Models**:
  - **Bubble Effect** — Creative bubble distortion effect for surreal imagery
  - **Cartoon Effect** — Stylizes video/images into a cartoon aesthetic
  - **Color Splash** — Selective color desaturation while preserving target color channels
  - **Pencil Sketch** — Real-time pencil sketch rendering from video input
  - **Custom Style Transfer** — Neural style transfer for artistic transformations
  - **Edge Detection** — Advanced DNN-based edge detection superior to traditional methods
- **Generic ONNX/YAML Loading**: Load any ONNX model using `--onnx <file>` without recompilation. YAML files specify:
  - Model path (absolute or relative to YAML file)
  - Input preprocessing parameters (size, scale, BGR/RGB swap)
  - Enables rapid model integration and experimentation

### Generate Mode 

- **Generate Mode** (`--generate`): New automatic image generation mode that creates image files at a given frame interval.
- **Random Generate in Interface**: Qt interface now includes a "Generate" button that generates images at a given interval.

### Output & Recording Enhancements

- **PNG Frame Export** (`--png`): Save processed frames as individual PNG files instead of video encoding, enabling frame-by-frame workflows and high-quality archival.
- **Audio Animation Mux**: Embedded audio track animation during file processing provides visual progress feedback that the application is working on long-running jobs.

### Color & Tone Controls

- **Black Point Control** (`--black <point>`): Shadow crush and black level threshold adjustment (default: 0.35)
- **White Point Control** (`--white <point>`): Opacity saturation and white level threshold adjustment (default: 0.75)
- **Settings Window Scaling**: Increased height of Settings dialog to accommodate expanded model lists and improved UI organization.

## Qt Interface Notes

Recent Qt interface updates focus on session usability and repeatability:

- The **Settings**, **Audio Settings**, **GPU Filter Settings**, and **MIDI Settings** dialogs now restore the last values you used when reopened.
- Dialog state is stored with `QSettings`, so the same values are also available again on the next launch.
- If no saved settings exist yet, the **Settings** dialog starts with camera capture resolution set to `1280x720` and screen/output resolution set to `Default`.
- Camera capability enumeration still dynamically fills the resolution/FPS lists, and the restored/default value is applied when available.
- Camera FPS now persists as a preferred value and is re-selected after capability repopulation when that FPS is available.
- The **Settings** dialog also includes an **Encoding Quality** section for software/NVENC presets, tune, CRF, codec selection (`auto`, `software`, `nvenc`, `h264_nvenc`, or `hevc_nvenc`), extra FFmpeg-style encoder parameters, and realtime low-latency encoding.
- Encoding controls map directly to the CLI flags `--encode-preset`, `--encode-tune`, `--encode-crf`, `--encode-codec`, `--encode-params`, and `--encode-realtime`.

### Shader Editor and Find in Files

The shader editor provides line numbers, GLSL syntax highlighting, current-line
and matching-bracket highlighting, and a line/column status display. It
automatically indents new blocks, inserts matching brackets and quotes, and
supports smart Home, four-space Tab/Shift+Tab indentation, duplicate line
(`Ctrl+D`), toggle comment (`Ctrl+/`), and move line (`Alt+Up`/`Alt+Down`).

The Edit and View menus also provide undo/redo, find next/previous, replace, Go
to Line (`Ctrl+G`), selection indentation, font zoom, and persistent word-wrap,
font-size, and window-geometry settings. On macOS these menus are embedded in each
editor window rather than being placed in the system-wide menu bar.

Use **List > Find in Files** (`Ctrl+Shift+F`) from the main interface to search
the active shader directory recursively. The query is a Qt regular expression,
with optional case sensitivity, and searches `.glsl`, `.frag`, and `.vert`
files. Each result includes its relative file, line number, matched text, and
source line. Double-click a result or choose **Open Result** to open that shader
at the exact line and column with the match selected.

### Live Shader Coding from the Qt Editor

Launch ACMX2 from the interface, open a shader from the library, edit it, and
save normally. The editor sends the saved shader's canonical path and stable
library index through the existing interface shared-memory channel. The render
process notices the new request on its next frame and recompiles only that
shader; it does not rebuild or reload the rest of the library.

Successful compilation replaces the shader immediately. ACMX2 prepares the new
program and all known uniform locations before swapping it into the live
library, including the 3D variant when dual mode is active. If compilation or
linking fails, the last valid program remains in use and the driver-provided
error string appears in the interface log. Fix the source and save again to
retry without stopping the session.

### Shader Library Manifests

Shader-library paths are directories. When both manifests exist, ACMX2 and its
Qt interface use `library.json`; otherwise they fall back to `index.txt`.
When the Qt interface loads an older library that only has `index.txt`, it
automatically creates an equivalent `library.json` and leaves the original
text manifest unchanged.
The JSON format is:

```json
{
  "version": 1,
  "shaders": [
    "plasma.glsl",
    "feedback_cache.glsl"
  ]
}
```

The New Shader Library dialog can create `library.json` directly. If a library
contains only `index.txt`, its first interface load creates the JSON manifest
automatically. Sorting, adding or removing shaders, live reload, cache builds,
and Remove Broken all operate on the selected manifest.

Legacy libraries can also be converted from the command line:

```bash
./convert-index-to-json.pl ./shaders
./convert-index-to-json.pl ./shaders/index.txt
```

The converter accepts either a library directory or an `index.txt` path. It
keeps the original text file and refuses to replace an existing `library.json`
unless `--force` is supplied. Use `--output <file>` to choose another output
path. The script depends only on Perl core modules, including `JSON::PP`.

---

## Building

### macOS (Intel & Apple Silicon)

ACMX2 builds and runs on macOS using OpenGL 4.1 backed by Metal. **CUDA is not available on macOS**, so the engine automatically builds with `-DWITH_CUDA=OFF`. Shader-based effects work fully; only the CUDA GPU-filter pipeline is omitted.

The Apple Metal-backed OpenGL 4.1 driver does not support `glProgramBinary`, so
**the shader binary cache is disabled at runtime** on macOS. The interface
disables **Run from Cache**, hides **Rebuild Shader Cache** and the compile-health
column, and passes `--no-cache` for normal launches so shaders compile from
source on every run. Live editor saves remain supported and request an in-place
source recompile without attempting to save a binary cache.

**Quick start** (one command from an empty directory):

```bash
./macos/install.sh
```

This installs Homebrew packages, clones `libmx2` and `acidcam-gpu` (which contains `ACMX2`), builds everything, and installs to `/usr/local/bin/`.

**Or step by step:**

1. Install Homebrew packages:

   ```bash
   ./macos/install-dep.sh
   ```

   Required: `cmake`, `pkg-config`, `sdl2`, `sdl2_ttf`, `sdl2_mixer`, `sdl2_image`, `glm`, `opencv`, `ffmpeg`, `qt6`

2. Build and install:

   ```bash
   ./macos/build-macos.sh
   ```

3. Download the **macOS-compatible shader pack** (the default library contains shaders that crash the Apple GL driver):

   ```
   https://lostsidedead.biz/acmx2/shaders.macos.zip

   ```

4. Run:

   ```bash
   acmx2_interface
   # or from command line:
   acmx2 -p /usr/local/share/acmx2/data -s /path/to/macos-shaders -d 0
   ```

See [macos/README.md](macos/README.md) for full details, troubleshooting, and CMake flags.

---

### Linux

ACMX2 is part of the acidcam-gpu project. See the [main README](../README.md) for full build instructions.

```bash
cd ACMX2
mkdir build && cd build
cmake .. -DAUDIO=ON -DMIDI=ON
make -j$(nproc) && sudo make install
```

The Qt6 GUI is built separately:

```bash
cd ACMX2/interface
mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

### Optional Features

CUDA, audio, and MIDI support are all **optional** at compile time. Each subsystem
is gated by a CMake option and a corresponding preprocessor definition:

| Option | Default | Definition | Disables |
|--------|---------|------------|----------|
| `WITH_CUDA` | `ON`  | `ACMX2_WITH_CUDA` | CUDA GPU filters, zero-copy CUDA/GL interop, FFmpeg CUDA hw-decode, `--gpu-filter`, `--gpu-buffer`, `--cuda-device`, `--list-cuda-devices` |
| `AUDIO`     | `OFF` | `AUDIO_ENABLED`   | RtAudio capture, audio reactivity, `--enable-audio`, `--audio-file`, `--record-audio`, audio controls |
| `MIDI`      | `OFF` | `MIDI_ENABLED`    | RtMidi input, `--midi-map`, `--midi-device`, `--list-midi`, MIDI overlay |

#### Building without CUDA (OpenGL-only)

If you do not have an NVIDIA GPU, do not want to install the CUDA toolkit, or want
to build against a stock OpenCV that was not compiled with CUDA support, turn
`WITH_CUDA` off. The engine falls back to pure OpenGL/SDL2 rendering; shader-based
effects still work, only the CUDA GPU filter stack is omitted.

```bash
cd ACMX2
mkdir build-nocuda && cd build-nocuda
cmake .. -DWITH_CUDA=OFF
make -j$(nproc) && sudo make install
```

You can combine the flags freely — for example an OpenGL-only build with audio:

```bash
cmake .. -DWITH_CUDA=OFF -DAUDIO=ON
```

Or to enable ONNX/DNN model loading:

```bash
cmake .. -DWITH_OPENCV_DNN=ON
```

Note: `-DWITH_OPENCV_DNN=ON` requires the `yaml-cpp` package to be installed for YAML configuration file parsing.
It can be combined with `-DWITH_CUDA=OFF`; DNN inference will then use the
OpenCV CPU backend. With CUDA enabled, ACMX2 benchmarks CPU and CUDA once per
loaded model and keeps the faster backend. Set `ACMX2_DNN_BACKEND=cpu`,
`cuda`, or `cuda_fp16` to override automatic selection.

Generic ONNX YAML files may enable runtime-dynamic spatial dimensions:

```yaml
model:
  path: comic.onnx
  input: input
preprocessing:
  width: 256
  height: 256
  dynamic: true
  alignment: 4
```

Set either `width` or `height` to `0` to derive that dimension from the source
aspect ratio, or set both to `0` to use the source dimensions. Dynamic sizes are
rounded to `alignment` because style-transfer networks commonly downsample by
four. ACMX2 re-plans and re-benchmarks the selected backend if the resolved
input dimensions change. Dynamic configurations at 256×256 or below also apply
an edge-preserving bilateral filter before upscaling. CUDA-selected models use
`cv::cuda::bilateralFilter`; CPU-selected models use the equivalent CPU filter.
Override its defaults with:

```yaml
postprocessing:
  bilateral:
    enabled: true
    diameter: 5
    sigma_color: 50.0
    sigma_space: 50.0
```

The model directory includes four optional dynamic quality tiers for every
YAML-configured style model:

- `*-256.yaml` — fastest; bilateral smoothing enabled automatically
- `*-512.yaml` — balanced quality
- `*-768.yaml` — high quality
- `*-1024.yaml` — ultra quality for slower or offline rendering

At startup, the Qt6 interface probes the installed `acmx2` binary with
`--check-cuda`, `--check-audio`, and `--check-midi` and automatically disables
menu entries (GPU Filter Settings, Audio Settings, MIDI Settings) and CLI
arguments corresponding to features that were not compiled in. You can also run
the probes manually:

```bash
acmx2 --check-cuda     # "CUDA: enabled"  or "CUDA: disabled"
acmx2 --check-audio    # "AUDIO: enabled" or "AUDIO: disabled"
acmx2 --check-midi     # "MIDI: enabled"  or "MIDI: disabled"
```

---

## Installed File Layout

`make install` places files under `CMAKE_INSTALL_PREFIX` (default `/usr/local`):

| Path | Contents |
|------|----------|
| `bin/acmx2` | Main engine binary |
| `bin/acmx2_interface` | Qt6 GUI launcher |
| `share/acmx2/data/` | Assets (icon, fonts, models, shaders, textures) |
| `share/acmx2/acmx2.png` | Application icon |
| `share/applications/acmx2.desktop` | Desktop entry for `acmx2` |
| `share/applications/acmx2-interface.desktop` | Desktop entry for `acmx2_interface` |

Both `acmx2` and `acmx2_interface` automatically locate the installed data directory at `<prefix>/share/acmx2/` when the local `./data` directory is not present. You can still override the assets path with `-p <dir>`.

### Distrobox Export

When running inside a Distrobox container, export the applications to the host desktop:

```bash
distrobox-export --app acmx2_interface
distrobox-export --app acmx2
```

The `.desktop` files include `StartupWMClass` entries so the correct icon appears in the dock while running.
---

## Usage Examples

**Camera with a shader library:**
```bash
./acmx2 -p ./data -s ./shaders -d 0 -r 1920x1080
```

**Process a video file with GPU filters and record output:**
```bash
./acmx2 -p ./data -i input.mp4 -s ./shaders --gpu-filter 0,5,12 -o output.mp4 --copy-audio
```

**Single shader, fullscreen, with audio reactivity:**
```bash
./acmx2 -p ./data -f effect.glsl -d 0 -n -w --audio-input 3
```

**3D mode with a model:**
```bash
./acmx2 -p ./data -s ./shaders --enable-3d --model cube.mxmod -d 0
```

**Silent (headless) batch processing:**
```bash
./acmx2 -p ./data -i input.mp4 -s ./shaders --shader 5 --silent -o output.mp4
```

**Silent HDR batch processing:**
```bash
./acmx2 -p ./data -i hdr_input.mp4 -s ./shaders --shader-pass 0,4,9 --silent -o hdr_output.mp4
```

**Build shader cache:**
```bash
./acmx2 -p ./data --build ./shaders --enable-3d
```

**Process a video with audio-reactive shaders driven by a music file:**
```bash
./acmx2 -p ./data -i input.mp4 -s ./shaders --audio-file music.mp3 -o output.mp4
```

**Same as above, but stop when the audio track ends:**
```bash
./acmx2 -p ./data -i input.mp4 -s ./shaders --audio-file music.mp3 --audio-trunc -o output.mp4
```

---

## Command-Line Arguments

### General

| Short | Long | Value | Description |
|-------|------|-------|-------------|
| `-v`, `-h` | `--help`, `--version` | | Display the full program information, arguments, and keyboard controls, then exit |
| `-p` | `--path` | `<dir>` | Assets path |
| `-r` | `--resolution` | `WxH` | Window resolution (e.g. `1920x1080`) |
| `-d` | `--device` | `<index>` | Camera device index |
| `-c` | `--camera-res` | `WxH` | Camera capture resolution |
| `-i` | `--input` | `<file>` | Input video file |
| `-g` | `--graphic` | `<file>` | Input image file |
| `-o` | `--output` | `<file>` | Output video file |
| `-b` | `--bitrate` | `<crf>` | Output bitrate in CRF |
| `-u` | `--fps` | `<fps>` | Frames per second |
| `-e` | `--prefix` | `<path>` | Snapshot save prefix |
| `-a` | `--repeat` | | Loop/repeat video playback |
| `-n` / `-N` | `--fullscreen` | | Fullscreen window (Escape to quit) |
| `-m` | `--cuda-device` | `<index>` | CUDA device index |
| | `--interface-shm` | | Enable Qt interface shared-memory control channel (off by default for normal CLI runs) |
| | `--duration` | `<seconds>` | Recording duration limit in seconds (float); stop recording and exit after elapsed |
| | `--encode-preset` | `<name>` | Encoder preset: `ultrafast`..`veryslow` or NVENC `p1`..`p7` |
| | `--encode-tune` | `<name>` | Encoder tune, including NVENC `hq`, `uhq`, `ll`, `ull`, and `lossless` |
| | `--encode-crf` | `<0-51>` | Encoder CRF quality override (default: `18`) |
| | `--encode-codec` | `<mode>` | Encoder codec mode: `auto`, `software`, `nvenc`, `h264_nvenc`, or `hevc_nvenc` |
| | `--encode-params` | `<string>` | Additional FFmpeg-style video encoder options passed through MXWrite |
| | `--encode-realtime` | | Enable low-latency realtime encoding flags |
| | `--no-drop` | | Video-file processing: never drop frames; block when the encoder queue is full |

### Shader Options

| Short | Long | Value | Description |
|-------|------|-------|-------------|
| `-s` | `--shaders` | `<directory>` | Shader library directory (`library.json` preferred, `index.txt` fallback) |
| `-f` | `--fragment` | `<file>` | Single fragment shader file |
| | `--shader` | `<index>` | Initial shader index in library |
| | `--shader-pass` | `<indices>` | Shader pass indices (comma-separated, e.g. `0,1,2`) |
| | `--playlist` | `<file>` | Shader playlist text file (one shader name per line) |
| | `--build` | `<path>` | Build shader cache for specified library path and exit |
| | `--no-cache` | | Disable shader caching (always recompile shaders) |
| | `--time-speed` | `<float>` | Constant `time_f` speed multiplier (default: `1.0`) |

### GPU Filter Options

| Long | Value | Description |
|------|-------|-------------|
| `--gpu-filter` | `<indices>` | GPU filter indices (comma-separated) |
| `--gpu-buffer` | `<size>` | GPU frame buffer size (`4`–`32`) |
| `--list-filters` | | List available GPU filters and exit |
| `--list-cuda-devices` | | List available CUDA devices and exit |
| `--disable-counter` | | Disable timer and FPS counter overlay |
| `--silent` | | Process video without window. Only valid for `-i/--input` video files and requires `-o/--output`; camera and image input are not supported. |

### 3D / Model Options

| Long | Value | Description |
|------|-------|-------------|
| `--texture-cache` | | Enable texture cache (camera, video, and graphic modes) |
| `--cache-delay` | `<frames>` | Texture cache delay in frames |
| `--texture-cache-size` | `<frames>` | Texture cache ring size (`1`–`64`, default `8`) |
| `--texture-cache-array` | | Bind the cache as `sampler2DArray history` instead of separate samplers |
| `--copy-audio` | | Copy audio track from input to output |
| `--enable-3d` | | Enable 3D cube rendering |
| `--model` | `<file>` | 3D model file (`.mxmod`) |

### Audio Options (requires `AUDIO_ENABLED` build)

| Short | Long | Value | Description |
|-------|------|-------|-------------|
| `-w` | `--enable-audio` | | Enable audio reactivity |
| `-l` | `--channels` | `<num>` | Audio channels |
| `-q` | `--sense` | `<float>` | Audio sensitivity |
| `-y` | `--pass-through` | | Enable audio pass-through |
| | `--audio-input` | `<index>` | Audio input device (`default` or index) |
| | `--audio-output` | `<index>` | Audio output device (`default` or index) |
| | `--list-devices` | | List audio devices and exit |
| | `--record-audio` | `<file>` | Record captured audio to WAV file |
| | `--record-gain` | `<float>` | Recording volume gain `0.0`–`2.0` (default: `1.0`) |
| | `--audio-file` | `<file>` | Use audio from a file for reactivity instead of the microphone; the audio track is muxed into the output video |
| | `--audio-trunc` | | Stop playback when the audio file reaches the end |
| | `--audio-warm-rate` | `<float>` | Startup audio warmup rate in `1/sec` (default: `0.5`; `0` disables warmup) |
| | `--enable-audio-buffers` | `<N>` | Allocate one FFT history `sampler1DArray` with `N` GPU-limited layers |

### MIDI Options (requires `MIDI_ENABLED` build)

| Long | Value | Description |
|------|-------|-------------|
| `--midi-map` | `<file>` | MIDI config file (`.midi_cfg`) |
| `--midi-device` | `<index>` | MIDI input device index |
| `--list-midi` | | List available MIDI input devices and exit |

---

## HDR Video Processing

ACMX2 automatically enables its HDR path when the input video is tagged as BT.2020 HDR with PQ (`SMPTE ST.2084`) or HLG (`ARIB STD-B67`) transfer characteristics, or when the decoded stream is a 10-bit BT.2020 format such as `yuv420p10le` or `p010le`.

On ingest, the program preserves the source HDR encoding long enough to upload each frame into a 16-bit RGBA texture. A dedicated HDR decode shader converts PQ or HLG into linear BT.2020 before the normal shader chain and optional CUDA filters run. After the last effect pass, a matching HDR encode shader converts the result back to the same HDR transfer family as the source.

HDR exports are written as HEVC Main10, not H.264. The output path feeds MXWrite a BT.2020 HDR frame that is quantized to a 10-bit `P010` HEVC stream, with BT.2020 primaries and the original PQ or HLG transfer preserved. If the input contains mastering-display or content-light metadata, ACMX2 forwards that HDR metadata to the encoded output.

That gives you an end-to-end HDR round-trip: HDR in, effects in linear BT.2020, HDR out. The terminal log reports this explicitly when active, for example `HDR output mode enabled: HEVC Main10 + BT.2020 + PQ`.

## Using `--silent` for Terminal Processing

`--silent` is the headless transcoding mode. It creates an off-screen context, does not open a visible SDL window, and skips playback pacing so the job runs as fast as possible instead of real time.

- Use it only with `-i/--input` video files.
- Always pair it with `-o/--output`.
- Do not use it with camera capture or `-g/--graphic` image input.
- Audio copy and mux steps still happen at the end when requested.

Typical use:

```bash
./acmx2 -p ./data -i input.mp4 -s ./shaders --shader 12 --silent -o output.mp4
```

HDR files use the same flag set and automatically stay in HDR:

```bash
./acmx2 -p ./data -i hdr_input.mp4 -s ./shaders --gpu-filter 0,5 --silent -o hdr_output.mp4 --copy-audio
```

Because progress is printed to stdout in headless mode, `--silent` is suitable for shell pipelines and logs:

```bash
./acmx2 -p ./data -i input.mp4 -s ./shaders --silent -o output.mp4 | tee acmx2-batch.log
```

---

## Keyboard Controls

### General Controls

| Key | Action |
|-----|--------|
| `Up` | Previous shader (or previous playlist shader if playlist enabled) |
| `Down` | Next shader (or next playlist shader if playlist enabled) |
| `K` | Toggle shader lock (prevent Up/Down from switching shaders) |
| `R` | Toggle random multipass mode (generates random 1–5 shader chain with crossfade; press again to restore previous state) |
| `G` | Generate a new random shader chain (while in random multipass mode) |
| `Left` | Previous GPU filter (if GPU filters enabled) |
| `Right` | Next GPU filter (if GPU filters enabled) |
| `Space` | Toggle shader processing bypass |
| `P` | Toggle playlist mode / Pause video (Video/Image modes) |
| `L` | Toggle video freeze (Video/Image modes) |
| `Z` | Take snapshot |
| `M` | Toggle multi-shader pass (if `--shader-pass` set) |
| `3` | Toggle 2D/3D mode (if `--enable-3d` active) |
| `E` | Toggle watermark |
| `F9` | Toggle overlay (timer/FPS counter) visibility |

### Time Controls

| Key | Action |
|-----|--------|
| `U` (hold) | Increase time step |
| `I` (hold) | Decrease time step |
| `T` | Toggle time on/off (Audio build) |
| `Q` | Toggle audio-reactive time (Audio build) |
| `Home` | Toggle audio delta time scaling (Audio build) |
|  `Page up/page Down` | Increase / Decrease Time speed |

### Audio Controls (Audio build)

| Key | Action |
|-----|--------|
| `Insert` | Increase audio sensitivity |
| `Delete` | Decrease audio sensitivity |

### 3D Mode Controls (when 3D enabled)

| Key | Action |
|-----|--------|
| `W` / `A` / `S` / `D` | Look around |
| `V` | Toggle view rotation |
| `O` | Toggle scale oscillation |
| `X` | Reset camera distance |
| `+` / `-` | Increase / decrease camera distance |
| `B` | Increase movement speed |
| `N` | Decrease movement speed |
| `C` | Toggle wave effect |

---

## Supported Shader Uniforms

All fragment shaders receive the following uniforms automatically. Uniforms that are not declared in your shader are silently ignored.

### Core Uniforms

| Uniform | Type | Description |
|---------|------|-------------|
| `samp` | `sampler2D` | Main video/camera texture |
| `alpha` | `float` | Alpha value (oscillates `0.0`–`1.0`) |
| `iTime` | `float` | Elapsed time in seconds since start |
| `time_f` | `float` | Time multiplier (adjustable with `U`/`I` keys and `--time-speed`) |
| `iFrame` | `int` | Frame counter |
| `iTimeDelta` | `float` | Time since last frame (seconds) |
| `iResolution` | `vec2` | Window resolution `(width, height)` |
| `iMouse` | `vec4` | Mouse position `(x, y, clickStartX, clickStartY)` |
| `iMouseClick` | `vec2` | Last mouse click position |
| `iDate` | `vec4` | Current date/time `(year, month, day, secondsOfDay)` |
| `iFrameRate` | `float` | Frame rate |
| `iChannelTime[0..3]` | `float` | Per-channel time |
| `iChannelResolution[0..3]` | `vec3` | Per-channel resolution |

### Texture Cache Uniforms (shaders with "cache" in the filename)

| Uniform | Type | Description |
|---------|------|-------------|
| `samp1`–`samp8` | `sampler2D` | Cached frame textures from the texture cache ring buffer |
| `textures[SIZE]` | `sampler2D[]` | Scalable oldest-to-newest cache view in the default binding mode |
| `history` | `sampler2DArray` | Optional ring-layer view, mapped oldest-to-newest with `history_head` |

Shaders can support both cache representations with the injected
`USE_HISTORY_TEXTURE_ARRAY` macro:

```glsl
#if USE_HISTORY_TEXTURE_ARRAY
uniform sampler2DArray history;
uniform int history_head;
#define HISTORY_LAYER(index) ((history_head + (index)) % SIZE)
#define SAMPLE_HISTORY(index, uv) \
    texture(history, vec3((uv), float(HISTORY_LAYER(index))))
#else
uniform sampler2D textures[SIZE];
#define SAMPLE_HISTORY(index, uv) texture(textures[index], (uv))
#endif
```

#### Why `sampler2DArray` Is the Optimal Choice

A `sampler2DArray` is one OpenGL texture object whose layers are equally sized
2D images. The engine binds the entire history cache to one texture unit instead
of binding `samp1` through `samp8` separately, which simplifies the host-side
OpenGL state and leaves more texture units available for other inputs.

The history layer is the third texture-coordinate component:

```glsl
uniform sampler2D samp;
uniform sampler2DArray history;
uniform int history_head;

vec4 sample_cache(int index, vec2 uv) {
    uv = mirror_repeat(uv);
    int layer = (history_head + index) % SIZE;
    return texture(history, vec3(uv, float(layer)));
}
```

Unlike dynamically indexing an array of sampler uniforms, the layer number is a
normal texture coordinate. GLSL 3.30 can therefore select it at runtime without
an eight-way `switch`, allowing the driver to emit a native array-texture fetch
on NVIDIA RTX and other supporting hardware.

The array is updated as a ring rather than shifted every frame.
`history_head` identifies the physical layer containing logical history index
zero; adding it preserves the oldest-to-newest shader contract while uploading
only one array layer per frame.

All layers must have identical dimensions and internal formats. Cache frames
already match the active framebuffer or viewport, so the texture cache
naturally satisfies that requirement. Enable this representation with
`--texture-cache-array`.

#### Gradual Shader Migration Has No Rendering Cost

Legacy sampler declarations may remain in a shader while it is being migrated.
GLSL compilers perform dead-code elimination from the fragment output, so an
unused `samp1`–`samp8` or `textures[SIZE]` declaration is removed from the
linked program. It performs no texture fetch, consumes no texture unit in the
linked shader, and allocates no texture storage by itself.

An optimized-out uniform has no active location:

```cpp
GLint location = glGetUniformLocation(program, "samp1"); // -1 when inactive
glUniform1i(location, 1);                                // specified no-op
```

OpenGL silently ignores a `glUniform*` call whose location is `-1`, so the
engine can continue querying legacy names during the transition. Shaders can
support both paths and be migrated one at a time:

```glsl
#if USE_HISTORY_TEXTURE_ARRAY
uniform sampler2DArray history;
uniform int history_head;
#else
uniform sampler2D samp1;
uniform sampler2D samp2;
// ...samp3 through samp8...
#endif

vec4 get_history(int index, vec2 uv) {
#if USE_HISTORY_TEXTURE_ARRAY
    int layer = (history_head + index) % SIZE;
    return texture(history, vec3(uv, float(layer)));
#else
    if (index == 0) return texture(samp1, uv);
    if (index == 1) return texture(samp2, uv);
    // ...remaining legacy cases...
    return vec4(0.0);
#endif
}
```

The engine defines `USE_HISTORY_TEXTURE_ARRAY` as `0` or `1`, so use `#if`
rather than `#ifdef`. The inactive branch is removed at preprocessing time and
adds no runtime branching. Mass conversion is optional; the repository utility
is available when a clean, array-only shader library is desired:

```bash
scripts/migrate_cache_samplers.pl --dry-run shaders
scripts/migrate_cache_samplers.pl shaders
```

### acidcamGL-Compatible Uniforms

| Uniform | Type | Description |
|---------|------|-------------|
| `value_alpha_r` | `float` | Oscillating red color alpha |
| `value_alpha_g` | `float` | Oscillating green color alpha |
| `value_alpha_b` | `float` | Oscillating blue color alpha |
| `alpha_r` | `float` | Red color alpha (same as `value_alpha_r`) |
| `alpha_g` | `float` | Green color alpha (same as `value_alpha_g`) |
| `alpha_b` | `float` | Blue color alpha (same as `value_alpha_b`) |
| `alpha_value` | `float` | Current alpha value |
| `index_value` | `float` | Current shader index in the library |
| `optx` | `vec4` | Option vector `(0.5, 0.5, 0.5, 0.5)` |
| `random_var` | `vec4` | Random variable vector |
| `restore_black` | `float` | Restore black flag (`0.0` or `1.0`) |
| `inc_value` | `vec4` | Incrementing value vector |
| `inc_valuex` | `vec4` | Secondary incrementing value vector |

### Audio-Reactive Uniforms (requires `AUDIO=ON` build + `-w` flag)

| Uniform | Type | Description |
|---------|------|-------------|
| `amp` | `float` | Amplitude scaled by sensitivity |
| `uamp` | `float` | Raw untouched amplitude |
| `iamp` | `float` | Estimated dominant frequency (Hz) |
| `amp_peak` | `float` | Highest sample value in the buffer |
| `amp_rms` | `float` | Root mean square energy |
| `amp_smooth` | `float` | Exponentially smoothed amplitude |
| `amp_low` | `float` | Low-frequency energy (<300 Hz) |
| `amp_mid` | `float` | Mid-frequency energy (300–3000 Hz) |
| `amp_high` | `float` | High-frequency energy (>3000 Hz) |
| `iSampleRate` | `float` | Audio sample rate (`44100.0`) |
| `spectrum` | `sampler1D` | FFT frequency-magnitude spectrum (256 bins, GL_TEXTURE9) |
| `spectrum0` | `sampler1D` | Current-frame alias of the live spectrum |
| `spectrum_history` | `sampler1DArray` | Runtime-sized FFT history array enabled by `--enable-audio-buffers <N>` |
| `spectrum_history_head` | `int` | Physical layer containing the newest history frame |
| `spectrum_history_size` | `int` | Allocated history-array layer count |

The `spectrum` uniform is a 1D texture (`GL_R32F`, 256 texels) holding the FFT magnitudes of the current audio frame. Texel coordinate `x = 0.0` is the DC bin and `x = 1.0` is the Nyquist frequency (22 050 Hz at 44 100 Hz sample rate). Sampling uses `GL_LINEAR` filtering and `GL_CLAMP_TO_EDGE` wrapping. Example usage:

```glsl
uniform sampler1D spectrum;           // bound to texture unit 9
float energy = texture(spectrum, x).r; // x in [0,1]
```

History age is a dynamic array coordinate, so one sampler binding supports any
requested depth up to `GL_MAX_ARRAY_TEXTURE_LAYERS`:

```glsl
uniform sampler1DArray spectrum_history;
uniform int spectrum_history_head;
uniform int spectrum_history_size;

int size = max(spectrum_history_size, 1);
int layer = (spectrum_history_head - (age % size) + size) % size;
float energy = texture(spectrum_history, vec2(frequency, float(layer))).r;
```

Here, `frequency` is in `[0,1]` and `age` is `0` for the newest frame, `1`
for the preceding frame, and so on. `spectrum0` remains a current-frame
`sampler1D` compatibility alias, but the engine no longer binds separate
`spectrum1`...`spectrumN` uniforms. Those names belong to the legacy shader
interface used by older ACMX2 releases; there is no current command-line mode
that restores them. Convert those legacy lookups with:

```bash
scripts/migrate_spectrum_samplers.pl --dry-run shaders
scripts/migrate_spectrum_samplers.pl shaders
```

### MIDI Slider Uniforms (requires `MIDI=ON` build)

| Uniform | Type | Description |
|---------|------|-------------|
| `slider1` | `float` | MIDI CC knob value mapped to 0.0–1.0 |
| `slider2` | `float` | MIDI CC knob value mapped to 0.0–1.0 |
| `slider3` | `float` | MIDI CC knob value mapped to 0.0–1.0 |
| `slider4` | `float` | MIDI CC knob value mapped to 0.0–1.0 |

These uniforms are optional. If a shader does not declare them they are silently skipped. Map physical MIDI knobs to Slider 1–4 in the midi-map tool, then use `uniform float slider1;` etc. in your GLSL code to receive live 0.0–1.0 values.

### Virtual Audio Device (PipeWire / PulseAudio)

Route application audio into ACMX2 for reactive effects:

```bash
pactl load-module module-null-sink sink_name=VirtualAudio sink_properties=device.description="Virtual_Audio"
```

Then select **Virtual_Audio.monitor** as the audio input device.

---

## Command-Line Arguments & Keyboard Controls

See the full tables in the [main acidcam-gpu README](../README.md#command-line-arguments).

---
