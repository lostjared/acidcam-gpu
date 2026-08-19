/# acidcam-gpu

<img width="2560" height="1440" alt="screenshot3" src="https://github.com/user-attachments/assets/3af169df-abd3-49e8-84cc-70021a42e253" />
<img width="2560" height="1440" alt="screenshot4" src="https://github.com/user-attachments/assets/35fb877a-4b2c-4bda-af6e-6dd894ae1593" />
<img width="1300" height="783" alt="about" src="https://github.com/user-attachments/assets/e6cd44c8-9da7-4a3b-b09b-e946926b9a25" />


[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Hardware: NVIDIA RTX (optional)](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%20(optional)-green.svg)](https://www.nvidia.com/en-us/geforce/rtx/)
[![Framework: CUDA (optional)](https://img.shields.io/badge/Framework-CUDA%2012.x%20(optional)-76b900.svg)](https://developer.nvidia.com/cuda-zone)

# ACMX2 – Linux / macOS (NVIDIA GPU Optional)
![screenshot](https://github.com/lostjared/acidcam-gpu/blob/main/image.jpg)

# acidcam-gpu / ACMX2

**acidcam-gpu** is a high-performance, real-time video manipulation engine designed to push the boundaries of psychedelic glitch art. Part of the **ACMX2** and **libmx2** ecosystem, it uses an OpenGL/GLSL shader pipeline as its core, with an **optional** CUDA GPU-filter path that can be enabled at compile time on NVIDIA hardware for additional accelerated effects.

> **OpenCV 5 is used.** The project has moved from OpenCV 4 to OpenCV 5. Use a stock OpenCV 5 build for the portable OpenGL configuration, or an OpenCV 5 build with CUDA support when enabling the optional CUDA pipeline.

> **OpenCV must include CUDA support for `WITH_CUDA=ON`.** Installing the NVIDIA CUDA toolkit by itself is not sufficient: CUDA support, including the OpenCV CUDA modules used by ACMX2, must be enabled when OpenCV is compiled. On Arch Linux, install `opencv-cuda` instead of the stock `opencv` package. On other distributions, use an equivalent CUDA-enabled OpenCV package or build OpenCV with CUDA enabled. If OpenCV has no CUDA modules, configure ACMX2 with `-DWITH_CUDA=OFF`.

> **NVIDIA GPUs are no longer required.** ACMX2 builds and runs on AMD, Intel, and Apple GPUs using the OpenGL/SDL2 path (`-DWITH_CUDA=OFF`). On NVIDIA systems with the CUDA toolkit and an OpenCV 5 build that includes CUDA support, you can opt in to the CUDA GPU-filter stack and FFmpeg CUDA hardware decode by configuring with `-DWITH_CUDA=ON` (the default when CUDA is detected).

[Full Documentation](https://lostsidedead.biz/acmx2/docs/)

[Download the ACMX2 Flatpak](https://lostsidedead.biz/acmx2/release)

**Current release: ACMX2 v2.100.0**

[YouTube Video Tutorial](https://youtu.be/-IDAF8MMmkg)

This project now builds with [pcons](https://pcons.org), a modern
software build tool. It's simpler, cleaner & faster than CMake, since
the build scripts are plain modern python, and it has a nice clean API
and supports lots of languages and tools.

To build with it, just `uvx pcons` if you have
[uv](docs.astral.sh/uv/), or `python -mpip install pcons;
pcons`.

## First Run: Qt Interface and Texture Cache

The Qt launcher stores these choices, so this setup normally needs to be done
only once:

1. Open **File > Properties**. Confirm that **Executable Path** points to the
   `acmx2` binary (the default `acmx2` is sufficient when it is installed in
   `PATH`), then select a shader directory containing `library.json` or the
   legacy `index.txt`. `library.json` is preferred. Also choose the directory
   used for snapshots.
2. Open **Session > Session Properties** and select the input mode, device or
   input file, capture resolution, FPS, and window resolution. A fresh setup
   defaults to `1280x720` camera input and `Default` window resolution.
3. To use the current texture-history shaders, check **Texture Cache** first,
   then check **Use sampler2DArray history**. The second control corresponds to
   `--texture-cache-array` and is disabled until Texture Cache is enabled.
4. Leave **Cache Size** at `8` for the normal shader pack, or choose `1`-`64`
   frames when a shader needs a different history depth. **Cache Delay** is the
   number of frames skipped before the next history update: the UI default of
   `1` updates the cache every second frame, while a larger value spreads the
   stored history farther apart in time. Larger caches use more GPU memory,
   especially at high resolutions.
5. After installing or updating a shader pack, use **Playback > Remove Broken**
   if some shaders fail on the current OpenGL driver. ACMX2 compile-checks the
   library, removes only failing entries from the active manifest, saves the
   original as `library.json.bak` or `index.txt.bak`, and reloads the list. It
   does not delete the shader source files.

To assemble a library from shaders stored in different locations, open
**List > Shader Library Builder...**. Add individual `.glsl` or `.comp` files,
or scan a complete folder with optional subfolder recursion. The builder keeps
the list alphabetically sorted, rejects duplicate and unreadable sources,
shows fragment/compute totals, and can reopen an existing manifest. Exporting
creates a self-contained directory containing safe copies of the listed
shaders and an ordered `library.json`; filename collisions are renamed instead
of overwriting unrelated files. The exported library is loaded into the main
interface automatically. Starting with v2.9.2, the copy and manifest-writing
work runs in the background so larger exports do not block the Qt interface.

Starting with v2.9.1, the Qt launcher identifies shaders by their exact path
relative to the selected library instead of relying on the displayed row
number. Initial selection, live selection, multipass chains, and playlists
therefore continue to load the named effect when a manifest is sorted,
filtered, or contains unsupported compute shaders. Compute shader entries are
kept in `library.json`; on systems without the required OpenGL support they
remain stable passthrough slots rather than being removed. Existing commands
and saved data that use numeric shader indices remain supported.

The maintained [shader collection](https://github.com/lostjared/shaders) uses
the `sampler2DArray history` interface for current cache effects. If one of
these effects is selected without both texture-cache settings, it may fail to
compile or appear not to apply. Legacy cache shaders that use `samp1` through
`samp8` or `textures[SIZE]` can run with Texture Cache enabled and array mode
disabled. On non-macOS systems, **Playback > Rebuild Shader Cache** can be run
after changing the cache size or array mode to precompile the matching shader
variants. The rebuild window remains responsive and shows the ACMX2 logo with
compile progress while a full library is processed. **Playback > Clean Shader
Cache** deletes every cached variant for the selected library without rebuilding
it. macOS always compiles shader source at runtime, so both cache maintenance
actions are unavailable there.

## 🚀 Purpose & Vision
The original project brought a massive library of "glitch" filters to digital artists. However, as resolutions climbed to 4K and filter stacks became more complex, CPU-based processing hit a bottleneck. 

**acidcam-gpu** solves this by:
* **Parallelizing the Chaos:** Running effects on the GPU — GLSL shaders for the core pipeline, with optional CUDA kernels for additional GPU filters on NVIDIA hardware.

## 🛠 Tech Stack
* **Language:** C++20
* **Computer Vision:** OpenCV 5
* **Graphics API:** OpenGL / SDL2 (cross-vendor, hardware-accelerated rendering — works on NVIDIA, AMD, Intel, and Apple GPUs)
* **Optional Parallel Computing:** NVIDIA CUDA (compile-time opt-in via `-DWITH_CUDA=ON`; tested on RTX 2070 and newer)
* **Format Support:** Native **MX2 MXMOD** 3D model parsing for real-time geometry glitching.

## ⚡ Optional NVIDIA / CUDA Acceleration
When built with `-DWITH_CUDA=ON` on a system with an NVIDIA GPU and CUDA-enabled OpenCV 5, ACMX2 can additionally leverage:
* **Shared Memory:** Fast on-chip memory to speed up neighborhood-based CUDA filters.
* **Massive Throughput:** Thousands of CUDA cores to apply multiple glitch layers in a single pass.
* **CUDA/OpenGL Zero-Copy Interop:** High-speed texture sharing between CUDA and OpenGL.
* **FFmpeg CUDA Hardware Decode:** Direct hardware-accelerated decoding for video files.

Without CUDA, all shader-based features continue to work — only the CUDA GPU-filter stack and CUDA-specific decode/encode paths are omitted.

## Project Goals:
* **Zero-Copy Interop:** High-speed texture sharing between CUDA and OpenGL.
* **FFmpeg CUDA Decode Path:** Prefer direct FFmpeg/CUDA hardware decode for video files, with automatic software/OpenCV fallback.
* **Hardware-First Encoding:** Prefer `h264_nvenc` when available, with automatic software H.264 fallback.
* **HDR Video Pipeline:** Detect BT.2020 PQ/HLG sources, process them in HDR, and write HEVC Main10 output with HDR signaling preserved.
* **Encoding Quality Controls:** Preset, tune, CRF, codec mode, and realtime low-latency flags are available for recording.
* **Visual User Interface** Simple to use User interface
* **Command line tool** Command line tool

## Revisions

### Version 2.100.0 (August 2026)

#### August 16

- **Coherent live controls**: the Qt launcher and rendering engine now protect
  their shared-memory control channel with a named POSIX semaphore. The engine
  processes a local snapshot, preventing partially updated shader, audio,
  watermark, uniform, and GPU-filter state from being observed.
- **Linux and macOS IPC support**: synchronization uses `sem_open`, `sem_wait`,
  and `sem_post` rather than process-local synchronization. The named semaphore
  works with the existing `shm_open`/`mmap` channel on Linux and macOS,
  including Apple Silicon.
- **Shared-memory protocol 10**: initialization, compound writes, sequence
  updates, and snapshot reads are all protected. Upgrade and restart the Qt
  launcher and engine together because older protocol versions are rejected.
- **Flatpak release refresh**: the x86_64 OpenGL build includes the Qt
  interface, command-line engine, RtAudio, MIDI support, MIDI mapper,
  Intel IPP-optimized OpenCV, and the pinned shader collection. CUDA is not
  required. The bundle is 22,185,976 bytes with SHA-256
  `252cb2417ba6e78775a149680347333200a40bdea835bf7be329fbf681d07308`.

### Version 2.9.2 (August 2026)

#### August 13

- **Responsive library export**: shader copying and `library.json` generation
  now run on a worker thread, keeping the Qt interface responsive while an
  export is in progress.
- **Full 360-degree 3D look**: free-look pitch now wraps through the complete
  360-degree range instead of stopping at ±89 degrees, with the camera up
  vector following the view orientation across the poles.

### Version 2.9.1 (August 2026)

#### August 12

- **Exact shader identity**: the Qt launcher and ACMX2 now exchange exact
  library-relative filenames for startup and live shader selection, preventing
  the displayed shader name from drifting away from the effect that is loaded.
- **Filename-based multipass chains**: generated commands use `--shader-file`
  and `--shader-pass-files` instead of emitting both old numeric indices and
  filenames. Multipass filenames use a length-prefixed UTF-8 encoding, so no
  separator character—including `:` or a newline—can make the list ambiguous.
- **Playlist compatibility**: playlist entries first resolve as exact relative
  filenames, then fall back to legacy stem matching for older playlists.
- **Shared-memory protocol 9**: live Qt control carries the selected shader
  filename and the filename of every multipass stage while retaining numeric
  fields for compatibility.
- **Compute shaders preserved**: fragment and compute entries remain together
  in `library.json`; unsupported compute programs use the existing passthrough
  behavior rather than changing library ordering.
- **Version 2.9.1 release**: CMake projects, runtime headers, Doxygen output,
  AppStream metadata, and the Flatpak download page now report v2.9.1.

### Version 2.9.0 (August 2026)

#### August 12

- **Direct library loading**: the Qt **File** menu can open a shader library
  directory and provides a persisted **Load Recent** submenu for returning to
  recently used libraries.

### Version 2.8.0 (August 2026)

#### August 10

- **Shader Library Builder**: **List > Shader Library Builder...** provides a
  theme-aware workflow for combining `.glsl` and `.comp` files from individual
  paths or recursively scanned folders. Lists stay alphabetically sorted and
  include duplicate, extension, readability, and missing-source validation.
- **Portable `library.json` export**: the builder can reopen existing
  manifests and export a self-contained shader directory. It copies sources
  safely, assigns unique names when basenames collide, preserves unrelated
  files in new destination folders, writes the displayed order to
  `library.json`, and immediately loads the result in the main interface.
- **Correct Default video window sizing**: ACMX2 probes video stream dimensions
  before constructing the SDL/OpenGL window when Window Resolution is
  `Default`. Low-resolution videos now open at their native size, including in
  the Flatpak build where resizing after context creation was unreliable.
- **Complete interface build inputs**: the PCons interface target now includes
  the built-in uniform reference and Shader Library Builder implementations;
  CMake includes the builder sources as well.
- **Version 2.8.0 metadata**: engine and launcher CMake metadata, runtime
  version headers, and versioned Doxygen output now identify 2.8.0.

### Version 2.7.0 (August 2026)

#### August 9

- **OpenGL compute shaders**: Linux builds now request an OpenGL 4.3 context
  when available and run `.comp` files as full-frame image passes. Compute and
  fragment shaders can share libraries, playlists, and multipass chains; an
  OpenGL 4.1 fallback keeps unsupported compute slots as passthrough entries so
  shader indices remain stable.
- **Compute-aware shader caching**: the version 4 whole-library cache records
  each shader's program type and stores compute program binaries alongside
  fragment programs. Source and driver-specific per-program caching also
  avoids relinking unchanged compute shaders.
- **Compute authoring in Qt**: New Shader File can create fragment or compute
  shaders, Find in Files and Save As recognize `.comp`, and the shader list has
  a Type column that distinguishes Compute from Fragment entries.
- **Built-in uniform reference**: **Help > Built-in Uniform Reference...** opens
  a searchable, modeless guide to runtime uniforms, including types,
  availability, declarations, descriptions, and GLSL examples. Engine startup
  also reports the active OpenGL uniform limits.
- **Faster multipass editing**: **Insert** replaces the selected multipass slot
  with the currently chosen shader without removing and re-adding the pass.
- **Cleaner Linux camera selection**: the Settings dialog hides secondary V4L2
  nodes exposed by the same physical device and maps a saved secondary-node
  selection back to its primary node when possible.
- **Version 2.7.0 metadata**: engine and launcher CMake metadata, runtime version
  headers, and versioned Doxygen output now identify 2.7.0.

### Version 2.6.1 (August 2026)

#### August 7

- **Theme-aware shader editor**: the current-line background is derived from
  the active editor palette instead of using a fixed dark color. GLSL syntax
  colors switch between high-contrast light and dark palettes and refresh when
  the interface theme changes.
- **Optimized Flatpak OpenCV build**: the x86_64 Flatpak builds OpenCV 4.12.0
  with Intel IPP 2022.1.0 and its integration wrappers enabled. OpenCV also
  retains runtime CPU dispatch for SSE4, AVX, AVX2/FMA, and AVX-512 where the
  host supports them; the package remains an OpenGL, no-CUDA build.
- **Flatpak release refresh**: the 2.6.1 bundle, AppStream metadata, download
  size, SHA-256 checksum, and documentation link are current on the release
  page.
- **Reproducible CUDA container builds**: the Arch Podman image accepts a
  `CUDA_ARCHITECTURES` build argument (default `75`), installs every component
  under `/usr/local`, verifies the installed executables, and launches
  `acmx2_interface` from `PATH`.
- **Current container assets and dependencies**: the image uses Arch's
  `sdl2-compat`, clones the maintained shader repository, downloads and
  extracts the model pack, and exports the runtime shader/data paths.
- **Repository-local MXWrite for acidcam-gpu**: the standalone CUDA project
  builds the sibling `MXWrite/` source tree for the `acidcam` CLI. Consumers of
  the installed `acidcam-gpu::acidcam-gpu` target inherit only its public
  OpenCV and CUDA dependencies; a separately installed MXWrite package is no
  longer required.

### Version 2.6.0 (August 2026)

#### August 6

- **M3U audio playlist playback**: `--audio-file` now accepts `.m3u` and
  `.m3u8` playlists. Entries play sequentially, relative paths resolve from
  the playlist directory, unusable entries are skipped, and recorded output
  is muxed in the same track order. Repeat loops the complete playlist; Audio
  Truncate stops the video only after the final track.
- **Audio playlist editor**: Audio Settings includes a **Create / Edit** M3U
  editor for adding and removing tracks, multi-selection reordering,
  drag-and-drop, sorting, shuffling, opening existing playlists, and Save/Save
  As. Saved playlists use portable relative paths where possible.
- **Unambiguous file-audio controls**: single-file and M3U sources are mutually
  exclusive. Selecting either source automatically enables file playback and
  its output-device selector, while retaining the selected output device and
  other audio settings between sessions.
- **Current-track HUD**: audible file and M3U playback displays
  `Track: filename.ext` in purple without exposing the containing path. The
  label advances automatically at playlist track boundaries.
- **Repeatable file audio**: `--audio-repeat` restarts an audio file at EOF for
  continuous reactive playback. The Qt Audio Settings window exposes a
  persistent Repeat checkbox, keeps it mutually exclusive with Audio Truncate,
  and forwards the option to both preview and recording runs.
- **DeepDream model bundle**: the repository now includes `ddream.onnx` plus
  YAML presets for 256, 512, 768, and 1024-pixel inference. The sized presets
  use dynamic, four-pixel-aligned input dimensions so the same model can be
  selected at different quality and performance points.
- **Faster generic ONNX display path**: CUDA builds can keep ONNX
  normalization, optional bilateral smoothing, resize, and RGBA conversion in
  GPU memory, then send the result directly to the OpenGL texture uploader.
  Models or configurations unsupported by that route automatically retain the
  portable CPU fallback, and CUDA filters continue to use their required host
  frame path.
- **CUDA ONNX correctness and synchronization**: multi-channel model output is
  flattened for global min/max normalization before its RGB shape is restored,
  matching CPU behavior. A CUDA event now orders default-stream processing
  before the asynchronous CUDA-to-OpenGL upload stream reads the frame.
- **Version 2.6.0 metadata**: CMake, runtime version headers, Doxygen output,
  release documentation, and Flatpak release metadata now identify 2.6.0.

#### August 5

- **Silent still-image rendering**: `--silent` now accepts `-g/--graphic`
  input as well as video. Still images require `-o/--output` and a positive
  `--duration`; rendering uses the off-screen context, runs without display
  pacing, reports periodic encoded progress, and finishes with a 100% update.
- **Deterministic normalized shader time**: `--normalized` advances `time_f`
  by `time-speed / FPS` for each rendered frame instead of elapsed wall time.
  Qt exposes the persistent **Playback > Normalized Time** action and can
  toggle it live through the existing shared-memory runtime channel.

- **Persistent launch settings**: camera/video/graphics mode, camera device,
  input and output files, capture and window resolution, FPS, recording,
  rotation, cache, ONNX, 3D, and encoding selections are restored when the Qt
  launcher is closed and started again instead of reverting to camera 0.
- **Safe restored dimensions**: malformed or missing saved resolutions fall
  back to valid defaults, preventing invalid `4294967295x4294967295` encoder
  dimensions when opening an output video.
- **Native graphics window sizing**: graphics mode probes the selected image
  before creating the SDL/OpenGL window, so the initial window matches the
  image dimensions when Window Resolution is `Default`.
- **OpenCV DNN capability reporting**: the interface startup log reports
  whether the selected `acmx2` build includes OpenCV DNN. When it is disabled,
  the Session Settings ONNX controls are disabled and any saved ONNX selection
  is ignored.
- **Responsive full cache rebuilds**: shader-cache rebuilding pumps window
  events and displays the ACMX2 loading graphic with per-library progress so
  desktop environments do not report the application as unresponsive.
- **Clean Shader Cache**: **Playback > Clean Shader Cache** removes current and
  legacy cache files for every cache-size/array-mode variant without starting
  a rebuild.
- **Library-defined custom uniforms**: JSON shader libraries can define up to 64 custom `float` uniforms with minimum, maximum, step, and current values. The Qt **List > Add Custom Uniforms...** dialog creates sliders for them, saves them in `library.json`, and sends value changes to a running ACMX2 process without restarting it.
- **Automatic custom-uniform declarations**: when a shader references a manifest-defined custom uniform but does not declare it, ACMX2 injects the required `uniform float` declaration before compilation. Adding or removing a definition reloads the current shader; ordinary slider changes only update the live value.
- **Incremental shader-cache refresh**: on platforms with persistent program-binary caching, startup validation recompiles only shader entries whose prepared source has changed, then writes those entries back to the existing cache instead of rebuilding the entire library.
- **JSON shader-library manifests**: `library.json` is now preferred over `index.txt`, the Qt interface can migrate text-only libraries automatically, and `convert-index-to-json.pl` provides standalone conversion.
- **Live shader coding**: saving in the Qt editor recompiles and replaces only the edited shader while ACMX2 keeps rendering; a failed compile leaves the previous valid program active and reports the full diagnostic.
- **Editor themes and search**: the shader editor now includes 25 built-in color themes and **List > Find in Files** (`Ctrl+Shift+F`) for regular-expression searches across GLSL source.
- **Lossless and custom encoder settings**: the Qt interface exposes H.264/HEVC NVENC modes, `p1`-`p7` presets, lossless tuning, and extra FFmpeg parameters through `--encode-params`.
- **Real-time webcam recording synchronization**: webcam output now follows wall-clock presentation timestamps from the first valid camera frame. If rendering or encoding falls behind, late frames are dropped and timestamp gaps preserve the correct playback speed instead of stretching the recording. This path works with audio disabled, with live pass-through, and with recorded audio.
- **Webcam No Drop behavior clarified**: `--no-drop` is now limited to video-file and graphics processing, where slowing production is safe. Webcam mode always uses timestamp-based late-frame dropping; the CLI ignores `--no-drop`, and the Qt Settings dialog unchecks and disables No Drop when Camera is selected.
- **Recorded webcam audio boundary**: live WAV capture starts with the media timeline and stops at the video-capture boundary, before queued video frames and encoder packets drain. Timestamped webcam video is muxed without rescaling its PTS, preventing an audio-only tail or a playback-speed change.
- **Editable window resolution**: the Qt Window Resolution control keeps its preset dropdown but also accepts a custom `WxH` value. Custom width and height must be positive even integers; invalid values show a warning and are not applied or saved.
- **Input-frame rotation**: Qt Settings now provides a Rotate checkbox with 90° clockwise, 180°, and 90° counterclockwise choices. CPU builds use `cv::rotate`; CUDA builds use `cv::cuda::rotate` on a `GpuMat`. With the default Window Resolution, 90° rotation swaps the output dimensions to preserve orientation.
- **High-frame-rate and loopback capture**: Linux `v4l2loopback` devices now expose common choices from 24 through 240 FPS in camera enumeration. ACMX2 preserves the requested loopback rate when the driver reports its stale producer rate, and requests a non-vsync window when the selected rate is above 60 FPS so 90/120/144/240 FPS capture is not capped by the display refresh path.
- **High-frame-rate video playback**: video-file input above 60 FPS now disables the SDL swap interval so processing is not capped at the display's 60 Hz vsync rate.
- **Modeless playlist and multipass dialogs**: Shader Playlist Settings and Multipass Shader Settings can remain open while the main interface is used. Reopening an existing dialog focuses it and refreshes its shader list instead of creating a duplicate.
- **Open multipass shaders in the editor**: double-click a shader in the selected multipass list to open or focus it in the built-in GLSL editor.
- **Unified dependency helper**: the root `install-required.sh` installs dependencies on Arch-based Linux or macOS. Arch selects `opencv-cuda` when NVIDIA hardware is detected and stock `opencv` otherwise; macOS uses Homebrew and builds without CUDA.

### July 2026

- **Audio spectrum history array**: `--enable-audio-buffers <N>` allocates one runtime-sized `sampler1DArray` for rolling FFT history, limited only by the GPU's array-layer limit.
- **Audio startup warmup envelope**: new `--audio-warm-rate <value>` option fades audio-reactive uniforms/spectrum from 0 to full strength at startup (default `0.5` 1/sec, about 2 seconds).
- **Camera/file A/V startup sync hardening**: cache and writer paths now include a startup warmup window so loading-screen frames are not pushed into `samp1..samp8`, and early audio/file processing is held until warmup completes.
- **Texture cache behavior update**: `--texture-cache` now works in camera, video, and graphic input modes (not video-only).
- **Texture cache array mode**: `--texture-cache-array` enables the cache and exposes frame history through one `sampler2DArray history` ring; the maintained [shader collection](https://github.com/lostjared/shaders) has been updated for this interface.
- **Shader time wrap stability fix**: `time_f` wrap/reset behavior now uses a large `2*PI` multiple to preserve long-running shader phase continuity.
- **Qt camera FPS persistence improvement**: preferred camera FPS is now saved/restored and retained across resolution/device repopulation when supported.
- **Headless and terminal workflow updates**: improved silent/headless processing behavior, terminal color-coded output, and related CLI flow refinements.
- **HDR pipeline refinements**: recent HLG-to-HDR10 conversion work and continued HDR + silent mode stabilization.
- **Startup logo splash**: `data/logo.png` is displayed on launch before the shader pipeline begins, with a smooth fade-out.
- **Watermark overlay**: embed a custom text watermark (color-configurable via RGB) in recorded video using `--use-watermark` and `--use-watermark-color`; also accessible from the Qt Playback menu with a live color preview.
- **Display-filter overlay**: `--display-filter` renders the active shader name, multipass stack, and GPU filter list in the upper-left corner of both the live window and the recorded output.
- **Autopilot random interval mode**: `--autopilot-random <N>` randomizes the frame interval between auto-switches (range 4..N) for more organic live performance variation.
- **Sequential autopilot mode**: Y key cycles the playlist in strict order rather than randomly when autopilot is active.
- **Post-multipass shader navigation**: Shift+Up / Shift+Down changes the shader that runs after the multipass chain without altering the playlist position.
- **Qt interface improvements**: Watermark Settings dialog with text input and color picker; Display Filter toggle in the Playback menu; autopilot random interval persisted in Qt session settings; YUV format options refresh automatically when the camera device changes.
- **Editor and shader workflow improvements**: shader reload support, shader cache rebuild path, and safer editor close behavior with save prompts.
- **Qt interface productivity features**: metadata viewer integration, settings text/scaling tweaks, and command edit/copy/run improvements.
- **Playlist and live-control workflow**: shuffle/concat/clear playlist actions, combined playlist updates, and keyboard/autopilot navigation improvements.
- **Output and recording path updates**: MKV output support, additional format handling, no-drop frame path work, and SDR TIFF/WebP snapshot behavior updates.
- **Audio-file recording behavior hardening**: explicit opt-in handling for file-mode audio recording/mux behavior.

### May 2026

#### DNN & ONNX Model Expansion

The ONNX/DNN system has been significantly expanded with new pre-trained models and improved inference performance:

- **FP16 Optimization**: DNN inference now uses FP16 (half-precision floating-point) for faster computation on supported hardware while maintaining visual quality.
- **New ONNX Models**:
  - **Bubble Effect** (`bubble.yaml` / `bubble.onnx`) — Creates bubble distortion effects for surreal imagery
  - **Cartoon Effect** — Converts video/images to a cartoon-like aesthetic with edge enhancement
  - **Color Splash** — Selectively desaturates the image while preserving selected color channels
  - **Pencil Sketch** — Generates pencil sketch-like renderings from video input
  - **Custom Style Transfer** — First custom neural-style-transfer model for artistic transformation
  - **Edge Detection** — Deep neural network-based edge detection (superior to traditional methods)
- **Generic ONNX/YAML File Support**: Load any ONNX model via YAML configuration file without recompilation. The YAML format specifies:
  - Model path and preprocessing parameters (input size, scale, swap RGB/BGR)
  - Integration with the `--edge` and `--human` options for extensibility
- **Settings Window Improvements**: Settings dialog now has increased height for better visibility of longer model lists.

#### Generate Mode & Randomization

New features for spontaneous creative generation:

- **Generate Mode** (`--generate`): New processing mode that generates random effects and shader combinations from a seed or random state, enabling algorithmic art creation without manual control.
- **Random Generate in Interface**: The Qt interface now includes a "Generate" button that randomizes effect parameters and creates new artistic variations in real-time.
- **Audio Animation Mux**: Embeds an animated audio track during file processing so you know the application is still processing and not frozen. Useful for long batch operations.

#### Output Format Enhancements

Expanded output format support for modern workflows:

- **PNG Frame Writing**: Write individual frames as PNG files instead of video, enabling frame-by-frame processing and archival. Use `--png` in video mode to enable PNG frame output instead of video encoding.
- **PNG Snapshot Mode**: Save snapshots and output directly as PNG images with lossless quality.

#### Color & Tone Adjustment

Fine-grained color control additions:

- **Black Point Control** (`--black <point>`): Precise shadow crush and black level correction (default: 0.35)
- **White Point Control** (`--white <point>`): Precise opacity saturation and white level adjustment (default: 0.75)

### Version 2.12.0

- Video-file decode prefers direct FFmpeg CUDA acceleration when available and retains automatic software/OpenCV fallback.
- Startup reports the active decode and encode paths.
- WAV capture and recorded-audio mux cleanup were hardened across file, image, and camera modes.

### Version 2.7.0

- Updated build scripts and Podman configuration.
- Audio muxing uses the video duration for more precise synchronization.
- Fixed audio-track copying for completed recordings.

## 📦 Installation & Environment
This project is developed and tested on **Bazzite Linux** using **Arch Linux** containers via **Distrobox**, but it builds on any modern Linux distribution as well as macOS.

### Flatpak download

For the portable Linux build with the Qt interface, RtAudio, MIDI support, and no
CUDA requirement, visit the [ACMX2 Flatpak download page](https://lostsidedead.biz/acmx2/release).
This x86_64 package includes an Intel IPP-optimized OpenCV build for accelerated
CPU image processing on supported processors.

The current bundle is **ACMX2 v2.100.0** (22,185,976 bytes). Verify it before
installation with:

```bash
sha256sum ACMX2.flatpak
# 252cb2417ba6e78775a149680347333200a40bdea835bf7be329fbf681d07308  ACMX2.flatpak
flatpak install --user --reinstall ./ACMX2.flatpak
```

### Prerequisites
* **GPU:** Any GPU with working OpenGL 3.3+ drivers (NVIDIA, AMD, Intel, or Apple Silicon). NVIDIA hardware is **optional** and only required if you want to enable the CUDA GPU-filter stack at compile time.
* **Drivers:** Up-to-date GPU drivers for your platform. NVIDIA proprietary drivers (v535+) are required only when building with `-DWITH_CUDA=ON`.
* **Environment:** Arch Linux (or compatible). Install all dependencies via `pacman`:

**Build Tools:**
```bash
sudo pacman -S --needed base-devel git cmake ninja pkg-config curl unzip
```

**NVIDIA & CUDA (optional — only needed for `-DWITH_CUDA=ON`):**
```bash
sudo pacman -S --needed nvidia-utils cuda
```

**OpenCV 5:**

The CUDA toolkit does not add CUDA support to an existing OpenCV installation.
For `-DWITH_CUDA=ON`, OpenCV itself must have been compiled with CUDA enabled
and must provide the CUDA modules required by ACMX2, including `cudaimgproc`.
On Arch Linux, `opencv-cuda` provides this build and is installed instead of
the conflicting stock `opencv` package. Use stock `opencv` only with
`-DWITH_CUDA=OFF`.

```bash
# Portable OpenGL build (-DWITH_CUDA=OFF):
sudo pacman -S --needed opencv hdf5 vtk fmt glew

# CUDA build (-DWITH_CUDA=ON); install instead of stock opencv:
sudo pacman -S --needed opencv-cuda hdf5 vtk fmt glew
```

**SDL2 & Qt6:**
```bash
sudo pacman -S --needed sdl2 sdl2_ttf sdl2_mixer sdl2_image qt6-base qt6-tools qt6-multimedia
```

**Graphics, Audio & Media Libraries:**
```bash
sudo pacman -S --needed glm mesa libglvnd ffmpeg rtaudio pulseaudio libpulse libjpeg-turbo libpng
```

**ONNX/DNN Support (optional — only needed for `-DWITH_OPENCV_DNN=ON`):**
```bash
sudo pacman -S --needed yaml-cpp
```

**libmx2 (built from source):**
```bash
git clone https://github.com/lostjared/libmx2.git
cd libmx2/libmx
mkdir build && cd build
cmake .. -DEXAMPLES=OFF -DOPENGL=ON
make -j$(nproc)
sudo make install
```

**Fonts:**
```bash
sudo pacman -S --needed ttf-dejavu ttf-liberation noto-fonts
```

Or install everything at once using the provided script:
```bash
sudo bash build-script/install-deps-arch.sh
```

For a cross-platform dependency-only setup, run the root helper instead:

```bash
./install-required.sh
```

On Arch-based Linux the helper uses `pacman` (and `sudo` when needed), detects
NVIDIA hardware, and selects `opencv-cuda` or stock `opencv` accordingly. On
macOS it uses Homebrew and configures the project without CUDA. The helper does
not install `libmx2`; build the current `libmx2/libmx` source first using the
steps above. ACMX2's high-frame-rate path requires the current `libmx2`
`GLWindow` constructor with explicit vsync control.

---

# Technical Documentation:

[Project Documentation](https://lostsidedead.biz/acmx2-explained.html)

[GPU Filters Explained](https://lostsidedead.biz/acmx2/filter_browser.html)
 
[Example Shaders](https://lostsidedead.biz/acmx2/shader_browser.html)

---

## Command-Line Arguments

### General

| Short | Long | Value | Description |
|-------|------|-------|-------------|
| `-v` | `--help` | | Display help message and exit |
| `-p` | `--path` | `<dir>` | Assets path |
| `-r` | `--resolution` | `WxH` | Window resolution (e.g. `1920x1080`) |
| | `--rotate` | `<clockwise\|180\|counterclockwise>` | Rotate input frames before DNN, CUDA-filter, and shader processing |
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
| | `--duration` | `<seconds>` | Recording duration limit in seconds (float); stop recording and exit after elapsed |
| | `--encode-preset` | `<name>` | Encoder preset: `ultrafast`..`veryslow` |
| | `--encode-tune` | `<name>` | Encoder tune: `none`, `film`, `animation`, `grain`, `stillimage`, `psnr`, `ssim`, `fastdecode`, `zerolatency` |
| | `--encode-crf` | `<0-51>` | Encoder CRF quality override (default: `18`) |
| | `--encode-codec` | `<name>` | Encoder policy (`auto`, `software`, `nvenc`) or exact installed FFmpeg encoder name |
| | `--list-encoders` | | List video encoders available through MXWrite |
| | `--list-encoder-options` | `<name>` | List FFmpeg AVOptions exposed by one encoder |
| | `--encode-realtime` | | Enable low-latency realtime encoding flags |
| | `--no-drop` | | Video-file/graphics processing: never drop frames and pace processing to encoder throughput; ignored in webcam mode |
| | `--use-watermark` | `<text>` | Embed a text watermark (upper-left) into recorded video |
| | `--use-watermark-color` | `<r,g,b>` | Watermark text color as 0-255 RGB components (default: `255,0,150`) |
| | `--display-filter` | | Show active shader/stack/GPU filter in upper-left corner of window and recording |

### Shader Options

| Short | Long | Value | Description |
|-------|------|-------|-------------|
| `-s` | `--shaders` | `<directory>` | Shader library directory (`library.json` preferred, `index.txt` fallback) |
| `-f` | `--fragment` | `<file>` | Single fragment shader file |
| | `--shader-file` | `<relative-file>` | Initial shader by exact library-relative filename (preferred) |
| | `--shader` | `<index>` | Initial shader by legacy library index |
| | `--shader-pass-files` | `<encoded-files>` | Exact multipass filenames encoded as repeated UTF-8 byte-length/name pairs, such as `13:4ac_rand.glsl14:addup_cos.glsl` |
| | `--shader-pass` | `<indices>` | Legacy shader pass indices (comma-separated, e.g. `0,1,2`) |
| | `--playlist` | `<file>` | Shader playlist text file; exact relative filenames are preferred, with legacy stem matching as a fallback |
| | `--autopilot-frames` | `<N>` | Auto-switch to a random playlist node every N rendered frames (minimum 4) |
| | `--autopilot-timeout` | `<N>` | Alias for `--autopilot-frames` |
| | `--autopilot-random` | `<N>` | Randomize autopilot interval to a value in the range 4..N frames for each auto-switch |
| | `--build` | `<path>` | Build shader cache for specified library path and exit |
| | `--no-cache` | | Disable shader caching (always recompile shaders) |
| | `--time-speed` | `<float>` | Constant `time_f` speed multiplier (default: `1.0`) |
| | `--normalized` | | Advance `time_f` by `time-speed / FPS` per rendered frame instead of elapsed wall time |
| | `--cross-fade` | `<seconds>` | Crossfade duration in seconds when switching playlist shaders (default: `0.5`) |

### GPU Filter Options

| Long | Value | Description |
|------|-------|-------------|
| `--gpu-filter` | `<indices>` | GPU filter indices (comma-separated) |
| `--gpu-buffer` | `<size>` | GPU frame buffer size (`4`–`32`) |
| `--list-filters` | | List available GPU filters and exit |
| `--list-cuda-devices` | | List available CUDA devices and exit |
| `--enumerate-device` | `<index>` | List supported resolutions and frame rates for a camera device (Linux only) |
| `--disable-counter` | | Disable timer and FPS counter overlay |
| `--silent` | | Process video or a graphics file without a window. Valid with `-i/--input` video or `-g/--graphic` image input and requires `-o/--output`; graphics input also requires a positive `--duration`, and camera input is rejected. |

### Texture Cache Options

| Long | Value | Description |
|------|-------|-------------|
| `--texture-cache` | | Enable texture cache (camera, video, and graphic modes) |
| `--cache-delay` | `<frames>` | Texture cache delay in frames |
| `--texture-cache-size` | `<frames>` | Texture cache ring size (`1`–`64`, default `8`) |

### Texture Cache Array Options

| Long | Value | Description |
|------|-------|-------------|
| `--texture-cache-array` | | Enable the cache and bind history as one `sampler2DArray history` ring |

### 3D / Model Options

| Long | Value | Description |
|------|-------|-------------|
| `--enable-3d` | | Enable 3D cube rendering |
| `--model` | `<file>` | 3D model file (`.mxmod`) |

### Recording and Output Options

| Long | Value | Description |
|------|-------|-------------|
| `--copy-audio` | | Copy audio track from input to output |
| `--png` | | Save output frames as PNG files instead of video encoding (use with `-o/--output`) |
| `--generate` | `<interval>` | Save a PNG image at the specified frame interval |

### ONNX Model Options

| Long | Value | Description |
|------|-------|-------------|
| `--edge` | `<file>` | DNN Model for edge detection file you want is: edge_detection_dexined_2024sep.onnx |
| `--human` | `<file>` | DNN Model for human detection file you want is: human_segmentation_pphumanseg_2023mar.onnx |
| `--background` | | Enable background processing on --human |
| `--onnx` | `<file>` | Load ONNX model from YAML configuration file (specifies model path and preprocessing parameters) |
| `--black` | `<point>` | Mask black point / shadow crush threshold for color adjustment (default: 0.35) |
| `--white` | `<point>` | Mask white point / opacity saturation threshold for color adjustment (default: 0.75) |

The bundled DeepDream model can be selected at several inference sizes. Larger
presets retain more detail but require more processing time and GPU memory:

```bash
./acmx2 -i input.mp4 --onnx ../models/ddream-512.yaml -o output.mp4
./acmx2 -g image.png --onnx ../models/ddream-1024.yaml --silent --duration 10 -o output.mp4
```

### Audio Options (requires `AUDIO_ENABLED` build)

| Short | Long | Value | Description |
|-------|------|-------|-------------|
| `-w` | `--enable-audio` | | Enable audio reactivity |
| `-l` | `--channels` | `<num>` | Audio channels |
| `-q` | `--sense` | `<float>` | Audio sensitivity |
| `-y` | `--pass-through` | | Play live input or file audio through the selected output device; this does not by itself record or mux audio |
| | `--audio-input` | `<index>` | Audio input device (`default` or index) |
| | `--audio-output` | `<index>` | Audio output device (`default` or index) |
| | `--list-devices` | | List audio devices and exit |
| | `--record-audio` | `<file>` | Record captured audio to WAV, mux it in sync with video, then remove it after a successful mux |
| | `--record-gain` | `<float>` | Recording volume gain `0.0`–`2.0` (default: `1.0`) |
| | `--audio-file` | `<file>` | Use an audio file or M3U/M3U8 playlist for shader reactivity instead of microphone input |
| | `--audio-trunc` | | Stop playback when the audio file reaches EOF |
| | `--audio-repeat` | | Restart audio-file playback, or the complete M3U playlist, at EOF; mutually exclusive with `--audio-trunc` in Qt |
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

ACMX2 automatically switches to its HDR pipeline when the input stream is tagged as BT.2020 with PQ (`SMPTE ST.2084`) or HLG (`ARIB STD-B67`) transfer characteristics, or when the decoded source is a 10-bit BT.2020 format such as `yuv420p10le` or `p010le`.

On ingest, the program preserves the source HDR code values long enough to upload each frame into a 16-bit RGBA working texture. A dedicated HDR decode shader converts PQ or HLG into linear BT.2020 before your GLSL shader passes and does not run the CUDA filters since they are tied to 8 bit RGBA. After processing, a matching HDR encode shader converts the result back into the same HDR transfer family as the source.

HDR exports use HEVC Main10 rather than the standard H.264 path. MXWrite receives BT.2020 HDR frames and writes a 10-bit `P010` HEVC stream with BT.2020 primaries and the original PQ or HLG transfer preserved. When the source includes mastering-display or content-light metadata, ACMX2 forwards that HDR metadata to the output stream as well.

In practice, SDR jobs still follow the H.264-first path, while HDR jobs are written as HDR HEVC so compatible players and editors continue to recognize the output as HDR. Startup logs report this explicitly with lines such as `HDR output mode enabled: HEVC Main10 + BT.2020 + HLG`.

## Using `--silent` in the Terminal

`--silent` is the headless batch-processing mode for video and graphics files. It creates an off-screen OpenGL context, suppresses the visible SDL window, and skips display pacing so the file is processed as fast as decode, effects, and encode allow. Graphics input has no natural end, so silent graphics mode requires a positive `--duration`.

While recording, headless progress lines include the current encoded output file size after the elapsed time. Graphics-file mode prints progress after about one second of output frames or 500 ms of wall time, whichever occurs first, followed by a final 100% update.

- Use it with `-i/--input` video files or `-g/--graphic` image files.
- Always pair it with `-o/--output`.
- Pair graphics input with a positive `--duration`.
- Do not use it with camera capture.
- Audio copy and mux steps still run after frame processing when you use options such as `--copy-audio`, `--audio-file`, or `--record-audio`.

Typical terminal usage:

```bash
./acmx2 -p ./data -i input.mp4 -s ./shaders -h 12 --silent -o output.mp4
```

Render ten seconds from a still image:

```bash
./acmx2 -p ./data -g image.png -s ./shaders -h 12 --silent --duration 10 -o output.mp4
```

Silent HDR processing uses the same flag and keeps HDR active automatically:

```bash
./acmx2 -p ./data -i hdr_input.mp4 -s ./shaders --gpu-filter 0,5 --silent -o hdr_output.mp4 --copy-audio
```

Because headless mode writes newline-delimited progress updates to stdout, it works well with terminal logging:

```bash
./acmx2 -p ./data -i input.mp4 -s ./shaders --silent -o output.mp4 | tee batch.log
```

---

## Keyboard Controls

### General Controls

| Key | Action |
|-----|--------|
| `Up` | Previous shader (or previous playlist tree node if playlist enabled) |
| `Down` | Next shader (or next playlist tree node if playlist enabled) |
| `Shift+Up` | In playlist/autopilot mode: step the post-multipass shader backward without moving the playlist position |
| `Shift+Down` | In playlist/autopilot mode: step the post-multipass shader forward without moving the playlist position |
| `J` | Toggle autopilot mode (requires playlist; randomly auto-advances through playlist nodes at the configured frame interval) |
| `Y` | Toggle sequential autopilot (cycles playlist in order instead of randomly; requires playlist and autopilot active) |
| `R` | Toggle random multipass mode (generates random 1–5 shader chain with crossfade; press again to restore previous state) |
| `G` | Generate a new random shader chain (while in random multipass mode) |
| `Left` | Previous GPU filter (if GPU filters enabled) |
| `Right` | Next GPU filter (if GPU filters enabled) |
| `Space` | Toggle shader processing bypass |
| `P` | Toggle playlist mode / Pause video (Video/Image modes) |
| `L` | Toggle video freeze (Video/Image modes) |
| `Z` | Take snapshot |
| `4` | Take 16-bit HDR TIFF snapshot (HDR input only; requires `ACMX2_WITH_TIFF` build) |
| `5` | Take HDR snapshot — lossless RGBA WebP if built with `ACMX2_WITH_WEBP`, otherwise 16-bit RGBA PNG (HDR input only) |
| `6` | Take raw RGBA snapshot — 16-bit RGBA (8 bytes/pixel) in HDR mode, 8-bit RGBA (4 bytes/pixel) otherwise |
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
| `Page Up/Down` | Increase/Decrease Time Speed |


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

### Playing back raw HDR snapshots with ffplay

The `6` key writes a headerless RGBA byte stream. The pixel layout depends on
whether the input is HDR. Use the dimensions embedded in the snapshot's
filename (ACMX2 always writes `WxH` into the name):

```sh
# HDR raw: 16-bit RGBA (8 bytes/pixel), BT.2020 PQ/HLG
ffplay -f rawvideo -pixel_format rgba64le -video_size 1920x1080 \
       ACMX2.HDR.Snapshot-YYYY.MM.DD-HH.MM.SS-1920x1080-1.raw

# SDR raw: 8-bit RGBA (4 bytes/pixel)
ffplay -f rawvideo -pixel_format rgba -video_size 1920x1080 \
       ACMX2.Raw-YYYY.MM.DD-HH.MM.SS-1920x1080-1.raw
```

Note: ffplay shows raw PQ/HLG values without HDR tone-mapping, so colors may
look saturated/clipped on SDR monitors.

---

## Feature Guides

### Random Multipass Mode

ACMX2 now supports a random multipass mode for spontaneous creative exploration.

- **R key** — Toggle random multipass mode. On entry, the current shader state is saved and a random chain of 1–5 shaders is generated with a crossfade transition. Press R again to crossfade back to the previous state.
- **G key** — While in random mode, generate a new random shader chain with crossfade.
- **Up/Down keys** — While in random mode, change the main (post-processing) shader with crossfade while keeping the random pass list intact.
- **Shift+Up/Down keys** — In playlist or autopilot mode, cycle the post-multipass shader backward/forward independently of the playlist position, with crossfade.
- **MIDI support** — R (code 82) and G (code 71) are available as MIDI-mappable actions in the MIDI Map Tool.

### Crossfade Transitions

ACMX2 now supports smooth crossfade transitions when switching shaders during playlist playback.

- **Command line:** Use `--cross-fade <seconds>` to set the transition duration (default: `0.5` seconds).
- **How it works:** When the active shader changes (via playlist navigation or keyboard controls), the previous frame is captured and linearly blended with the new shader output over the configured duration using a dedicated GLSL crossfade shader.
- **Qt Interface:** The **Settings** dialog includes a "Crossfade Duration" spin box (0.0–10.0 seconds, step 0.1).
- **Implementation:** A separate FBO and shader program (`crossfade.glsl`) perform the blend. The `fade_alpha` uniform ramps from 0 to 1 over the configured duration, mixing the previous and current textures via `mix(prev, curr, fade_alpha)`.

### Encoding Quality Controls

Recording now exposes more detailed encoder controls in both the command line and Qt launcher.

- **Command line options:**
  - `--encode-preset <name>` — preset from `ultrafast` through `veryslow`
  - `--encode-tune <name>` — tune from `none`, `film`, `animation`, `grain`, `stillimage`, `psnr`, `ssim`, `fastdecode`, or `zerolatency`
  - `--encode-crf <0-51>` — explicit CRF quality control (default: `18`)
  - `--encode-codec <name>` — choose `auto`, `software`, `nvenc`, or an exact encoder such as `libx264`, `libx265`, `libsvtav1`, `h264_qsv`, or `hevc_vaapi`
  - `--list-encoders` — show the video encoders registered by the FFmpeg libraries linked to MXWrite
  - `--list-encoder-options <name>` — show option names, types, defaults, ranges, named values, and descriptions for one encoder
  - `--encode-realtime` — enable low-latency realtime encoding flags for live capture
- **Qt Interface:** The **Settings** dialog discovers installed encoders, keeps `libx264` and `libx265` separate, labels hardware/software backends, shows supported pixel formats, and provides an option table. Double-clicking an option adds it to the extra FFmpeg parameters field.
- **Persistence:** Encoding selections are stored and restored for later sessions.
- **Webcam timing:** Camera recording always uses a real-time timestamp clock. Frames that arrive in an already-used nominal FPS slot are discarded, while gaps in PTS preserve elapsed time and keep the result at the correct speed even when the renderer or encoder cannot sustain every frame.
- **No Drop scope:** No Drop remains available for video-file and graphics inputs, where processing can safely wait for encoder capacity. Selecting Camera in the Qt Settings dialog unchecks and disables it; command-line webcam mode also ignores `--no-drop`.
- **Audio behavior:** Pass-through plays audio but does not add it to the output file. With `--record-audio`, capture begins on the first valid source frame, stops with video capture, and is muxed without changing timestamped webcam video speed.
- **Custom window size:** The Window Resolution combo remains a preset selector and is also editable. Enter `Default` or an even `WxH` value such as `1920x1080`; malformed, zero, negative, or odd dimensions are rejected with a warning.

### Input-Frame Rotation

The Qt **Settings > Playback** group includes a **Rotate** checkbox. Enabling it activates a dropdown with **90 degrees clockwise**, **180 degrees**, and **90 degrees counterclockwise**. The enabled state and selected direction are saved with the other interface settings.

The equivalent command-line option is:

```bash
acmx2 --rotate clockwise
acmx2 --rotate 180
acmx2 --rotate counterclockwise
```

Rotation is applied to each source frame before DNN models, CUDA filters, texture caching, and GLSL shader processing, so every later stage sees the same orientation. Non-CUDA builds rotate the `cv::Mat` with `cv::rotate`; CUDA builds upload the frame to a `cv::cuda::GpuMat` and use `cv::cuda::rotate`.

With Window Resolution set to `Default`, either 90-degree mode swaps the source width and height—for example, `1920x1080` becomes `1080x1920`. An explicitly selected or entered Window Resolution is preserved and remains the final display/recording size. A 180-degree rotation does not swap dimensions.

### Qt Interface Session Persistence

The ACMX2 Qt interface restores last-used values both when dialogs are reopened
and when the whole application is closed and started again.

- **Persistent dialogs:** **Settings**, **Audio Settings**, **GPU Filter Settings**, and **MIDI Settings** now save their visible state through `QSettings`.
- **Reopen workflow:** Closing a dialog with either **OK** or **Cancel** preserves the current values so you can reopen it and continue adjusting from where you left off.
- **Restart workflow:** The launcher loads the persisted input mode, camera
  device, files, resolutions, FPS, output, recording, cache, rotation, 3D,
  ONNX, and encoder options during startup, so a previous session does not
  silently revert to camera 0.
- **Startup defaults:** When no saved preference exists yet, the **Settings** dialog now defaults the camera capture resolution to `1280x720` and the display/output resolution to `Default`.
- **Device-aware restore:** Camera, CUDA, audio, and MIDI selectors restore by stored values where possible, which keeps the selected device stable even if combo-box ordering changes.
- **Build-aware ONNX controls:** The startup log reports `OpenCV DNN: enabled`
  or `OpenCV DNN: disabled`. ONNX selection is unavailable when the chosen
  engine binary was built with `-DWITH_OPENCV_DNN=OFF`.

### MIDI Controller Support

ACMX2 now supports MIDI input devices for real-time control of shaders and parameters via hardware knobs and buttons.

- **Command line options:**
  - `--midi-map <file.midi_cfg>` — Load a MIDI mapping configuration file
  - `--midi-device <index>` — Select MIDI input device by index
  - `--list-midi` — List available MIDI input devices
- **MIDI Map Tool:** New standalone `midi-map` application for creating MIDI controller mappings:
  - Live MIDI message monitor
  - Capture button/knob assignments to ACMX2 actions (shader navigation, time control, pitch, yaw, speed, etc.)
  - Updated action descriptions matching actual ACMX2 keybindings (playlist toggle, freeze frame, shader bypass, 3D camera controls, etc.)
  - Save/load `.midi_cfg` configuration files
- **Qt Interface:** New **MIDI Settings** dialog with:
  - Enable/disable MIDI
  - Browse for config file or launch the MIDI Map Tool directly
  - Select MIDI device from detected inputs
- **Velocity-sensitive knobs:** Knob turn speed controls the rate of action firing
- **MIDI Overlay:** Real-time on-screen display showing MIDI status, knob states, and button presses with fade animation
- **F9 key:** Toggle overlay visibility on/off

### Code Editor Improvements

The built-in GLSL shader editor has been significantly enhanced:

- Line number gutter
- Current line highlighting
- Bracket matching
- Auto-indentation on new lines
- Duplicate line (Ctrl+D)
- Move line up/down (Alt+Up/Down)
- Toggle comment (Ctrl+/)
- Smart Home key behavior
- Block indent/unindent (Tab/Shift+Tab with selection)

### Shader Playlist Tree with Named Nodes

ACMX2 now supports shader playlists organized into named tree nodes, allowing you to group shaders and cycle through node groups during playback.

- **Command line:** Use `--playlist <file.txt>` to load a playlist file. Supports the new `[NodeName]` section format as well as flat shader-per-line files.
- **Runtime controls:**
  - **P** — Toggle playlist mode on/off (loads first node's shaders into multi-pass pipeline)
  - **Up/Down arrows** — Navigate to the previous/next tree node and load its shaders into multi-pass
  - **Shift+Up/Down arrows** — Change the post-multipass shader (the shader that runs after the node's multi-pass chain) without altering the current playlist position
- **Qt Interface:** The **Shader Playlist Settings** dialog features a tree widget with named nodes:
  - Add, rename, and remove node groups
  - Add shaders to specific nodes via search
  - Each node's shaders are loaded as a multi-pass chain when selected at runtime
  - **Save List... / Load List...** buttons persist playlists using `[NodeName]` section format
- **File format:** Playlist files use `[NodeName]` headers to group shaders:
  ```
  [Ambient]
  glow.glsl
  blur.glsl
  [Intense]
  fractal.glsl
  distort.glsl
  ```

### Distrobox Export

A script is provided to export ACMX2 applications from a Distrobox container to the host desktop:

```bash
bash scripts/export-distrobox.sh
```

This installs the application icon, creates `.desktop` files for both `acmx2_interface` and `midi-map`, and registers them with the host application menu.

### Multipass Shader Pass Save/Load

The **Multipass Shader Settings** dialog now includes **Save List...** and **Load List...** buttons, allowing you to save and restore your multipass shader chain as a text file.

### Camera Device Enumeration

- **Command line:** Use `--enumerate-device <index>` to list all supported resolutions and frame rates for a V4L2 camera device (Linux only).
- **Qt Interface:** The **Settings** dialog now automatically queries the selected camera device for its supported resolutions and frame rates. The resolution and FPS dropdowns are dynamically populated based on the device capabilities. Changing the camera device re-enumerates, and changing the resolution updates the available frame rates. In graphics file mode the FPS options default to 24, 30, and 60.
- **FPS preference persistence:** The selected camera FPS is now saved as a preferred value and restored when available after dialog reopen, app restart, or resolution list repopulation.
- **Linux loopback devices:** `v4l2loopback` often reports only its current producer interval even though it accepts a different consumer interval. ACMX2 adds 24, 25, 30, 50, 60, 90, 120, 144, and 240 FPS choices for these devices and uses the explicitly requested value when the driver continues to report a stale rate.
- **Above 60 FPS:** requesting a camera rate above 60 FPS creates the desktop OpenGL window with vsync disabled through the current `libmx2` API, preventing the display swap interval from imposing a 60 FPS ceiling. Rates at or below 60 FPS keep the normal vsync request.

### Modeless Playlist and Multipass Editing

The **Shader Playlist Settings** and **Multipass Shader Settings** windows are
modeless, so they can stay open while you use the main interface or edit shader
source. Choosing either menu action again raises the existing window and
refreshes its available shader list rather than opening another copy. Closing a
window deletes that dialog instance; the next menu action creates a fresh one
from the current saved selection.

In **Multipass Shader Settings**, double-click any shader in the selected list
to open it in the built-in code editor. If that source is already open, ACMX2
focuses the existing editor window.

### GPU Filter Save/Load

The **GPU Filter Settings** dialog now includes **Save List...** and **Load List...** buttons, allowing you to save and restore your GPU filter chain as a text file.

### Overlay Improvements

- All MIDI overlay text is now rendered in green for better readability
- When GPU filters are enabled, the overlay now displays the active GPU filter names in a comma-separated list (e.g., `GPU: Filter1, Filter2, Filter3`)

---

ACMX2 can also be built locally using a **Podman container** via the included `Containerfile.arch`.
This avoids dependency issues and produces a self-contained image. The provided container recipe is the **CUDA-enabled** variant and therefore **requires an NVIDIA GPU**; if you do not have one, use the native build below with `-DWITH_CUDA=OFF` instead.

---

## System Requirements (CUDA-enabled Container Build)

The Podman container build below targets the optional CUDA path. For that specific path your system must have:

- Linux (x86_64)
- NVIDIA GPU
- NVIDIA proprietary drivers installed on the host
- Podman
- NVIDIA Container Toolkit (for Podman)
- X11 or XWayland
- Webcam device (`/dev/video0`) for camera input
- Audio input device (microphone)
- Shader/Model files: https://lostsidedead.biz/packs/

> ⚠️ **Important**
> The container recipe in `podman/Containerfile.arch` is the CUDA build and will only run on NVIDIA GPUs.
> For AMD, Intel, or Apple hardware, use the native build with `-DWITH_CUDA=OFF` (see below).

---

## Step 1: Build the ACMX2 Container Image

From the repository root, build the image using the Arch Linux Containerfile:

```bash
cd podman
podman build -t acmx2-arch:latest -f Containerfile.arch .
```

> **Note:** The default CUDA architecture is `75` (Turing / RTX 20xx / GTX 16xx).
> Select another architecture with a build argument instead of editing the
> Containerfile:
>
> ```bash
> podman build -t acmx2-arch:latest -f Containerfile.arch \
>   --build-arg CUDA_ARCHITECTURES=86 .
> ```
>
> Common values are `86` for RTX 30xx (Ampere), `89` for RTX 40xx (Ada),
> `90` for Hopper, and `120` for RTX 50xx (Blackwell). To create one image for
> several GPU generations, quote a semicolon-separated value such as
> `--build-arg 'CUDA_ARCHITECTURES=75;86;89'`.

The image builds the repository-local `MXWrite/` tree together with the
standalone `acidcam` CLI, so it does not require a system MXWrite package. It
installs `acidcam`, `acmx2`, and `acmx2_interface` under `/usr/local`, clones
the maintained shader collection into `/opt/src/files/shaders`, and downloads
the model pack into `/opt/src/files/models`. The first time the GUI starts,
select `/opt/src/files/shaders` as its shader directory.

---

## Step 2: Verify the Image

```bash
podman images | grep acmx2-arch
```

---

## Step 3: Run ACMX2

```bash
cd podman
chmod +x run-acmx2-arch.sh
./run-acmx2-arch.sh
```

The script:
- Detects all `/dev/video*` webcam devices
- Enables NVIDIA GPU acceleration
- Mounts PulseAudio for audio input
- Passes `--device nvidia.com/gpu=all` for GPU access
- Mounts `~/container_share` at `/root/share` for file exchange
- Opens the ACMX2 interface window on your desktop

---

## Native Build (Without Container)

You can also build directly on Arch Linux using the scripts in `build-script/`:

```bash
# Install all dependencies
sudo bash build-script/install-deps-arch.sh

# Build and install ACMX2
sudo bash build-script/acidcam-gpu-arch.sh
```

---

## NVIDIA License Notice

When built with `-DWITH_CUDA=ON`, this project uses NVIDIA CUDA libraries.

Use of CUDA is subject to the NVIDIA Deep Learning Container License:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

By building or running the CUDA-enabled variant of this project (including the provided NVIDIA container), you agree to NVIDIA’s license terms. The default OpenGL-only build (`-DWITH_CUDA=OFF`) does not link any CUDA libraries and is not subject to this notice.

---

## Troubleshooting

### NVIDIA Driver Not Detected
Verify:
```bash
nvidia-smi
```

---

## Quick Start Summary

```bash
cd podman
podman build -t acmx2-arch:latest -f Containerfile.arch .
chmod +x run-acmx2-arch.sh
./run-acmx2-arch.sh
```
---

### Build Instructions
```bash
#!/bin/sh
git clone https://github.com/lostjared/libmx2.git
cd libmx2/libmx
mkdir build && cd build
cmake .. -DEXAMPLES=OFF -DOPENGL=ON
make -j$(nproc)
sudo make install
cd ../../../
git clone https://github.com/lostjared/acidcam-gpu.git
cd acidcam-gpu
mkdir build && cd build
cmake .. 
make -j$(nproc) && sudo make install
cd ../../
cd ACMX2
mkdir build && cd build
cmake .. -DAUDIO=ON
make -j$(nproc) && sudo make install
cd ../interface
mkdir build && cd build
cmake ..
make -j $(nproc)
cp -rf ../data/ .
cd ../midi-map
mkdir build && cd build
cmake .. && make -j$(nproc)
cd ../../../
echo "completed..."
```

### Optional Build Features (CUDA / Audio / MIDI)

Each of the major optional subsystems is toggled by a CMake flag on ACMX2:

`-DWITH_CUDA=ON` requires both the NVIDIA CUDA toolkit and an OpenCV build that
was compiled with CUDA support. Installing only the toolkit is not enough. On
Arch Linux, use `opencv-cuda` instead of `opencv`; elsewhere, install an
equivalent package or compile OpenCV with CUDA enabled.

| Flag | Default | Effect when `OFF` |
|------|---------|-------------------|
| `-DWITH_CUDA=ON/OFF` | `ON`  | Skips all CUDA GPU-filter paths and CUDA/OpenGL zero-copy interop; OpenCV `cudaimgproc` is no longer required; FFmpeg CUDA hw-decode is disabled; `--gpu-filter`, `--gpu-buffer`, `--cuda-device`, `--list-cuda-devices` are not available |
| `-DWITH_OPENCV_DNN=ON/OFF` | `OFF` | Disables ONNX/DNN model loading via YAML config files; requires `yaml-cpp` package; enables `--onnx`, `--edge`, `--human`, `--background`, `--black`, `--white` options |
| `-DAUDIO=ON/OFF`     | `OFF` | No RtAudio / audio reactivity |
| `-DMIDI=ON/OFF`      | `OFF` | No RtMidi / MIDI control |

#### Building without CUDA (pure OpenGL build)

If you do not have an NVIDIA GPU, cannot install the CUDA toolkit, or want to
build against a stock OpenCV 5 installation (no CUDA modules), configure ACMX2 with
`-DWITH_CUDA=OFF`. The engine falls back to the OpenGL/SDL2 shader path — all
shader-based features continue to work; only the CUDA GPU filter stack is
omitted.

```bash
# libmx2 (built from source, same as above)
git clone https://github.com/lostjared/libmx2.git
cd libmx2/libmx
mkdir build && cd build
cmake .. -DEXAMPLES=OFF -DOPENGL=ON
make -j$(nproc) && sudo make install
cd ../../../

# ACMX2 without CUDA
git clone https://github.com/lostjared/acidcam-gpu.git
cd acidcam-gpu/ACMX2
mkdir build && cd build
cmake .. -DWITH_CUDA=OFF
make -j$(nproc) && sudo make install

# Qt6 GUI (unchanged)
cd ../interface
mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

Note: when `WITH_CUDA=OFF` you do **not** need `opencv-cuda` or the NVIDIA
CUDA toolkit — stock OpenCV 5 is sufficient, and the top-level `acidcam-gpu`
CUDA library does not need to be installed.

You can combine flags freely — for example an OpenGL-only build with audio:

```bash
cmake .. -DWITH_CUDA=OFF -DAUDIO=ON
```

#### Runtime feature detection

At startup the Qt6 interface probes the installed `acmx2` binary with
`--check-cuda`, `--check-audio`, and `--check-midi`, and automatically disables
the menu entries (GPU Filter Settings, Audio Settings, MIDI Settings), the
Session-Properties CUDA device selector, and the corresponding CLI arguments
for any feature that is not compiled in. You can also run the probes directly:

```bash
acmx2 --check-cuda     # "CUDA: enabled"  or "CUDA: disabled"
acmx2 --check-audio    # "AUDIO: enabled" or "AUDIO: disabled"
acmx2 --check-midi     # "MIDI: enabled"  or "MIDI: disabled"
```

Early Example (as a GIF)

![jaredrgb](https://github.com/user-attachments/assets/1d2115ba-7b86-4c30-8845-1f2154af00c2)

![fractal](https://github.com/lostjared/acidcam-gpu/blob/main/ac.gif)

[Example YouTube Music Video](https://youtu.be/R0Y1wN6XHZI)


# Latest Shader Pack

The maintained shader collection is available at
[github.com/lostjared/shaders](https://github.com/lostjared/shaders). It includes
the cache-shader updates for the `sampler2DArray history` interface.

```bash
git clone https://github.com/lostjared/shaders.git
```

# Latest 3D Geometry 

https://lostsidedead.biz/acmx2/models.zip

## Older Packs

https://lostsidedead.biz/packs

# ACMX2 Container Environment Documentation

This guide covers the setup and usage of the ACMX2 / Acidcam-GPU development environment on **Bazzite**. It details how to build the container, manage file paths, and ensure full hardware access (NVIDIA GPU, Webcam, and X11 Display).

---

## 1. Host System Setup

Before launching the container, you must establish a specific folder structure on your Bazzite host. This ensures your code is persistent and files can be easily transferred.

Open a terminal on your host and run:

```bash

# Create a "scratch pad" for transferring files (videos, models, loose shaders)
mkdir -p ~/container_share
```

**Folder purposes:**

- `~/container_share`  
  Shared volume. Files placed here are visible to both the host and the container.

---

### A. Program to Run
- **Run**
  ```bash
  ./acmx2_interface
  ```

### B. File Paths (Shaders & Models)

- **External assets (models, videos)**  

  1. Copy files to `~/container_share` on the host.
  2. Access them in the container from:
     ```
     /root/share/test_video.mp4
     ```

### C. Saving Output

- **Binaries / render output**  
  Copy output files to `/root/share` inside the container.  
  They will appear in `~/container_share` on the host.

---

## 5. Troubleshooting

### Camera errors
**Error:**  
```
Could not open camera index: 0
```

**Fix:**  
Check available devices:
```bash
ls /dev/video*
```
If your camera is `/dev/video2`, update the `--device` flag in `run.sh`.

---

### X11 display errors
**Error:**  
```
qt.qpa.xcb: could not connect to display
```

**Fix:**  
Ensure the following line exists in `run.sh`:
```bash
xhost +local:
```
Re-run `./run-acmx2.sh` to refresh permissions.

---

### Permission denied on files

Files created inside the container are owned by root.

**Fix ownership on the host:**
```bash
sudo chown -R $USER:$USER ~/ACMX2
```
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

### Custom Uniforms

Shader libraries that use `library.json` can define up to 64 custom `float`
uniforms. Each definition supplies a range, a step size, and a persisted value:

```json
{
  "version": 1,
  "shaders": [
    "plasma.glsl",
    "feedback_cache.glsl"
  ],
  "custom_uniforms": {
    "warp_amount": {
      "minimum": 0.0,
      "maximum": 2.0,
      "step": 0.05,
      "value": 0.75
    },
    "color_shift": {
      "minimum": -1.0,
      "maximum": 1.0,
      "step": 0.01,
      "value": 0.0
    }
  }
}
```

Open the library in the Qt interface, then choose **List > Add Custom
Uniforms...**. The dialog can add or delete definitions and provides a slider
and numeric control for every value. Slider changes are saved back to
`library.json` and published immediately to a running ACMX2 process. Adding or
removing a uniform also requests a live reload of the current shader.

Use a custom value like any other GLSL `float` uniform:

```glsl
uniform float warp_amount;

void main() {
    vec2 uv = gl_FragCoord.xy / iResolution;
    uv.x += sin(uv.y * 20.0 + time_f) * warp_amount * 0.05;
    gl_FragColor = texture(samp, uv);
}
```

The declaration is optional: if shader source references a configured custom
uniform but does not declare it, ACMX2 injects `uniform float <name>;` before
compilation. Explicit declarations remain useful when the shader is also run by
other GLSL hosts.

Names must be unique GLSL identifiers, must not begin with `gl_`, and must be
shorter than 64 UTF-8 bytes. Ranges and values must be finite,
`maximum > minimum`, and `step > 0`; loaded values outside the range are
clamped. Custom uniforms require `library.json` and are not available from a
legacy `index.txt` manifest.

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
`--texture-cache-array`; the option also enables the texture cache, so a
separate `--texture-cache` flag is not required:

```bash
acmx2 -s ./shaders/index.txt --texture-cache-array --texture-cache-size 16
```

The maintained [ACMX2 shader collection](https://github.com/lostjared/shaders)
has been updated to support this array-backed history path. Clone it for the
latest compatible cache shaders, or run `git pull` in an existing checkout.

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
| `spectrum` | `sampler1D` | FFT frequency-magnitude spectrum for the current frame (256 bins, GL_TEXTURE9) |
| `spectrum0` | `sampler1D` | Current-frame alias of the live spectrum |
| `spectrum_history` | `sampler1DArray` | Runtime-sized FFT history array enabled by `--enable-audio-buffers <N>` |
| `spectrum_history_head` | `int` | Physical layer containing the newest history frame |
| `spectrum_history_size` | `int` | Allocated history-array layer count |

History age is a dynamic array coordinate, so one sampler binding supports any
requested depth up to `GL_MAX_ARRAY_TEXTURE_LAYERS`:

```glsl
int size = max(spectrum_history_size, 1);
int layer = (spectrum_history_head - (age % size) + size) % size;
float energy = texture(spectrum_history, vec2(frequency, float(layer))).r;
```

Convert legacy `spectrum1`, `spectrum2`, and later lookups with:

```bash
scripts/migrate_spectrum_samplers.pl --dry-run shaders
scripts/migrate_spectrum_samplers.pl shaders
```

---

## Notes

This setup is designed to keep your development workflow fast and reproducible while maintaining full access to GPU acceleration, camera devices, and graphical output.

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/110f4959-67ff-4cef-aa0c-f036e6ee78ba" />


Related Videos [YouTube Channel](https://youtube.com/+JaredBruni)
Contact me: [Contact](https://lostsidedead.biz/contact.html)
