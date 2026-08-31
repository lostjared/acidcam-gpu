# ACMX Interface

Qt-based GUI launcher for the ACMX2 engine with staged ACMXVK integration.

## Backend Integration

Interface version 2.128.0 includes the current ACMXVK integration increments:

- **Backend > ACMX2** and **Backend > ACMXVK** are exclusive, persisted
  selections.
- Each backend retains its own executable, active shader library, and recent
  library list.
- `library.json` may include a top-level `"backend": "acmx2"` or
  `"backend": "acmxvk"` hint. Loading a library tagged for the other backend
  offers to switch before loading it.
- Legacy manifests without a backend hint remain valid.
- ACMXVK source manifests use `"library_type": "source"`; compiled SPIR-V
  manifests use `"library_type": "runtime"`.
- With ACMXVK selected, **List > New Shader Library** creates a Vulkan source
  library with `library.json`, and **New Shader File** creates either a `.frag`
  source or a `compute/*.comp` source using ACMXVK-compatible descriptor
  bindings. Both actions run the native C++ source-manifest generator, which
  recursively refreshes the shader list and custom-uniform slots while
  preserving edited ranges and values. ACMX2 keeps its legacy `.glsl` and
  `index.txt`/JSON creation behavior.
- New ACMXVK libraries always include a valid minimal `default.frag`
  passthrough shader. It declares only the fragment input/output, binding-zero
  source sampler, and texture-copy `main`, so a newly created library can be
  built and run immediately instead of containing an empty shader source.
- **Help > Built-in Uniform Reference** follows the selected backend. ACMXVK
  value entries include their ready-to-paste `#define` mapping into the
  `SpriteExtended` block, while descriptor-backed resources retain their GLSL
  declarations. ACMX2 continues to show its native uniform declarations.
- **List > Shader Library Builder** now follows the active backend. In ACMXVK
  mode it accepts `.frag` and `.comp` sources, rejects compiled runtime
  libraries, exports compute sources beneath `compute/`, and invokes the same
  native manifest generator used by New Shader Library. Existing uniform
  ranges and values are retained when replacing an ACMXVK source library.
  ACMX2 mode continues to assemble `.glsl`/`.comp` libraries with its original
  layout.
- **Run Selected**, **Run All**, and **Copy Command** launch or generate a
  command for the active backend. ACMXVK runs use `--shaders` and
  `--shader-file`, preserving fragment/compute type and custom-uniform metadata
  from `library.json`.
- Feature checks, encoder queries, and CUDA-device discovery use the selected
  backend executable. ACMXVK distinguishes MXVK CUDA interop from optional
  acidcam-gpu filter support.
- Installed ACMXVK data is discovered beside its executable or under the normal
  `/usr/local`, `/opt/homebrew`, and `/usr` share directories.
- Loading an ACMXVK source library changes **Playback > Rebuild Shader Cache**
  to **Playback > Build**. It invokes the selected ACMXVK executable with
  `--build library.json --builddir .acmxvk-build`; unchanged shaders remain
  up to date while changed fragment and compute sources are compiled.
- The generated `.acmxvk-build/library.json` is an ACMXVK runtime manifest.
  **Run Selected**, **Run All**, multipass chains, playlists, and copied commands
  transparently translate source names such as `effect.comp` to
  `effect.comp.spv` in that build directory. The source library remains loaded
  for browsing and editing.
- A missing or stale build produces a Build prompt instead of launching old or
  incomplete SPIR-V output. The hidden `.acmxvk-build/` directory is generated
  data and should normally be excluded from commits.
- ACMXVK source libraries show `Up to Date`, `Stale`, or `Not Built` for each
  shader in the **Build Status** column. Loading or sorting an unchanged list no
  longer rewrites `library.json`, so an up-to-date build is launched without a
  redundant Build prompt. Launch validation compares shader timestamps plus the
  actual runtime shader list and custom-uniform metadata; manifest timestamps
  alone do not make a library stale.
- Running a stale, incomplete, or unbuilt ACMXVK source library now presents a
  **Yes/No** rebuild prompt. Choosing **Yes** starts **Playback > Build**
  immediately and streams its progress into the interface log; choosing
  **No** leaves the library unchanged.
- **Playback > Fix Build** runs ACMXVK with `--build library.json --fix
  .acmxvk-build`. It uses the normal build output directory and progress log,
  but removes failed outputs from the generated runtime manifest so the
  remaining successfully compiled shaders can still be used.
- When an automatic rebuild was accepted from Run Selected, Run All, or Copy
  Command, a successful build now resumes that original action automatically.
  Failed or cancelled builds remain stopped and retain their diagnostics in the
  interface log.
- Interface-launched ACMXVK runs and source builds include `--unbuffered`, so
  ACMXVK stdout and stderr are streamed into the interface log while the process
  is running instead of appearing only when an operating-system buffer fills.
- Interface-launched ACMXVK runs include `--interface-shm`. While ACMXVK is
  running, right-click a shader (or use **Set Current Shader**) to switch the
  active fragment or compute shader through its normal crossfade path. Source
  names are resolved to their corresponding `.spv` files in `.acmxvk-build`.
- The **Custom Uniforms** dialog now controls a running ACMXVK process through
  the same synchronized interface channel. Values are matched by manifest name
  and applied live to fragment, compute, multipass, and 3D shader paths.
- Custom-uniform `slot` fields are preserved when the dialog saves values.
  Explicit slots are validated as unique and contiguous, preventing interface
  edits from silently remapping the uniform ABI used by compiled shaders. The
  source-manifest generator can repair slots while retaining edited ranges and
  values from an existing `library.json`.
- The Custom Uniforms dialog now assigns and writes a contiguous explicit slot
  for every newly added uniform, removes its JSON entry when deleted, and
  renumbers later entries to preserve the runtime ABI. Both the dialog and the
  manifest writer enforce the 64-scalar (`custom_uniforms[16]`) maximum. Each
  row shows its current slot/location and provides **Copy** for the exact GLSL
  declaration (`#define name ext.custom_uniforms[N].component` in ACMXVK
  mode).
- **List > Remove Shader** now removes the selected entry from the active
  `library.json` or legacy `index.txt` atomically before updating the shader
  list. The source shader file remains on disk, and a manifest write failure
  leaves the interface row intact.
- Creating another ACMXVK shader refreshes the manifest from its current
  entries plus the new source instead of rescanning every file in the folder.
  Sources intentionally removed from the manifest therefore remain excluded;
  the standalone manifest-generator command remains the explicit full-rescan
  operation.
- ACMXVK source-build freshness compares custom-uniform metadata semantically
  by explicit slot, name, range, step, and value. Equivalent floating-point
  values rewritten by ACMXVK's runtime-manifest serializer no longer trigger a
  repeated "build is out of date or incomplete" prompt.
- The Settings dialog exposes ACMXVK-only **Maximize FPS**, **Use Source FPS**,
  and **Use Source Audio** controls. They are enabled only for compatible input
  modes and emit `--maximize-fps`, `--use-source-fps`, and
  `--use-source-audio` respectively.
- With the ACMXVK backend selected, the Settings dialog obtains camera names,
  native formats, resolutions, and FPS choices from ACMXVK's
  `--list-camera-devices` and `--enumerate-device` probes. ACMX2 selection
  retains its existing Linux, macOS, and Windows discovery paths.
- Maximum duration and output-size limits are added to launch commands only
  while output video recording is enabled.
- Shared interface control verifies that its named semaphore remains published
  before every engine launch and recreates it after an external unlink. Engine
  and launcher errors now include the failing semaphore name and OS error.
- Shared-memory startup checks the existing object with `fstat` and calls
  `ftruncate` only for a newly created zero-length object. Reopening the
  interface therefore preserves a correctly sized macOS shared-memory object;
  syscall failures and unsafe size mismatches are reported in the ACMX log.
- Multipass settings are published to ACMXVK at startup and while it is
  running. Applying, reordering, or disabling passes in the interface rebuilds
  the Vulkan fragment/compute chain without restarting the engine.
- Playback **Repeat** and **Normalized Time** changes are published to a running
  ACMXVK process. Video looping and fixed-per-output-frame shader timing can be
  enabled or disabled without restarting the engine.
- **Display Filter** and **Watermark** changes are applied live by ACMXVK,
  including watermark text and RGB color. Runtime watermark text is validated
  before use, and enabling it preserves the hidden-HUD default.
- **GPU Filter Settings** can replace, disable, or enable ACMXVK's ordered CUDA
  filter chain and temporal-buffer size while it is running. Invalid chains are
  rejected without discarding the active filter engine.
- Changing an audio file in **Audio Settings** switches a running ACMXVK
  file-audio session without restarting it. The new source is decoded before
  replacing the current source, together with repeat, truncation, output-device,
  and pass-through settings.
- Saving a `.frag` or `.comp` source in the built-in editor while ACMXVK is
  running invokes `glslc` for only that source. The interface validates and
  atomically installs the resulting `.spv` in `.acmxvk-build`, then requests a
  live Vulkan pipeline reload. Compile failures preserve the previous module
  and print the complete compiler diagnostic in the ACMX log.
- **Properties** (`Ctrl+,`) includes an ACMXVK shader-compiler section. Use
  **Automatic glslc** to resolve `glslc` from `PATH` or `VULKAN_SDK`, or select
  a custom glslc-compatible executable. The selected compiler is used for
  **Build**, **Fix Build**, and live single-shader compilation.

The remaining shared runtime controls are still ACMX2-only. ACMX2-specific
binary-cache maintenance actions remain disabled when ACMXVK is selected.

## Building

```bash
cd interface
mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

## Install

`make install` places:

- `acmx2_interface` → `<prefix>/bin/`
- `create_acmxvk_source_manifest` → `<prefix>/bin/`
- `acmx2-interface.desktop` → `<prefix>/share/applications/`
- `acmx2.png` → `<prefix>/share/acmx2/`

The interface automatically resolves the ACMX2 data directory at `<prefix>/share/acmx2/` when the local `./data` directory is not present beside the binary.

## Session Behavior

- The **Settings**, **Audio Settings**, **GPU Filter Settings**, and **MIDI Settings** dialogs preserve their last-used values with `QSettings`.
- Closing these dialogs with **OK** or **Cancel** keeps the current selections so reopening them resumes from the previous state.
- On a fresh configuration with no saved values, the main **Settings** dialog defaults camera capture resolution to `1280x720` and display/output resolution to `Default`.
- Persisted selections are restored by stored values when possible, which helps keep device selections stable across dialog rebuilds.

## Live Shader Editing

When ACMX2 is launched from the interface, saving a shader in the built-in
editor sends a shared-memory reload request to the running process. ACMX2
recompiles only that shader and installs the replacement immediately. If the
edited source does not compile or link, the current program remains active and
the complete OpenGL compiler message is written to the interface log.

## Encoding Controls

- The main **Settings** dialog includes an **Encoding Quality** group.
- Available controls:
	- preset: `ultrafast` through `veryslow`, plus NVENC `p1` through `p7`
	- tune: software tunes plus NVENC `hq`, `uhq`, `ll`, `ull`, and `lossless`
	- CRF quality override
	- codec mode: `auto`, `software`, `nvenc`, `h264_nvenc`, `hevc_nvenc`
	- NVENC presets `p1` through `p7` and tunes including `lossless`
	- extra FFmpeg-style encoder parameters, such as `-profile:v rext -pix_fmt yuv444p`
	- realtime low-latency encoding toggle
- These settings are persisted with `QSettings` and are forwarded to `acmx2` using the matching CLI flags.
