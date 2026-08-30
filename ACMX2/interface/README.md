# ACMX Interface

Qt-based GUI launcher for the ACMX2 engine with staged ACMXVK integration.

## Backend Integration

Interface version 2.106.0 includes the current ACMXVK integration increments:

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
- Interface-launched ACMXVK runs and source builds include `--unbuffered`, so
  ACMXVK stdout and stderr are streamed into the interface log while the process
  is running instead of appearing only when an operating-system buffer fills.
- Interface-launched ACMXVK runs include `--interface-shm`. While ACMXVK is
  running, right-click a shader (or use **Set Current Shader**) to switch the
  active fragment or compute shader through its normal crossfade path. Source
  names are resolved to their corresponding `.spv` files in `.acmxvk-build`.

Live source reload and the remaining shared runtime controls are still
ACMX2-only. Source editing and binary-cache maintenance actions also remain
disabled when ACMXVK is selected.

## Building

```bash
cd interface
mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

## Install

`make install` places:

- `acmx2_interface` → `<prefix>/bin/`
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
