# ACMX2 Interface

Qt6-based GUI launcher for the ACMX2 engine.

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
