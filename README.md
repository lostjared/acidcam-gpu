# acidcam-gpu

![screenshot](https://github.com/lostjared/acidcam-gpu/raw/main/acmx2.png)
<img width="2560" height="1440" alt="Screenshot From 2026-04-12 19-33-51" src="https://github.com/user-attachments/assets/88b6c714-031d-422d-9a9f-65d56ac190bf" />

[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Hardware: NVIDIA RTX](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%202070-green.svg)](https://www.nvidia.com/en-us/geforce/rtx/)
[![Framework: CUDA](https://img.shields.io/badge/Framework-CUDA%2012.x-76b900.svg)](https://developer.nvidia.com/cuda-zone)

# ACMX2 – Linux (NVIDIA GPU Required)
![screenshot](https://github.com/lostjared/acidcam-gpu/blob/main/image.jpg)

# acidcam-gpu / ACMX2

[Full Documentation](https://lostsidedead.biz/acmx2/docs/)


**acidcam-gpu** is a high-performance, real-time video manipulation engine designed to push the boundaries of psychedelic glitch  art. Part of the **ACMX2** and **libmx2** ecosystem, it offloads complex glitch filters to **NVIDIA GPUs**, enabling fluid, high-resolution visual transformations at 60+ FPS. Requires you have OpenCV 4 compiled with CUDA support.

## 🚀 Purpose & Vision
The original project brought a massive library of "glitch" filters to digital artists. However, as resolutions climbed to 4K and filter stacks became more complex, CPU-based processing hit a bottleneck. 

**acidcam-gpu** solves this by:
* **Parallelizing the Chaos:** Using custom CUDA kernels to process millions of pixels simultaneously.

## 🛠 Tech Stack
* **Language:** C++20
* **Parallel Computing:** NVIDIA CUDA (Optimized for **RTX 2070**)
* **Graphics API:** OpenGL / SDL (Hardware-accelerated rendering)
* **Format Support:** Native **MX2 MXMOD** 3D model parsing for real-time geometry glitching.

## ⚡ Why NVIDIA & CUDA?
This project is built specifically for the NVIDIA ecosystem to leverage:
* **Shared Memory:** Fast on-chip memory to speed up neighborhood-based filters.
* **Massive Throughput:** Harnessing thousands of CUDA cores to apply multiple glitch layers in a single pass.

## Project Goals:
* **Zero-Copy Interop:** High-speed texture sharing between CUDA and OpenGL.
* **Visual User Interface** Simple to use User interface
* **Command line tool** Command line tool

## 📦 Installation & Environment
This project is developed and tested on **Bazzite Linux** using **Arch Linux** containers via **Distrobox**.

### Prerequisites
* **NVIDIA GPU:** RTX 20-series or newer.
* **Drivers:** NVIDIA Proprietary Drivers (v535+).
* **Environment:** Arch Linux (or compatible). Install all dependencies via `pacman`:

**Build Tools:**
```bash
sudo pacman -S --needed base-devel git cmake ninja pkg-config curl unzip
```

**NVIDIA & CUDA:**
```bash
sudo pacman -S --needed nvidia-utils cuda
```

**OpenCV (with CUDA support):**
```bash
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

### Shader Options

| Short | Long | Value | Description |
|-------|------|-------|-------------|
| `-s` | `--shaders` | `<file>` | Shader library index file |
| `-f` | `--fragment` | `<file>` | Single fragment shader file |
| `-h` | `--shader` | `<index>` | Initial shader index in library |
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
| `--silent` | | Process video without window (video files only, requires `-o`) |

### 3D / Model Options

| Long | Value | Description |
|------|-------|-------------|
| `--texture-cache` | | Enable texture cache |
| `--cache-delay` | `<frames>` | Texture cache delay in frames |
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

### MIDI Options (requires `MIDI_ENABLED` build)

| Long | Value | Description |
|------|-------|-------------|
| `--midi-map` | `<file>` | MIDI config file (`.midi_cfg`) |
| `--midi-device` | `<index>` | MIDI input device index |
| `--list-midi` | | List available MIDI input devices and exit |

---

## Keyboard Controls

### General Controls

| Key | Action |
|-----|--------|
| `Up` | Previous shader (or previous playlist tree node if playlist enabled) |
| `Down` | Next shader (or next playlist tree node if playlist enabled) |
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

---

## Recent Updates

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

### Version 2.7.0

- Version bump to 2.7.0
- Updated build scripts and Podman container configuration
- Audio muxing fix: uses video duration for precise audio/video sync
- Audio track copy fix for finished recordings

### Shader Playlist Tree with Named Nodes

ACMX2 now supports shader playlists organized into named tree nodes, allowing you to group shaders and cycle through node groups during playback.

- **Command line:** Use `--playlist <file.txt>` to load a playlist file. Supports the new `[NodeName]` section format as well as flat shader-per-line files.
- **Runtime controls:**
  - **P** — Toggle playlist mode on/off (loads first node's shaders into multi-pass pipeline)
  - **Up/Down arrows** — Navigate to the previous/next tree node and load its shaders into multi-pass
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

### GPU Filter Save/Load

The **GPU Filter Settings** dialog now includes **Save List...** and **Load List...** buttons, allowing you to save and restore your GPU filter chain as a text file.

### Overlay Improvements

- All MIDI overlay text is now rendered in green for better readability
- When GPU filters are enabled, the overlay now displays the active GPU filter names in a comma-separated list (e.g., `GPU: Filter1, Filter2, Filter3`)

---

ACMX2 is built locally using a **Podman container** via the included `Containerfile.arch`.
This avoids dependency issues and produces a self-contained image, but it **requires an NVIDIA GPU**.

---

## System Requirements

Before building and running ACMX2, your system must have:

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
> This build uses NVIDIA CUDA.
> It will **not run on AMD or Intel GPUs**.

---

## Step 1: Build the ACMX2 Container Image

From the repository root, build the image using the Arch Linux Containerfile:

```bash
cd podman
podman build -t acmx2-arch:latest -f Containerfile.arch .
```

> **Note:** The default CUDA architecture is `75` (Turing / RTX 20xx / GTX 16xx).
> Edit `Containerfile.arch` if your GPU differs:
> - RTX 30xx (Ampere): `86`
> - RTX 40xx (Ada Lovelace): `89`

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

This poject uses NVIDIA CUDA libraries.

Use of CUDA is subject to the NVIDIA Deep Learning Container License:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

By running this container, you agree to NVIDIA’s license terms.

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
cd acidcam-gpu/MXWrite
mkdir build1 && cd build1
cmake .. && make -j$(nrpoc) && sudo make install
cd ../..
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

Early Example (as a GIF)

![jaredrgb](https://github.com/user-attachments/assets/1d2115ba-7b86-4c30-8845-1f2154af00c2)

![fractal](https://github.com/lostjared/acidcam-gpu/blob/main/ac.gif)

# Latest Shader Pack

https://lostsidedead.biz/packs

# Latest 3D Geometry 

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

### Texture Cache Uniforms (shaders with "cache" in the filename)

| Uniform | Type | Description |
|---------|------|-------------|
| `samp1`–`samp8` | `sampler2D` | Cached frame textures from the texture cache ring buffer |

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

---

## Notes

This setup is designed to keep your development workflow fast and reproducible while maintaining full access to GPU acceleration, camera devices, and graphical output.

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/110f4959-67ff-4cef-aa0c-f036e6ee78ba" />


Related Videos [YouTube Channel](https://youtube.com/+JaredBruni)
Contact me: [Contact](https://lostsidedead.biz/contact.html)

