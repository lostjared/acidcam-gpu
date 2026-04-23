
# ACMX2
<img width="2560" height="1440" alt="Screenshot From 2026-04-13 07-41-08" src="https://github.com/user-attachments/assets/0e0cd74f-ce6b-47e5-abfa-bc268cd74d4b" />

#  Now  Works on AMD and Apple Hardware (without CUDA support)

<img width="3360" height="2100" alt="acmx2 macos" src="https://github.com/user-attachments/assets/851527a0-978f-40ee-9edb-00b7d09b9b91" />


[Full Documentation](https://lostsidedead.biz/acmx2/docs/)

To regenerate local docs in a versioned folder and refresh `docs/latest`:

```bash
./build-docs.sh
```

This keeps `docs/index.html` as a stable redirect to the newest generated docs.

The command-line engine for **acidcam-gpu**. Applies GLSL shaders and CUDA GPU filters to live camera feeds, video files, or static images in real time. Supports 3D model rendering, audio reactivity, MIDI control, shader playlists, and multipass shader chains.

---

## Features

- **Camera, video, or image input** with configurable resolution
- **Shader library** — load a single fragment shader or a full library via `index.txt`
- **CUDA GPU filters** — apply GPU-accelerated pixel filters in addition to shaders
- **3D mode** — render shaders onto a 3D model (`.mxmod`)
- **Multipass shaders** — chain multiple shader passes in a single frame
- **Random multipass mode** — generate random 1–5 shader chains on the fly with crossfade transitions; navigate the main shader with Up/Down while in random mode
- **Shader playlists** — cycle through an ordered list of shaders
- **Audio reactivity** — shaders respond to real-time audio input (RtAudio/PulseAudio)
- **File-based audio reactivity** — drive audio-reactive shaders from an audio or video file instead of a live microphone via `--audio-file`; the audio track is automatically muxed into the output video
- **MIDI control** — map hardware knobs and buttons to shader parameters
- **Video recording** with optional audio muxing via FFmpeg
- **Silent mode** — headless video processing without a window
- **Shader cache** — precompile shader binaries for fast startup
- **Qt6 GUI** available via the `interface/` subdirectory (`acmx2_interface`)
- **MIDI Map Tool** — standalone Qt6 app for creating MIDI controller mappings (`interface/midi-map/`)

---

## Building

### macOS (Intel & Apple Silicon)

ACMX2 builds and runs on macOS using OpenGL 4.1 backed by Metal. **CUDA is not available on macOS**, so the engine automatically builds with `-DWITH_CUDA=OFF`. Shader-based effects work fully; only the CUDA GPU-filter pipeline is omitted.

The Apple Metal-backed OpenGL 4.1 driver does not support `glProgramBinary`, so **the shader binary cache is disabled by default at runtime** on macOS.

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
   acmx2 -p /usr/local/share/acmx2/data -s /path/to/macos-shaders/index.txt -d 0
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
./acmx2 -p ./data -i input.mp4 -s ./shaders -h 5 --silent -o output.mp4
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
| | `--audio-file` | `<file>` | Use audio from a file for reactivity instead of the microphone; the audio track is muxed into the output video |
| | `--audio-trunc` | | Stop playback when the audio file reaches the end |

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

The `spectrum` uniform is a 1D texture (`GL_R32F`, 256 texels) holding the FFT magnitudes of the current audio frame. Texel coordinate `x = 0.0` is the DC bin and `x = 1.0` is the Nyquist frequency (22 050 Hz at 44 100 Hz sample rate). Sampling uses `GL_LINEAR` filtering and `GL_CLAMP_TO_EDGE` wrapping. Example usage:

```glsl
uniform sampler1D spectrum;           // bound to texture unit 9
float energy = texture(spectrum, x).r; // x in [0,1]
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




