
# ACMX2

The command-line engine for **acidcam-gpu**. Applies GLSL shaders and CUDA GPU filters to live camera feeds, video files, or static images in real time. Supports 3D model rendering, audio reactivity, MIDI control, shader playlists, and multipass shader chains.

---

## Features

- **Camera, video, or image input** with configurable resolution
- **Shader library** — load a single fragment shader or a full library via `index.txt`
- **CUDA GPU filters** — apply GPU-accelerated pixel filters in addition to shaders
- **3D mode** — render shaders onto a 3D model (`.mxmod`)
- **Multipass shaders** — chain multiple shader passes in a single frame
- **Shader playlists** — cycle through an ordered list of shaders
- **Audio reactivity** — shaders respond to real-time audio input (RtAudio/PulseAudio)
- **MIDI control** — map hardware knobs and buttons to shader parameters
- **Video recording** with optional audio muxing via FFmpeg
- **Silent mode** — headless video processing without a window
- **Shader cache** — precompile shader binaries for fast startup
- **Qt6 GUI** available via the `interface/` subdirectory (`acmx2_interface`)

---

## Building

ACMX2 is part of the acidcam-gpu project. See the [main README](../README.md) for full build instructions.

```bash
cd ACMX2
mkdir build && cd build
cmake .. -DAUDIO=ON
make -j$(nproc) && sudo make install
```

The Qt6 GUI is built separately:

```bash
cd ACMX2/interface
mkdir build && cd build
cmake .. && make -j$(nproc)
cp -rf ../data/ .
```
---

## Usage Examples

**Camera with a shader library:**
```bash
./acmx2 -p ./data -s ./shaders -d 0 -r 1920x1080
```

**Process a video file with GPU filters and record output:**
```bash
./acmx2 -i input.mp4 -s ./shaders --gpu-filter 0,5,12 -o output.mp4 --copy-audio
```

**Single shader, fullscreen, with audio reactivity:**
```bash
./acmx2 -f effect.glsl -d 0 -n -w --audio-input 3
```

**3D mode with a model:**
```bash
./acmx2 -s ./shaders --enable-3d --model cube.mxmod -d 0
```

**Silent (headless) batch processing:**
```bash
./acmx2 -i input.mp4 -s ./shaders -h 5 --silent -o output.mp4
```

**Build shader cache:**
```bash
./acmx2 -p ./data --build ./shaders --enable-3d
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




