
# ACMX2

The command-line engine for **acidcam-gpu**. Applies GLSL shaders and CUDA GPU filters to live camera feeds, video files, or static images in real time. Supports 3D model rendering, audio reactivity, MIDI control, shader playlists, and multipass shader chains.

Built on [libmx2](https://github.com/lostjared/libmx2) and requires an NVIDIA GPU with CUDA support.

![image](https://github.com/user-attachments/assets/7cdf6c57-0938-49ea-906d-594b48149acb)
<img width="2048" height="1152" alt="image" src="https://github.com/user-attachments/assets/8aaba334-3e80-46e9-951f-4da2d75ec527" />

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

## Audio-Reactive Shader Uniforms

When built with `AUDIO=ON` and launched with `-w`, the following GLSL uniforms are available:

| Uniform | Description |
|---------|-------------|
| `amp` | Amplitude scaled by sensitivity |
| `uamp` | Raw untouched amplitude |
| `amp_peak` | Highest sample value in the buffer |
| `amp_rms` | Root mean square energy |
| `amp_smooth` | Exponentially smoothed amplitude |
| `amp_low` | Low-frequency energy (<300 Hz) |
| `amp_mid` | Mid-frequency energy (300–3000 Hz) |
| `amp_high` | High-frequency energy (>3000 Hz) |
| `iamp` | Estimated dominant frequency (Hz) |
| `iSampleRate` | Audio sample rate (44100 Hz) |

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

## Screenshots

<img width="2048" height="1152" alt="Main Window" src="https://github.com/user-attachments/assets/1720bf11-9270-431a-8dba-96172482f483" />
<img width="936" height="540" alt="Main Window" src="https://github.com/user-attachments/assets/a3ea7c6c-a761-4aa9-9843-4502e9fcb8da" />




