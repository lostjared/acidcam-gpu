# ACMX

<img width="2560" height="1440" alt="ACMX shader library and live preview" src="https://github.com/user-attachments/assets/3af169df-abd3-49e8-84cc-70021a42e253" />
<img width="2560" height="1440" alt="ACMX shader editor" src="https://github.com/user-attachments/assets/35fb877a-4b2c-4bda-af6e-6dd894ae1593" />

[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](LICENSE)
[![Language: C++20](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg)](https://isocpp.org/)
[![Backends: OpenGL + Vulkan](https://img.shields.io/badge/Backends-OpenGL%20%2B%20Vulkan-8a2be2.svg)](#rendering-backends)

ACMX is an open-source, real-time video-synthesis and glitch-art workstation.
It processes cameras, video files, and still images through GPU shader chains,
then previews, records, or captures the result. One Qt interface manages shader
libraries, live editing, custom uniforms, playlists, multipass effects, audio
and MIDI controls, recording, and both rendering engines.

The repository name, `acidcam-gpu`, is historical. The current application is
**ACMX**, and the shared interface and ACMXVK backend are version **2.137.0**.
The established ACMX2 engine retains its own backend version, **2.101.1**.

## Current status

ACMX is usable on Linux, macOS, and Windows, with feature availability depending
on the selected backend and build options. The portable shader paths do not
require NVIDIA hardware. CUDA filters, CUDA/Vulkan interop, and Deep Dream are
optional features for supported NVIDIA configurations.

- **ACMX2** is the mature OpenGL/libmx2 backend. It supports runtime GLSL,
  OpenGL fragment and compute chains, shader caching where the driver permits
  it, HDR video, headless processing, audio/MIDI, OpenCV DNN effects, and the
  optional acidcam-gpu CUDA filter library.
- **ACMXVK** is the modern Vulkan/MXVK backend. It supports mixed fragment and
  compute SPIR-V chains, incremental shader-library builds, live recompilation,
  HDR processing, surface-free headless rendering, Vulkan playlists and
  transitions, Windows named shared memory, and optional CUDA interop.
- **Deep Dream** is implemented in ACMXVK through CUDA LibTorch, with VGG16 and
  Inception V3 feature models, temporal feedback, octaves, random animation,
  channel targeting, and live interface control.
- **Stable Diffusion video** is an optional ACMXVK integration with a compatible
  `stable-diffusion.cpp` `sd-server`. It keeps the model loaded for img2img
  processing, supports preview and constant-frame-rate recording, and can use
  either the Vulkan compute upscaler or a server-side ESRGAN model.

ACMXVK is the active expansion path, but it is not a drop-in shader replacement
for ACMX2: OpenGL GLSL libraries use a different uniform and descriptor ABI from
Vulkan source libraries compiled to SPIR-V. The interface keeps each backend's
executable, library, recent paths, and compatible controls separate.

## Rendering backends

| Backend | Graphics stack | Shader workflow | Best fit |
| --- | --- | --- | --- |
| [ACMX2](ACMX2/README.md) | OpenGL / libmx2 / SDL2 | Runtime GLSL fragment and compute shaders | Existing ACMX2 libraries, broad GPU compatibility, mature OpenGL workflows |
| [ACMXVK](ACMXVK/README.md) | Vulkan / MXVK / SDL3 | Source manifests compiled incrementally to SPIR-V | Modern fragment/compute pipelines, HDR, advanced headless work, Deep Dream, and Stable Diffusion |

Both engines share the same creative workflow through `acmx2_interface`. Select
the renderer from the interface's **Backend** menu before choosing a library or
running a session.

## Major features

- Camera, video-file, and still-image inputs with configurable input and output
  resolution
- Fragment/compute multipass chains, named playlists, crossfades, shader
  randomization, and random autopilot
- Built-in shader editor with diagnostics, live preview/reload, Find in Files,
  snippets, and custom-uniform controls
- Texture history, FFT and spectrum history, audio-reactive time, live or
  source-file audio, recording, and MIDI mapping
- Software and hardware FFmpeg encoding through the repository-local
  [MXWrite](MXWrite/README.md), including CRF/CQ or bitrate control, constant
  frame rate, editing-compatible PTS fill, source-audio muxing, and HDR Main10
- PNG, WebP, TIFF, and raw snapshots according to enabled dependencies
- Optional OpenCV DNN, acidcam-gpu CUDA filters, Deep Dream, and Stable
  Diffusion processing
- Qt settings that persist independently where backend behavior differs

See each backend README for its complete option list, requirements, platform
notes, shader ABI, examples, and current limitations.

## Repository layout

| Path | Purpose |
| --- | --- |
| `ACMX2/` | OpenGL engine, its backend documentation, runtime assets, shader packs, and models |
| `ACMXVK/` | Vulkan engine, SPIR-V build workflow, AI integrations, and Vulkan backend documentation |
| `ACMX2/interface/` | Shared Qt interface and standalone MIDI-map utility |
| `acidcam-gpu/` | Optional CUDA filter library and command-line filter utility |
| `MXWrite/` | Repository-local FFmpeg video writer used by both engines |
| `models/` and `shaders/` | Shared model configurations and example GLSL assets |
| `flatpak/` | Linux Flatpak manifest, patches, and packaging scripts |
| `build-script/`, `scripts/`, and `podman/` | Dependency, build, export, and container helpers |

Generated build trees, `.acmxvk-build` shader outputs, compiler databases,
logs, packaged binaries, and model downloads should remain outside commits.

## Build overview

The backend READMEs are authoritative for feature-specific builds:

- [Build and use ACMX2](ACMX2/README.md#building)
- [Build and use ACMXVK](ACMXVK/README.md#build)
- [Build the shared Qt interface](ACMX2/interface/README.md)

The portable ACMX2 CMake configuration is:

```bash
cmake -S ACMX2 -B build/acmx2 -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=OFF
cmake --build build/acmx2 --parallel
./build/acmx2/acmx2 --help
```

After MXVK and its dependencies are installed, a basic ACMXVK configuration is:

```bash
cmake -S ACMXVK -B build/acmxvk -DCMAKE_BUILD_TYPE=Release
cmake --build build/acmxvk --parallel
./build/acmxvk/acmxvk --help
```

Build the shared interface separately:

```bash
cmake -S ACMX2/interface -B build/interface -DCMAKE_BUILD_TYPE=Release
cmake --build build/interface --parallel
```

### Complete project helpers

`build-project-pcons.py` builds libmx2, MXVK, ACMX2, ACMXVK, and the Qt
interface into a local prefix without installing system packages. Dependencies
must already be available through `PATH` and `pkg-config`.

```bash
python3 build-project-pcons.py --jobs 4
```

The script clones missing libmx2 and MXVK checkouts beside this repository and
prints the final `PATH` export. Use `--help` to select existing dependency
sources, a build directory, an install prefix, or optional components. The
Pcons path covers the normal portable feature set; use CMake for ACMXVK's
LibTorch Deep Dream and Stable Diffusion integrations.

On macOS, `build-project-macos-cmake.sh` installs the required Homebrew
dependencies, builds libmx2 and MXVK, configures ACMXVK with MoltenVK, then
builds both engines and the interface into a local prefix:

```bash
./build-project-macos-cmake.sh
```

`build-project-macos-pcons.py` is available for the corresponding Pcons build.
Windows builds use an MSYS2 UCRT64 environment; see the Windows sections in the
backend READMEs for exact commands and runtime layout.

## Optional acceleration and AI

The normal OpenGL and Vulkan shader workflows work without CUDA. Enable only
the components needed by the target system:

| Component | Build option | Main requirements |
| --- | --- | --- |
| acidcam-gpu filters | `-DWITH_CUDA=ON` | NVIDIA CUDA Toolkit and CUDA-enabled OpenCV |
| OpenCV DNN effects | `-DWITH_OPENCV_DNN=ON` | OpenCV DNN and yaml-cpp |
| Deep Dream | `-DWITH_DEEP_DREAM=ON` | NVIDIA CUDA, cuDNN, and CUDA-enabled LibTorch |
| Stable Diffusion | `-DWITH_STABLE_DIFFUSION=ON` | libcurl, jsoncpp, and a compatible local `sd-server` |
| Audio / MIDI | `-DAUDIO=ON`, `-DMIDI=ON` | RtAudio / RtMidi and the backend's media dependencies |

For ACMX2, installing the CUDA Toolkit alone does not make a stock OpenCV build
CUDA-capable. Use a CUDA-enabled OpenCV build or configure `WITH_CUDA=OFF`.
ACMXVK's Deep Dream option is independent of the acidcam-gpu filter option.

## Downloads and documentation

- [ACMX documentation](https://lostsidedead.biz/acmx2/docs/)
- [Linux Flatpak](https://lostsidedead.biz/acmx2/release/)
- [Linux installation and first-use tutorial](https://lostsidedead.biz/acmx-tutorial/)
- [Windows beta](https://lostsidedead.biz/acmx-windows/)
- [Default Vulkan shader library](https://github.com/lostjared/shaders)
- [ACMX2 backend guide](ACMX2/README.md)
- [ACMXVK backend guide](ACMXVK/README.md)
- [Qt interface guide](ACMX2/interface/README.md)
- [MXWrite guide](MXWrite/README.md)

Regenerate the versioned Doxygen documentation with:

```bash
cd ACMX2
./build-docs.sh
```

## License

ACMX is distributed under the [BSD 2-Clause License](LICENSE). Optional
dependencies, models, shader packs, and NVIDIA components retain their own
licenses and distribution terms.
