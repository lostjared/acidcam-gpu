# Dependencies and licenses

This document catalogs the direct build dependencies and notable supporting
components used by acidcam-gpu, ACMX2, and the related tools in this
repository. It is based on the repository's CMake files, build scripts, and
source includes as of July 27, 2026.

## Project license

The code authored for this repository is licensed under the
[BSD 2-Clause License](LICENSE). The ACMX2 command-line engine is explicitly
licensed under BSD 2-Clause in [`ACMX2/LICENSE`](ACMX2/LICENSE).

The Qt graphical interface under `ACMX2/interface/` is a deliberate
directory-specific exception: it remains licensed under the
[GNU General Public License version 3](ACMX2/interface/LICENSE). Its companion
`midi-map` tool is part of that GPLv3-licensed interface tree.

The BSD 2-Clause license applies only to the project-owned code. It does not
replace, override, sublicense, or otherwise alter the licenses of third-party
libraries, drivers, tools, fonts, models, codecs, or other components.

## Builder and distributor responsibility

This project has optional features that are selected at compile time. By
enabling an optional feature, compiling against an optional component, or
redistributing a resulting binary or container, **the person or organization
performing that action accepts responsibility for identifying and complying
with the licenses, notices, source-offer requirements, patent restrictions,
codec restrictions, and other terms that apply to the exact component
versions and build configuration chosen**.

In particular:

- Enabling a feature does not grant a license to its dependencies.
- The project authors do not select or accept third-party license terms on a
  builder's or distributor's behalf.
- Package maintainers can compile dependencies with different options, so the
  effective license of an installed binary can differ from the upstream
  project's usual or default license.
- Anyone distributing binaries should retain required notices, inspect the
  actual packages and shared libraries being shipped, and provide relinking
  facilities or corresponding source when a dependency's license requires it.
- Codec availability does not itself grant patent rights. Patent and media
  format rules vary by country and use case.

This catalog is provided for convenience and is not legal advice. The license
files supplied with the exact dependency versions being used are authoritative.

## ACMX2 core dependencies

These components are required to build the `ACMX2/acmx2` executable unless the
table says otherwise.

| Component | How it is used | Upstream license |
| --- | --- | --- |
| libmx2 (`mx`, `mxgl`) | SDL/OpenGL application and rendering framework | MIT |
| SDL2 | Windowing, input, and multimedia support through libmx2 | zlib |
| SDL2_ttf | Font rendering through libmx2 | zlib |
| SDL2_mixer | Audio/media support through libmx2 | zlib |
| GLM | Header-only vector and matrix mathematics used directly by ACMX2 | MIT |
| OpenGL | Core graphics API | The API is an open standard; the implementation/driver license varies. Mesa is primarily MIT, while proprietary GPU drivers use vendor terms. |
| OpenCV | Video capture and image processing (`core`, `imgproc`, `highgui`, `videoio`) | Apache-2.0 for OpenCV 4.5.0 and later; older OpenCV releases use BSD-3-Clause |
| FFmpeg libraries | Decode, encode, mux, scale, and resample (`libavcodec`, `libavformat`, `libavutil`, `libswscale`, `libswresample`) | Normally LGPL-2.1-or-later; becomes GPL-2.0-or-later when built with GPL components, and may include separately licensed/nonfree components |
| MXWrite | Repository-local video writer built from `MXWrite/` | BSD-2-Clause as project-owned code covered by the root license |
| C/C++ threads | Concurrency via CMake `Threads::Threads` and the C++ standard library | Platform/toolchain dependent; commonly covered by the system C library and compiler runtime licenses |

libmx2 is built with its OpenGL module by the provided instructions. Its own
configuration can bring in GLAD (MIT), libpng (libpng-2.0), zlib (Zlib),
libjpeg/libjpeg-turbo (IJG and BSD-style licenses), FreeType
(FTL or GPL-2.0-or-later), and SDL2_image (zlib). Those are transitive
dependencies and must be checked against the particular libmx2 build.

### ACMX2 optional compile-time components

| CMake option or target | Component(s) | Purpose | Upstream license/terms |
| --- | --- | --- | --- |
| `WITH_CUDA=ON` | NVIDIA CUDA Toolkit and runtime | CUDA kernels, CUDA/OpenGL interop, and GPU decode/filter paths | Proprietary; NVIDIA CUDA Toolkit EULA and any applicable driver terms |
| `WITH_CUDA=ON` | acidcam-gpu | Repository GPU-filter library | BSD-2-Clause |
| `WITH_CUDA=ON` | CUDA-enabled OpenCV modules | `cudaimgproc`, `cudawarping`, and `cudaarithm` | OpenCV license above, plus the terms of CUDA and any bundled third-party code |
| `WITH_OPENCV_DNN=ON` | OpenCV DNN | ONNX model inference | OpenCV license above |
| `WITH_OPENCV_DNN=ON` | yaml-cpp | Model configuration files | MIT |
| `AUDIO=ON` | RtAudio | Live audio capture/reactivity | Custom permissive, MIT-like RtAudio license; retain its notice |
| `AUDIO=ON` | Audio backend selected by RtAudio, such as PulseAudio, ALSA, or JACK | Platform audio I/O | Backend dependent; PulseAudio is LGPL-2.1-or-later, ALSA library is LGPL-2.1-or-later, and JACK is GPL-2.0-or-later with the project library exception |
| `MIDI=ON` | RtMidi | MIDI controller input | Custom permissive, MIT-like RtMidi license; upstream also requests that modifications be sent to the maintainer |
| `WEBP=ON` | libwebp | HDR snapshots in WebP format | BSD-3-Clause |
| `TIFF=ON` | libtiff | 16-bit TIFF snapshots | libtiff license (BSD-like permissive license) |

`WITH_CUDA` defaults to `ON` in `ACMX2/CMakeLists.txt`, but it can be disabled
with `-DWITH_CUDA=OFF`. `WITH_OPENCV_DNN`, `AUDIO`, `MIDI`, `WEBP`, and `TIFF`
default to `OFF`.

## Standalone acidcam-gpu library

The `acidcam-gpu/` CMake project is the optional CUDA implementation used by
ACMX2. When this subproject is built on its own, its direct dependencies are:

| Component | Required | Upstream license/terms |
| --- | --- | --- |
| NVIDIA CUDA Toolkit/runtime | Yes | Proprietary NVIDIA CUDA Toolkit EULA |
| OpenCV with CUDA (`core`, `imgproc`, `highgui`, `videoio`, `cudaimgproc`) | Yes | Apache-2.0 for 4.5.0+; BSD-3-Clause for older releases |
| MXWrite | Yes | BSD-2-Clause as repository-owned code |
| FFmpeg development libraries | Yes at configure time | LGPL/GPL/build-dependent terms described above |
| POSIX/C++ threads | Yes | Platform/toolchain dependent |

## Graphical interface and companion tools

These are separate executables and are not needed to compile the ACMX2
command-line engine.

| Target | Component(s) | Upstream license |
| --- | --- | --- |
| `ACMX2/interface/acmx2_interface` | GPL-3.0 project code; Qt Core, Gui, Widgets, Concurrent, and Network (Qt 6 preferred; Qt 5.15 fallback) | Interface: GPL-3.0. Qt: commercial license or applicable open-source terms; common open-source Qt packages offer LGPL-3.0 and/or GPL terms. Confirm the exact Qt edition and modules distributed. |
| `ACMX2/interface/midi-map` | GPL-3.0 project code; Qt 6 Widgets and RtMidi | Tool: GPL-3.0. Qt terms as above; custom permissive RtMidi license |
| `ACMX2/shader_generator/shader_generator` | libcurl | curl license (MIT/X11-style permissive license) |

Qt's LGPL terms can impose obligations on binary distribution, including
allowing recipients to replace or relink the Qt libraries. Static linking
usually requires additional compliance steps. A commercial Qt license is a
separate option and must be obtained from Qt under its terms.

## Build-time and packaging tools

These tools build or package the project but are not normally linked into its
executables.

| Tool | Role | Upstream license |
| --- | --- | --- |
| CMake | Build configuration | BSD-3-Clause |
| pkg-config | Dependency discovery | GPL-2.0-or-later for the original `pkg-config` implementation |
| pkgconf | Common alternative implementation of `pkg-config` | ISC |
| Ninja | Optional build runner used by scripts/container | Apache-2.0 |
| GNU Make | Alternative build runner | GPL-3.0-or-later |
| GCC | C/C++ compiler | GPL-3.0-or-later; generated binaries are generally covered by the GCC Runtime Library Exception where applicable |
| Clang/LLVM | Alternative C/C++ compiler | Apache-2.0 WITH LLVM-exception |
| NVIDIA `nvcc` | CUDA compiler | NVIDIA CUDA Toolkit EULA |
| Git | Source retrieval | GPL-2.0-only |
| curl command-line tool | Downloads optional content in the container recipe | curl license |
| unzip | Extracts downloaded archives | Info-ZIP license |
| Arch Linux container base and packages | Optional container build environment | Per-package licenses; the image is an aggregation and has no single license replacing its packages' licenses |

## Bundled and downloaded content

Software-library licensing is not the only consideration:

- `ACMX2/data/font.ttf` and `ACMX2/interface/data/font.ttf` identify themselves
  as the **Hack** font. Hack is distributed under the MIT License, and portions
  derived from Bitstream Vera retain the Bitstream Vera license and notices.
- ONNX files under `models/` are optional model data, not BSD-licensed merely
  because they are present beside this project. The repository does not
  currently contain model-by-model provenance or license notices. A builder or
  distributor must establish the source, license, and permitted uses of every
  model selected or redistributed.
- Shader packs and model archives downloaded by `podman/Containerfile.arch`
  are external content. Their inclusion in a container does not place them
  under BSD-2-Clause; their own licenses and notices must be reviewed before
  redistribution.
- GPU drivers, camera drivers, fonts installed from the operating system,
  codecs, and other runtime plugins retain their respective vendor or upstream
  terms.

## Primary license references

- [Project BSD 2-Clause license](LICENSE)
- [ACMX2 interface GPLv3 license](ACMX2/interface/LICENSE)
- [libmx2 license and repository](https://github.com/lostjared/libmx2)
- [SDL license](https://github.com/libsdl-org/SDL/blob/main/LICENSE.txt)
- [OpenCV license](https://github.com/opencv/opencv/blob/4.x/LICENSE)
- [FFmpeg legal and license information](https://ffmpeg.org/legal.html)
- [Qt licensing](https://doc.qt.io/qt-6/licensing.html)
- [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html)
- [RtAudio license](https://github.com/thestk/rtaudio/blob/master/LICENSE)
- [RtMidi license](https://github.com/thestk/rtmidi/blob/master/LICENSE)
- [yaml-cpp license](https://github.com/jbeder/yaml-cpp/blob/master/LICENSE)
- [libwebp license](https://github.com/webmproject/libwebp/blob/main/COPYING)
- [libtiff license](https://gitlab.com/libtiff/libtiff/-/blob/master/LICENSE.md)
- [curl license](https://curl.se/docs/copyright.html)
- [Hack font license](https://github.com/source-foundry/Hack/blob/master/LICENSE.md)
