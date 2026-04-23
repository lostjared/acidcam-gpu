# Building ACMX2 on macOS

ACMX2 builds and runs on macOS (Intel and Apple Silicon) using OpenGL
4.1 backed by Metal. CUDA, NVENC, and the CUDA-accelerated OpenCV
modules are **not** available on macOS, so the engine is built with
`-DWITH_CUDA=OFF` automatically by the scripts in this directory. All
shader-based features still work; only the CUDA GPU-filter pipeline is
omitted.

The Apple OpenGL 4.1 driver does not support `glProgramBinary`, so the
shader **binary cache is disabled by default at runtime** on macOS. The
Qt6 interface also greys out the cache-related options on Apple builds.

## One-shot install

From an empty working directory:

```sh
./install.sh
```

This runs `install-dep.sh` (Homebrew packages) and then `build-macos.sh`
(clone, build, and `sudo make install`).

## Step-by-step

1. **Install Homebrew packages**

   ```sh
   ./install-dep.sh
   ```

   Required: `cmake`, `pkg-config`, `sdl2`, `sdl2_ttf`, `sdl2_mixer`,
   `sdl2_image`, `glm`, `opencv`, `ffmpeg`, `qt6`.
   Optional (only if you build with `-DAUDIO=ON` / `-DMIDI=ON`):
   `rtaudio`, `rtmidi`.

2. **Build and install the engine + GUI**

   ```sh
   ./build-macos.sh
   ```

   The script clones `libmx2` and `ACMX2`, builds `MXWrite`, then the
   `acmx2` CLI engine, then `acmx2_interface` (Qt6 GUI). All targets
   are installed to `/usr/local/`.

3. **Download the macOS-compatible shader pack**

   The default shader library targets desktop NVIDIA OpenGL drivers
   and contains shaders that the Apple GL 4.1 driver rejects (or in
   some cases crashes on). Use the curated macOS pack instead:

   ```
   [macos_link]
   ```

   Unzip it somewhere and pass the path with `-s` or set it in the
   GUI's *Settings → Shader Library Path*.

4. **Run**

   ```sh
   acmx2 -p /usr/local/share/acmx2/data -s /path/to/macos-shaders/index.txt -d 0
   acmx2_interface
   ```

## What the scripts pass to CMake

`build-macos.sh` invokes CMake with:

| Flag | Value | Reason |
|------|-------|--------|
| `-DWITH_CUDA` | `OFF`            | No CUDA on macOS |
| `-DAUDIO`     | `OFF`            | Toggle on if you `brew install rtaudio` |
| `-DMIDI`      | `OFF`            | Toggle on if you `brew install rtmidi` |
| `-DCMAKE_PREFIX_PATH` | `$(brew --prefix);$(brew --prefix)/opt/qt6;/usr/local` | Find Qt6 + Homebrew libs on Apple Silicon |
| `-DCMAKE_INSTALL_RPATH` | `$(brew --prefix)/lib;/usr/local/lib` | Resolve dylibs at runtime |

## Troubleshooting

- **`Qt6 not found`** — make sure `brew install qt6` succeeded; the
  scripts already pass `$(brew --prefix)/opt/qt6` to `CMAKE_PREFIX_PATH`.
- **Shader scan crashes mid-run** — the Apple GL driver can crash on
  certain shaders. Re-run *Remove Broken Shaders* from the interface;
  the scanner writes a per-shader crash marker and will skip the
  offender on the next run.
- **`glGetString` returned NULL / segfault on launch** — usually means
  no GL context was created (e.g. running headless over SSH without a
  display). Run from a local Terminal window.
