# ACMX2 Podman Container (NVIDIA / CUDA Variant)

This directory contains everything needed to build and run ACMX2 inside a rootless Podman container on an Arch Linux host with an NVIDIA GPU.

> **Note:** ACMX2 itself does **not** require an NVIDIA GPU — the engine is built on OpenGL/SDL2 and runs on AMD, Intel, and Apple GPUs when configured with `-DWITH_CUDA=OFF`. The container recipe in this directory is the **CUDA-enabled variant** and is therefore NVIDIA-only. For non-NVIDIA hardware, use the native build instructions in the top-level [README.md](../README.md) with `-DWITH_CUDA=OFF`.

---

## Files

| File | Purpose |
|---|---|
| `Containerfile.arch` | Container build definition — installs all dependencies, clones repositories, compiles all components |
| `run-acmx2-arch.sh` | Launch script — grants X11/audio access, discovers video devices, passes the GPU through, and starts the container |

---

## Requirements

- **Podman** (rootless, with CDI configured for NVIDIA)
- **NVIDIA GPU** with the host driver installed
- **NVIDIA Container Device Interface (CDI)** — so Podman can see `nvidia.com/gpu=all`
- **PulseAudio** running on the host
- **X11** display server (Wayland with XWayland also works)
- A webcam or `/dev/video*` device (optional — the app will still launch without one)

---

## Building the Image

From inside this directory:

```bash
podman build -f Containerfile.arch -t acmx2-arch:latest .
```

The default CUDA architecture is `75` (Turing). Select the architecture for
your GPU at build time when needed:

```bash
podman build -f Containerfile.arch -t acmx2-arch:latest \
  --build-arg CUDA_ARCHITECTURES=86 .
```

Multiple architectures can be compiled into one image by quoting a
semicolon-separated list, for example
`--build-arg 'CUDA_ARCHITECTURES=75;86;89'`.

The build is split into labelled steps and will take 20–40 minutes on first run because it compiles CUDA kernels. Subsequent builds use the layer cache and are much faster.

### Build Steps Explained

```dockerfile
FROM archlinux:latest
```
Starts from the official rolling-release Arch Linux base image.

---

```dockerfile
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video,display
LABEL com.nvidia.volumes.needed="nvidia_driver"
```
Tells the NVIDIA container runtime which GPU capabilities to expose inside the container. These must be set at image build time so they are baked into the metadata.

---

```dockerfile
# Step 1 — Base system update + build tools
RUN pacman -Syu --noconfirm && pacman -S --noconfirm --needed \
    base-devel git cmake ninja pkg-config curl unzip
```
Updates the package database and installs the core build toolchain: `cmake`, `ninja` (the build backend), `pkg-config`, and `git` for cloning sources.

---

```dockerfile
# Step 2 — NVIDIA + CUDA
RUN pacman -S --noconfirm --needed nvidia-utils cuda
```
Installs the NVIDIA userspace libraries (`nvidia-utils`) and the full CUDA toolkit. The host kernel module is **not** installed here — the host driver is passed through at runtime.

---

```dockerfile
# Step 3 — OpenCV with CUDA
RUN pacman -S --noconfirm --needed opencv-cuda hdf5 vtk fmt glew
```
Installs the Arch community `opencv-cuda` package, which is pre-compiled against CUDA — saving you from a multi-hour OpenCV build from source. `hdf5`, `vtk`, and `fmt` are pulled in as required dependencies.

---

```dockerfile
# Step 4 — SDL2 + Qt6
RUN pacman -S --noconfirm --needed \
    sdl2-compat sdl2_ttf sdl2_mixer sdl2_image \
    qt6-base qt6-tools qt6-multimedia
```
SDL2 is used for the real-time OpenGL window and audio. Qt6 is used by the `acmx2_interface` GUI frontend.

---

```dockerfile
# Step 5 — Remaining libs
RUN pacman -S --noconfirm --needed \
    glm mesa libglvnd ffmpeg rtaudio pulseaudio libpulse
```
- `glm` — GLSL-compatible math library used in shaders
- `mesa` / `libglvnd` — OpenGL dispatch layer
- `ffmpeg` — video encode/decode
- `rtaudio` / `pulseaudio` / `libpulse` — audio output via PulseAudio

---

```dockerfile
# Step 6 — Fonts
RUN pacman -S --noconfirm --needed ttf-dejavu ttf-liberation noto-fonts \
    && fc-cache -fv
```
Installs fonts and rebuilds the font cache so the Qt GUI renders text correctly.

---

```dockerfile
ENV PATH="/opt/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/cuda/lib64:/usr/local/lib:/usr/lib:${LD_LIBRARY_PATH}"
ENV CUDACXX="/opt/cuda/bin/nvcc"
ENV BUILDJOBS=4
ARG CUDA_ARCHITECTURES=75
```
Sets CUDA paths and limits parallel compilation jobs to 4 to avoid out-of-memory kills during `nvcc` compilation. Increase `BUILDJOBS` if your machine has plenty of RAM.

---

```dockerfile
# Build & install libmx2
RUN git clone --depth=1 https://github.com/lostjared/libmx2.git
RUN cmake -S . -B build -G Ninja -DEXAMPLES=OFF -DOPENGL=ON ...
```
Clones and installs **libmx2** — the MX2/MXMOD 3D model and scene library that underpins ACMX2.

---

```dockerfile
# Clone acidcam-gpu repo
RUN git clone --depth=1 https://github.com/lostjared/acidcam-gpu.git
```
Fetches the main project. All subsequent build steps work from this clone.

---

```dockerfile
# Build acidcam-gpu library and CLI with the repository-local MXWrite
#   -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
```
The CUDA project builds the sibling `MXWrite/` source tree as a private
dependency of the `acidcam` CLI, so no system MXWrite package is required.
Architecture `75` targets Turing (RTX 2070/2080); pass a different
`CUDA_ARCHITECTURES` build argument for a different GPU generation:

| GPU Generation | `CUDA_ARCHITECTURES` value |
|---|---|
| Pascal (GTX 10xx) | `61` |
| Volta | `70` |
| Turing (RTX 20xx) | `75` |
| Ampere (RTX 30xx) | `86` |
| Ada Lovelace (RTX 40xx) | `89` |
| Hopper | `90` |
| Blackwell (RTX 50xx) | `120` |

---

```dockerfile
# Build ACMX2 command-line tool (-DAUDIO=ON)
# Build ACMX2 Qt interface
# Copy data/ alongside the build
```
Builds the command-line `acmx2` tool with audio support enabled, then the `acmx2_interface` Qt GUI.

---

```dockerfile
# Check out shaders and download models
RUN git clone --depth=1 https://github.com/lostjared/shaders.git files/shaders
RUN curl -L https://lostsidedead.biz/acmx2/models.zip ...
```
Checks out the shader collection directly into `/opt/src/files/shaders` and
downloads the model pack into `/opt/src/files/models`. The image exports
`ACMX2_SHADER_PATH` and `ACMX2_PATH` so the CLI can locate the installed assets.
In the GUI, select `/opt/src/files/shaders` as the shader directory the first
time it starts.

---

```dockerfile
CMD ["acmx2_interface"]
```
The default command starts the Qt GUI. Override this with `bash` to drop into an interactive shell instead.

---

## Running the Container

Make the script executable once:

```bash
chmod +x run-acmx2-arch.sh
```

Then launch:

```bash
./run-acmx2-arch.sh
```

---

## Run Script Explained — `run-acmx2-arch.sh`

```bash
xhost +local:docker
```
Opens X11 to local connections before the `set -euo pipefail` guard runs (xhost exits non-zero on some systems, so it must appear first).

---

```bash
set -euo pipefail
IMAGE="acmx2-arch:latest"
```
Enables strict error handling — any unset variable or failed command aborts the script. `IMAGE` is the tag you built with `podman build -t`.

---

```bash
PULSE_SOCKET="/run/user/$(id -u)/pulse/native"
PULSE_COOKIE="$HOME/.config/pulse/cookie"
HOST_SHARE="$HOME/container_share"
mkdir -p "$HOST_SHARE"
```
Locates the PulseAudio socket and authentication cookie for the current user. `HOST_SHARE` is a directory on your host that is bind-mounted into `/root/share` inside the container — use it to pass files in and out (videos, presets, exports).

---

```bash
if command -v xhost >/dev/null 2>&1; then
  xhost +si:localuser:root >/dev/null 2>&1 || true
fi
```
Grants the container's `root` user access to your X11 display. The `|| true` prevents the script from aborting if `xhost` fails (e.g. on Wayland-only systems).

---

```bash
VIDEO_DEVICES=""
for i in 0 1 2 3 4 5 6 7 8 9; do
  if [ -e "/dev/video$i" ]; then
    VIDEO_DEVICES="$VIDEO_DEVICES --device /dev/video$i"
  fi
done
```
Scans `/dev/video0` through `/dev/video9` and builds a list of `--device` flags. Any webcam or capture card present on the host is automatically made available inside the container. Remove this block entirely if you do not need camera input.

---

```bash
exec podman run -it \
```
Replaces the shell process with `podman run` (no leftover parent process). `-it` allocates a pseudo-TTY and keeps stdin open — required for interactive use and clean Ctrl-C handling.

---

```bash
  --security-opt=label=disable \
```
Disables SELinux/AppArmor label confinement for this container. Required for GPU and X11 passthrough on most systems. Remove only if you have a custom policy that permits all the device mounts below.

---

```bash
  --net=host \
```
Shares the host network stack. Needed so the container can reach PulseAudio's Unix socket at its native path and simplifies X11 display resolution.

---

```bash
  --cap-add=SYS_NICE \
  --cap-add=SYS_RESOURCE \
```
Allows the process inside the container to set real-time thread priorities — used by the audio subsystem to reduce latency.

---

```bash
  --device nvidia.com/gpu=all \
```
Passes all NVIDIA GPUs through via CDI (Container Device Interface). Requires `nvidia-container-toolkit` configured for CDI on the host. To target a specific GPU use `nvidia.com/gpu=0`.

---

```bash
  $VIDEO_DEVICES \
  --device /dev/snd \
```
Passes the detected webcam devices and the ALSA sound card node into the container.

---

```bash
  -e DISPLAY="${DISPLAY:-}" \
  -e QT_QPA_PLATFORM=xcb \
  -e XDG_RUNTIME_DIR=/tmp/xdg \
```
Forwards your display address. `QT_QPA_PLATFORM=xcb` forces Qt to use X11 (XCB backend) rather than trying Wayland. `XDG_RUNTIME_DIR` is set to a writable path inside the container.

---

```bash
  -e PULSE_SERVER=unix:/tmp/pulse-socket \
  -e PULSE_COOKIE=/tmp/pulse-cookie \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video,display \
```
Points PulseAudio to the bind-mounted socket and cookie. The NVIDIA variables reinforce GPU capability exposure at runtime.

---

```bash
  -v "$PULSE_SOCKET":/tmp/pulse-socket \
  -v "$PULSE_COOKIE":/tmp/pulse-cookie \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOST_SHARE":/root/share \
  -v /usr/share/fonts:/usr/share/fonts:ro \
```
Bind mounts:

| Host path | Container path | Purpose |
|---|---|---|
| `$PULSE_SOCKET` | `/tmp/pulse-socket` | PulseAudio IPC socket |
| `$PULSE_COOKIE` | `/tmp/pulse-cookie` | PulseAudio auth cookie |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | X11 display sockets |
| `$HOME/container_share` | `/root/share` | File exchange directory |
| `/usr/share/fonts` | `/usr/share/fonts` (read-only) | Host font cache |

---

```bash
  "$IMAGE" bash -lc '
    mkdir -p /tmp/xdg
    chmod 700 /tmp/xdg
    echo "Checking audio connection..."
    pactl info || echo "pactl failed, continuing anyway..."
    exec acmx2_interface
  '
```
Inside the container: creates the XDG runtime directory, verifies the PulseAudio connection, then `exec`s the ACMX2 Qt interface (replacing the shell so the process has PID 1 within the container init scope).

To launch the command-line tool instead, change the last line to:

```bash
    exec acmx2
```

Or drop to a shell for debugging:

```bash
  "$IMAGE" bash
```

---

## Sharing Files with the Container

Anything placed in `~/container_share` on your host is accessible at `/root/share` inside the container. Use this to load source videos or retrieve exported files.

---

## Common Customisations

| What to change | Where |
|---|---|
| Target GPU architecture | Pass `--build-arg CUDA_ARCHITECTURES=<value>` to `podman build` |
| Parallel build jobs | `Containerfile.arch` — `ENV BUILDJOBS=4` |
| Select a specific GPU | `run-acmx2-arch.sh` — `--device nvidia.com/gpu=0` |
| Disable camera passthrough | `run-acmx2-arch.sh` — remove the `VIDEO_DEVICES` loop and `$VIDEO_DEVICES` flag |
| Launch CLI instead of GUI | `run-acmx2-arch.sh` — replace `acmx2_interface` with `acmx2` in the final `bash -lc` block |
| Use a different image tag | `run-acmx2-arch.sh` — change `IMAGE="acmx2-arch:latest"` |
