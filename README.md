# acidcam-gpu

![screenshot](https://github.com/lostjared/acidcam-gpu/raw/main/acmx2.png)

[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Hardware: NVIDIA RTX](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%202070-green.svg)](https://www.nvidia.com/en-us/geforce/rtx/)
[![Framework: CUDA](https://img.shields.io/badge/Framework-CUDA%2012.x-76b900.svg)](https://developer.nvidia.com/cuda-zone)

# ACMX2 – Linux (NVIDIA GPU Required)

ACMX2 is distributed as a **Podman container** for Linux.
This makes installation simple and avoids dependency issues, but it **requires an NVIDIA GPU**.

---

## System Requirements

Before running ACMX2, your system must have:

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

## Step 1: Pull the ACMX2 Container Image

```bash
podman pull ghcr.io/lostjared/acmx2:latest
```

---

## Step 2: Go to the Podman Script Directory

```bash
cd podman
chmod +x run-acmx2.sh
```

---

## Step 3: Run ACMX2

```bash
./run-acmx2.sh
```

The script:
- Enables NVIDIA GPU acceleration
- Passes through camera and audio devices
- Opens the ACMX2 interface window on your desktop

---

## NVIDIA License Notice

This container includes NVIDIA CUDA libraries.

Use of the container is subject to the NVIDIA Deep Learning Container License:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

By pulling or running this container, you agree to NVIDIA’s license terms.

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
podman pull ghcr.io/lostjared/acmx2:latest
cd podman
./run-acmx2.sh
```

---

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
* **Environment:** Arch Linux (with `cuda`, `opencv` (compiled with CUDA support), `sdl2`, `sdl2-ttf` / `sdl2-mixer`, `glm`, `cmake`, `gcc` (g++), `qt6` (for the interface) installed via `pacman`).

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
cd ../../
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

## 2. Building the Image

Navigate to the directory containing your `Containerfile` and build the image. We tag it as `dev` to match the launch script.

```bash
podman build -t acmx2-cuda-opencv:dev -f Containerfile .
```

---

## 3. The Launch Script (`acmx2-run.sh`)

Use this script to launch the container. It handles the complex flags required for GPU, Webcam, and X11/Wayland compatibility.

Create a file named `acn20run.sh` on your host:

```bash
#!/bin/bash
xhost +local:docker
set -euo pipefail

IMAGE="ghcr.io/lostjared/acmx2:latest"

# 1. Get Host Audio Paths
PULSE_SOCKET="/run/user/$(id -u)/pulse/native"
PULSE_COOKIE="$HOME/.config/pulse/cookie"
HOST_SHARE="$HOME/container_share"

if command -v xhost >/dev/null 2>&1; then
  xhost +si:localuser:root >/dev/null 2>&1 || true
fi

# 2. Check if cookie exists
if [ ! -f "$PULSE_COOKIE" ]; then
    echo "Warning: Pulse Cookie not found at $PULSE_COOKIE"
fi

# 3. Ensure the share directory exists on host
mkdir -p "$HOST_SHARE"

exec podman run -it \
  --security-opt=label=disable \
  --net=host \
  --cap-add=SYS_NICE \
  --cap-add=SYS_RESOURCE \
  --device nvidia.com/gpu=all \
  --device /dev/video0 \
  --device /dev/snd \
  -e DISPLAY="${DISPLAY:-}" \
  -e QT_QPA_PLATFORM=xcb \
  -e XDG_RUNTIME_DIR=/tmp/xdg \
  -e PULSE_SERVER=unix:/tmp/pulse-socket \
  -e PULSE_COOKIE=/tmp/pulse-cookie \
  -v "$PULSE_SOCKET":/tmp/pulse-socket \
  -v "$PULSE_COOKIE":/tmp/pulse-cookie \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOST_SHARE":/root/share \
  "$IMAGE" bash -lc '
    mkdir -p /tmp/xdg
    chmod 700 /tmp/xdg
    # Double check audio inside before launching
    echo "Checking audio connection..."
    pactl info || echo "pactl failed, continuing anyway..."
    exec /opt/src/acidcam-gpu/ACMX2/interface/build/acmx2_interface
  '
```

Make it executable:

```bash
chmod +x acmx2-run.sh
```

---

## 4. Workflow & Usage

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

## Notes

This setup is designed to keep your development workflow fast and reproducible while maintaining full access to GPU acceleration, camera devices, and graphical output.

<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/1c3fb4db-39c3-4684-9256-88aa71f58711" />



