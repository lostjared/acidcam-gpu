# acidcam-gpu

![screenshot](https://github.com/lostjared/acidcam-gpu/raw/main/acmx2.png)

[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Hardware: NVIDIA RTX](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%202070-green.svg)](https://www.nvidia.com/en-us/geforce/rtx/)
[![Framework: CUDA](https://img.shields.io/badge/Framework-CUDA%2012.x-76b900.svg)](https://developer.nvidia.com/cuda-zone)

# ACMX2 – Linux (NVIDIA GPU Required)
![screenshot](https://github.com/lostjared/acidcam-gpu/blob/main/image.jpg)

Technical Documentation:

[Project Documentation](https://lostsidedead.biz/acmx2-explained.html)

[GPU Filters Explained](https://lostsidedead.biz/acmx2/filter_browser.html)
 
[Example Shaders](https://lostsidedead.biz/acmx2/shader_browser.html)


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

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/110f4959-67ff-4cef-aa0c-f036e6ee78ba" />



