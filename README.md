# acidcam-gpu

[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Hardware: NVIDIA RTX](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%202070-green.svg)](https://www.nvidia.com/en-us/geforce/rtx/)
[![Framework: CUDA](https://img.shields.io/badge/Framework-CUDA%2012.x-76b900.svg)](https://developer.nvidia.com/cuda-zone)

# In Early stages of Development


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
* **Environment:** Arch Linux (with `cuda`, `opencv`, and `cmake` installed via `pacman`).

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
qmake6 ..
make -j $(nproc)
cp -rf ../data/ .
cd ../../
echo "completed..."
```

Early Example (as a GIF)

![Project Demo](https://github.com/lostjared/acidcam-gpu/blob/main/output.gif)

# Latest Shader Pack

https://lostsidedead.biz/activity/shader.pack.2025.12.07.zip

# Latest 3D Geometry 

https://lostsidedead.biz/activity/models.2025.12.08.zip

# ACMX2 Container Environment Documentation

This guide covers the setup and usage of the ACMX2 / Acidcam-GPU development environment on **Bazzite**. It details how to build the container, manage file paths, and ensure full hardware access (NVIDIA GPU, Webcam, and X11 Display).

---

## 1. Host System Setup

Before launching the container, you must establish a specific folder structure on your Bazzite host. This ensures your code is persistent and files can be easily transferred.

Open a terminal on your host and run:

```bash
# Create a folder for your source code (if not already present)
mkdir -p ~/ACMX2

# Create a "scratch pad" for transferring files (videos, models, loose shaders)
mkdir -p ~/container_share
```

**Folder purposes:**

- `~/ACMX2`  
  Permanent C++ source code. Edit files here using host tools (VS Code, etc.).

- `~/container_share`  
  Shared volume. Files placed here are visible to both the host and the container.

---

## 2. Building the Image

Navigate to the directory containing your `Containerfile` and build the image. We tag it as `dev` to match the launch script.

```bash
podman build -t acmx2-cuda-opencv:dev -f Containerfile .
```

---

## 3. The Launch Script (`run.sh`)

Use this script to launch the container. It handles the complex flags required for GPU, Webcam, and X11/Wayland compatibility.

Create a file named `run.sh` on your host:

```bash
#!/bin/bash

# Authorize local X11 connections (required for GUI)
xhost +local:

podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  --device /dev/video0 \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=/tmp/xdg \
  -e QT_QPA_PLATFORM=xcb \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/xdg/$WAYLAND_DISPLAY \
  -v ~/ACMX2:/root/share \
  -w /opt/src/acidcam-gpu/ACMX2/interface/build \
  localhost/acmx2-cuda-opencv:dev
```

Make it executable:

```bash
chmod +x run.sh
```

---

## 4. Workflow & Usage

### A. Development Loop

Since your source code is mounted, you do not need to edit files inside the container.

- **Edit**  
  Modify `~/ACMX2/main.cpp` (or other files) on your host machine.

- **Build**  
  Inside the container terminal:
  ```bash
  make
  ```

- **Run**
  ```bash
  ./interface
  ```

### B. File Paths (Shaders & Models)

The container uses different paths than your host. Ensure your C++ code looks in the correct locations.

- **Source code & shaders**  
  `/opt/src/acidcam-gpu/ACMX2`  
  Example:
  ```
  /opt/src/acidcam-gpu/ACMX2/shaders/my_shader.glsl
  ```

- **External assets (models, videos)**  

  1. Copy files to `~/container_share` on the host.
  2. Access them in the container from:
     ```
     /root/share/test_video.mp4
     ```

### C. Saving Output

- **Source code**  
  Changes are saved automatically to `~/ACMX2` on the host.

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
Re-run `./run.sh` to refresh permissions.

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



