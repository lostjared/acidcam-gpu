# acidcam-gpu

[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Hardware: NVIDIA RTX](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%202070-green.svg)](https://www.nvidia.com/en-us/geforce/rtx/)
[![Framework: CUDA](https://img.shields.io/badge/Framework-CUDA%2012.x-76b900.svg)](https://developer.nvidia.com/cuda-zone)

**acidcam-gpu** is a high-performance, real-time video manipulation engine designed to push the boundaries of psychedelic and fractal art. Part of the **ACMX2** and **libmx2** ecosystem, it offloads complex glitch filters to **NVIDIA GPUs**, enabling fluid, high-resolution visual transformations at 60+ FPS.

## 🚀 Purpose & Vision
The original project brought a massive library of "glitch" filters to digital artists. However, as resolutions climbed to 4K and filter stacks became more complex, CPU-based processing hit a bottleneck. 

**acidcam-gpu** solves this by:
* **Parallelizing the Chaos:** Using custom CUDA kernels to process millions of pixels simultaneously.

## 🛠 Tech Stack
* **Language:** C++20
* **Parallel Computing:** NVIDIA CUDA (Optimized for **RTX 2070**)
* **Graphics API:** OpenGL / GLFW (Hardware-accelerated rendering)
* **Format Support:** Native **MXMOD** 3D model parsing for real-time geometry glitching.

## ⚡ Why NVIDIA & CUDA?
This project is built specifically for the NVIDIA ecosystem to leverage:
* **Shared Memory:** Fast on-chip memory to speed up neighborhood-based filters.
* **Massive Throughput:** Harnessing thousands of CUDA cores to apply multiple glitch layers in a single pass.
* **Zero-Copy Interop:** High-speed texture sharing between CUDA and OpenGL.

## 📦 Installation & Environment
This project is developed and tested on **Bazzite Linux** using **Arch Linux** containers via **Distrobox**.

### Prerequisites
* **NVIDIA GPU:** RTX 20-series or newer.
* **Drivers:** NVIDIA Proprietary Drivers (v535+).
* **Environment:** Arch Linux (with `cuda`, `glfw`, and `cmake` installed via `pacman`).

### Build Instructions
```bash
# Clone the repository
git clone https://github.com/lostjared/acidcam-gpu.git
cd acidcam-gpu

# Build the project
mkdir build && cd build
cmake ..
make -j$(nproc)

