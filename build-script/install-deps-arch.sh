#!/bin/bash

# ---- Step 1: Base system update + build tools ----
pacman -Syu --noconfirm && \
    pacman -S --noconfirm --needed \
    base-devel git cmake ninja pkg-config curl unzip \
    && rm -rf /var/cache/pacman/pkg/*

# ---- Step 2: NVIDIA + CUDA ----
pacman -S --noconfirm --needed \
    nvidia-utils cuda \
    && rm -rf /var/cache/pacman/pkg/*

# ---- Step 3: OpenCV with CUDA ----
pacman -S --noconfirm --needed \
    opencv-cuda hdf5 vtk fmt glew \
    && rm -rf /var/cache/pacman/pkg/*

# ---- Step 4: SDL2 + Qt6 ----
pacman -S --noconfirm --needed \
    sdl2 sdl2_ttf sdl2_mixer sdl2_image \
    qt6-base qt6-tools qt6-multimedia \
    && rm -rf /var/cache/pacman/pkg/*

# ---- Step 5: Remaining libs ----
pacman -S --noconfirm --needed \
    glm mesa libglvnd ffmpeg \
    rtaudio rtmidi pulseaudio libpulse  2>/dev/null \
    && rm -rf /var/cache/pacman/pkg/*

# ---- Step 6: Fonts ----
pacman -S --noconfirm --needed \
    ttf-dejavu ttf-liberation noto-fonts \
    && rm -rf /var/cache/pacman/pkg/* \
    && fc-cache 

