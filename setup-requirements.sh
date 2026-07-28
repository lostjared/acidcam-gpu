#!/usr/bin/env bash

set -Eeuo pipefail

if ! command -v pacman >/dev/null 2>&1; then
    echo "error: pacman was not found; this script requires Arch Linux or an Arch-based distribution." >&2
    exit 1
fi

if [[ ! -r /etc/arch-release ]]; then
    echo "warning: /etc/arch-release was not found; continuing because pacman is available." >&2
fi

if ((EUID == 0)); then
    PACMAN=(pacman)
else
    if ! command -v sudo >/dev/null 2>&1; then
        echo "error: run this script as root or install sudo." >&2
        exit 1
    fi
    PACMAN=(sudo pacman)
fi

# Full non-CUDA development environment, including the optional Qt launcher,
# audio/MIDI, CPU OpenCV DNN, WebP, and TIFF features.
packages=(
    # Compiler and build tools
    base-devel
    git
    cmake
    ninja
    pkgconf
    curl
    unzip

    # Stock CPU/OpenGL OpenCV and its commonly used modules
    opencv
    hdf5
    vtk
    fmt
    glew

    # SDL2 and Qt launcher
    # Arch replaced the sdl2 package with the SDL3-backed compatibility package.
    sdl2-compat
    sdl2_ttf
    sdl2_mixer
    sdl2_image
    qt6-base
    qt6-tools
    qt6-multimedia

    # Graphics, video, image, audio, and MIDI libraries
    glm
    mesa
    libglvnd
    ffmpeg
    rtaudio
    rtmidi
    libpulse
    libjpeg-turbo
    libpng

    # Optional non-CUDA ACMX2 features
    yaml-cpp
    libwebp
    libtiff

    # Runtime fonts
    fontconfig
    ttf-dejavu
    ttf-liberation
    noto-fonts
)

echo "Installing ACMX2 non-CUDA build and runtime dependencies..."
"${PACMAN[@]}" -Syu --needed "${packages[@]}"

if command -v fc-cache >/dev/null 2>&1; then
    fc-cache -f
fi

cat <<'EOF'

Pacman dependencies installed.

CUDA, NVIDIA utilities, opencv-cuda, and acidcam-gpu were intentionally omitted.
Configure the engine with:

  cmake -S ACMX2 -B build/acmx2-dev -DWITH_CUDA=OFF

ACMX2 also requires libmx2, which is not available as an official pacman
package. Build and install it from https://github.com/lostjared/libmx2 if it is
not already installed.
EOF
