#!/usr/bin/env bash

set -Eeuo pipefail

OS_NAME="$(uname -s)"

if [[ "$OS_NAME" == "Linux" ]]; then
    # ==========================================
    # LINUX (ARCH) INSTALLATION PATH
    # ==========================================
    
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

    packages=(
        base-devel git cmake ninja pkgconf curl unzip pciutils
        hdf5 vtk fmt glew
        sdl2-compat sdl2_ttf sdl2_mixer sdl2_image
        qt6-base qt6-tools qt6-multimedia
        glm mesa libglvnd ffmpeg rtaudio rtmidi libpulse
        libjpeg-turbo libpng yaml-cpp libwebp libtiff
        fontconfig ttf-dejavu ttf-liberation noto-fonts
    )

    CMAKE_CUDA_FLAG="-DWITH_CUDA=OFF"
    CUDA_MSG="CUDA, NVIDIA utilities, opencv-cuda, and acidcam-gpu were intentionally omitted."

    # Detect NVIDIA GPU via runtime tools or PCI bus
    if command -v nvidia-smi >/dev/null 2>&1 || (command -v lspci >/dev/null 2>&1 && lspci | grep -i -q "NVIDIA"); then
        echo "NVIDIA hardware detected. Swapping opencv for opencv-cuda..."
        packages+=(opencv-cuda)
        CMAKE_CUDA_FLAG="-DWITH_CUDA=ON"
        CUDA_MSG="NVIDIA hardware detected. opencv-cuda included in installation."
    else
        echo "No NVIDIA hardware detected. Utilizing standard opencv..."
        packages+=(opencv)
    fi

    echo "Installing ACMX2 build and runtime dependencies for Arch Linux..."
    "${PACMAN[@]}" -Syu --needed "${packages[@]}"

    if command -v fc-cache >/dev/null 2>&1; then
        fc-cache -f
    fi

elif [[ "$OS_NAME" == "Darwin" ]]; then
    # ==========================================
    # macOS INSTALLATION PATH
    # ==========================================
    
    if ! command -v brew >/dev/null 2>&1; then
        echo "error: Homebrew was not found. Please install it first from https://brew.sh/" >&2
        exit 1
    fi

    packages=(
        git cmake ninja pkgconf curl unzip
        opencv hdf5 vtk fmt glew
        sdl2 sdl2_ttf sdl2_mixer sdl2_image
        qt glm ffmpeg rtaudio rtmidi
        jpeg-turbo libpng yaml-cpp webp libtiff fontconfig
    )

    CMAKE_CUDA_FLAG="-DWITH_CUDA=OFF"
    CUDA_MSG="CUDA is not supported on macOS."

    echo "Installing ACMX2 build and runtime dependencies for macOS..."
    brew install "${packages[@]}"

else
    echo "error: Unsupported operating system ($OS_NAME). This script supports Linux (Arch) and macOS." >&2
    exit 1
fi

# ==========================================
# POST-INSTALLATION INSTRUCTIONS
# ==========================================

cat <<EOF

Dependencies installed successfully.

$CUDA_MSG
Configure the engine with:

  cmake -S ACMX2 -B build/acmx2-dev $CMAKE_CUDA_FLAG

ACMX2 also requires libmx2, which is not available via the default package managers. 
Build and install it from https://github.com/lostjared/libmx2 if it is not already installed.
EOF
