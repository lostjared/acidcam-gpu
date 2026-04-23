#!/bin/sh
#
# install-dep.sh -- Install ACMX2 build dependencies on macOS via Homebrew.
#
# Required packages:
#   cmake, pkg-config        - build system
#   sdl2, sdl2_ttf,          - windowing, fonts, mixer, image loading
#   sdl2_mixer, sdl2_image
#   glm                      - GLSL-style math (header-only)
#   opencv                   - capture / image processing (no CUDA on macOS)
#   ffmpeg                   - video decode / encode / muxing
#   qt6                      - GUI (acmx2_interface, midi-map)
#
# Optional packages (only needed if building with -DAUDIO=ON / -DMIDI=ON):
#   rtaudio                  - real-time audio capture for shader reactivity
#   rtmidi                   - MIDI controller input
#
# CUDA is NOT available on macOS, so ACMX2 is always built with
# -DWITH_CUDA=OFF; OpenCV does not need CUDA support either.

set -e

if ! command -v brew >/dev/null 2>&1; then
    echo "error: Homebrew is not installed."
    echo "install it from https://brew.sh and re-run this script."
    exit 1
fi

echo "==> updating Homebrew"
brew update

echo "==> installing required packages"
brew install \
    cmake \
    pkg-config \
    sdl2 \
    sdl2_ttf \
    sdl2_mixer \
    sdl2_image \
    glm \
    opencv \
    ffmpeg \
    qt6

echo "==> installing optional packages (audio + MIDI)"
brew install rtaudio rtmidi || true

echo "done. dependencies installed."
