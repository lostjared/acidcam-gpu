#!/bin/sh
#
# build-macos.sh -- Build and install ACMX2 + Qt6 interface on macOS.
#
# This script builds the full stack in a sibling work directory:
#
#     <cwd>/libmx2               (cloned from github.com/lostjared/libmx2)
#     <cwd>/acidcam-gpu/         (cloned from github.com/lostjared/acidcam-gpu)
#         ACMX2/
#             MXWrite/           (built first, in-tree)
#             build/             (acmx2 CLI engine)
#             interface/build/   (Qt6 GUI: acmx2_interface)
#
# CUDA is not available on macOS, so the engine is built with
# -DWITH_CUDA=OFF. The shader-binary cache is also disabled by default
# at runtime on Apple targets because the Apple Metal-backed OpenGL 4.1
# driver does not support glProgramBinary.
#
# Notes for Apple Silicon (M1/M2/M3): Homebrew installs libraries under
# /opt/homebrew on arm64 macs and under /usr/local on Intel macs. We
# detect this and pass CMAKE_PREFIX_PATH so CMake finds Qt6, SDL2, etc.

set -e

# ---- detect Homebrew prefix --------------------------------------------------
if command -v brew >/dev/null 2>&1; then
    BREW_PREFIX="$(brew --prefix)"
elif [ -d /opt/homebrew ]; then
    BREW_PREFIX=/opt/homebrew
else
    BREW_PREFIX=/usr/local
fi

CMAKE_PREFIX="${BREW_PREFIX};${BREW_PREFIX}/opt/qt6;/usr/local"
JOBS="$(sysctl -n hw.ncpu)"
RPATH="${BREW_PREFIX}/lib;/usr/local/lib"

echo "==> Homebrew prefix : ${BREW_PREFIX}"
echo "==> Parallel jobs   : ${JOBS}"

# ---- libmx2 ------------------------------------------------------------------
if [ ! -d libmx2 ]; then
    echo "==> cloning libmx2"
    git clone https://github.com/lostjared/libmx2.git
fi

(
    cd libmx2/libmx
    rm -rf build
    mkdir build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX}" \
        -DCMAKE_INSTALL_RPATH="${RPATH}"
    make -j"${JOBS}"
    sudo make install
)

# ---- acidcam-gpu (contains ACMX2) -------------------------------------------
if [ ! -d acidcam-gpu ]; then
    echo "==> cloning acidcam-gpu"
    git clone https://github.com/lostjared/acidcam-gpu.git
fi

cd acidcam-gpu/ACMX2

# MXWrite ships in-tree; build and install it before acmx2.
(
    cd MXWrite
    rm -rf build
    mkdir build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX}" \
        -DCMAKE_INSTALL_RPATH="${RPATH}"
    make -j"${JOBS}"
    sudo make install
)

# acmx2 engine. CUDA is forced OFF on macOS.
rm -rf build
mkdir build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=OFF \
    -DAUDIO=OFF \
    -DMIDI=OFF \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX}" \
    -DCMAKE_INSTALL_RPATH="${RPATH}"
make -j"${JOBS}"
sudo make install
cd ..

# Qt6 interface. Uses CMakeLists.txt (NOT the legacy interface.pro).
(
    cd interface
    rm -rf build
    mkdir build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX}" \
        -DCMAKE_INSTALL_RPATH="${RPATH}"
    make -j"${JOBS}"
    sudo make install
)

echo
echo "==> build complete"
echo
echo "    acmx2            installed to /usr/local/bin/acmx2"
echo "    acmx2_interface  installed to /usr/local/bin/acmx2_interface"
echo
echo "next: download the macOS-compatible shader pack and point the"
echo "      interface at it (File -> Properties -> Shader Directory), or run"
echo "      from the command line with: acmx2 -s /path/to/shaders"
echo
echo "      macOS shader pack: https://lostsidedead.biz/acmx2/shaders.macos.zip"
