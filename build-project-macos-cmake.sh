#!/usr/bin/env bash
# Build the macOS/MoltenVK configuration in a self-contained local prefix.
#
# Optional environment overrides:
#   ACMX_MACOS_BUILD_DIRECTORY=/path/to/build
#   ACMX_MACOS_INSTALL_PREFIX=/path/to/prefix
#   MXVK_SOURCE_DIR=/path/to/MXVK
#   LIBMX2_SOURCE_DIR=/path/to/libmx2

set -Eeuo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: build-project-macos-cmake.sh must be run on macOS" >&2
    exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
    echo "error: Homebrew is required. Install it from https://brew.sh first." >&2
    exit 1
fi

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIRECTORY="$SCRIPT_DIRECTORY"
BUILD_DIRECTORY="${ACMX_MACOS_BUILD_DIRECTORY:-$PROJECT_DIRECTORY/build/macos}"
INSTALL_PREFIX="${ACMX_MACOS_INSTALL_PREFIX:-$PROJECT_DIRECTORY/build/macos-prefix}"
MXVK_DIRECTORY="${MXVK_SOURCE_DIR:-$PROJECT_DIRECTORY/../MXVK}"
LIBMX2_DIRECTORY="${LIBMX2_SOURCE_DIR:-$PROJECT_DIRECTORY/../libmx2}"
MXVK_REPOSITORY="https://github.com/lostjared/MXVK.git"
LIBMX2_REPOSITORY="https://github.com/lostjared/libmx2.git"
CPU_COUNT="$(sysctl -n hw.ncpu)"
ARCHITECTURE="$(uname -m)"

case "$ARCHITECTURE" in
    arm64|x86_64) ;;
    *)
        echo "error: unsupported macOS architecture: $ARCHITECTURE" >&2
        exit 1
        ;;
esac

# These cover ACMX2, ACMXVK/MXVK, the Qt interface, optional snapshots,
# audio/MIDI, shader compilation, and the MoltenVK Vulkan implementation.
BREW_PACKAGES=(
    cmake ninja pkgconf git curl unzip
    ffmpeg opencv
    sdl2 sdl2_ttf sdl2_mixer sdl2_image glew
    sdl3 sdl3_ttf sdl3_mixer
    qt glm eigen fmt
    rtaudio rtmidi
    jpeg-turbo libpng libtiff webp yaml-cpp zlib freetype fontconfig
    vulkan-headers vulkan-loader molten-vk shaderc glslang
)

echo "Installing required Homebrew packages..."
brew install "${BREW_PACKAGES[@]}"

for REQUIRED_COMMAND in cmake git; do
    if ! command -v "$REQUIRED_COMMAND" >/dev/null 2>&1; then
        echo "error: Homebrew did not provide required command: $REQUIRED_COMMAND" >&2
        exit 1
    fi
done

if [[ ! -d "$LIBMX2_DIRECTORY/.git" ]]; then
    if [[ -e "$LIBMX2_DIRECTORY" ]]; then
        echo "error: LIBMX2_SOURCE_DIR exists but is not a Git checkout: $LIBMX2_DIRECTORY" >&2
        exit 1
    fi
    echo "Cloning libmx2 from $LIBMX2_REPOSITORY..."
    git clone "$LIBMX2_REPOSITORY" "$LIBMX2_DIRECTORY"
else
    echo "Using existing libmx2 checkout: $LIBMX2_DIRECTORY"
fi

if [[ ! -f "$LIBMX2_DIRECTORY/libmx/CMakeLists.txt" ]]; then
    echo "error: libmx2 checkout does not contain libmx/CMakeLists.txt: $LIBMX2_DIRECTORY" >&2
    exit 1
fi

if [[ ! -d "$MXVK_DIRECTORY/.git" ]]; then
    if [[ -e "$MXVK_DIRECTORY" ]]; then
        echo "error: MXVK_SOURCE_DIR exists but is not a Git checkout: $MXVK_DIRECTORY" >&2
        exit 1
    fi
    echo "Cloning MXVK from $MXVK_REPOSITORY..."
    git clone "$MXVK_REPOSITORY" "$MXVK_DIRECTORY"
else
    echo "Using existing MXVK checkout: $MXVK_DIRECTORY"
fi

if [[ ! -f "$MXVK_DIRECTORY/CMakeLists.txt" ]]; then
    echo "error: MXVK checkout does not contain CMakeLists.txt: $MXVK_DIRECTORY" >&2
    exit 1
fi

BREW_PREFIX="$(brew --prefix)"
VULKAN_LOADER_PREFIX="$(brew --prefix vulkan-loader)"
MOLTENVK_PREFIX="$(brew --prefix molten-vk)"
SHADERC_PREFIX="$(brew --prefix shaderc)"
CMAKE_PREFIX_PATH="$INSTALL_PREFIX;$BREW_PREFIX;$VULKAN_LOADER_PREFIX;$MOLTENVK_PREFIX;$SHADERC_PREFIX"

# Respect an already sourced Vulkan SDK without depending on a machine-local
# shell script such as ~/vulkan.sh.
if [[ -n "${VULKAN_SDK:-}" ]]; then
    CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;$VULKAN_SDK"
fi
COMMON_CMAKE_OPTIONS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    -DCMAKE_OSX_ARCHITECTURES="$ARCHITECTURE"
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
)

mkdir -p "$BUILD_DIRECTORY" "$INSTALL_PREFIX"

echo "Configuring libmx2..."
cmake -S "$LIBMX2_DIRECTORY/libmx" -B "$BUILD_DIRECTORY/libmx2" \
    "${COMMON_CMAKE_OPTIONS[@]}" \
    -DEXAMPLES=OFF
cmake --build "$BUILD_DIRECTORY/libmx2" --parallel "$CPU_COUNT"
cmake --install "$BUILD_DIRECTORY/libmx2"

echo "Configuring MXVK..."
cmake -S "$MXVK_DIRECTORY" -B "$BUILD_DIRECTORY/mxvk" \
    "${COMMON_CMAKE_OPTIONS[@]}" \
    -DEXAMPLES=OFF \
    -DWITH_CUDA=OFF \
    -DCV=ON \
    -DVALIDATION=OFF
cmake --build "$BUILD_DIRECTORY/mxvk" --parallel "$CPU_COUNT"
cmake --install "$BUILD_DIRECTORY/mxvk"

echo "Configuring ACMX2..."
cmake -S "$PROJECT_DIRECTORY/ACMX2" -B "$BUILD_DIRECTORY/acmx2" \
    "${COMMON_CMAKE_OPTIONS[@]}" \
    -DAUDIO=ON \
    -DMIDI=ON \
    -DWEBP=ON \
    -DTIFF=ON \
    -DWITH_CUDA=OFF \
    -DWITH_OPENCV_DNN=ON
cmake --build "$BUILD_DIRECTORY/acmx2" --parallel "$CPU_COUNT"
cmake --install "$BUILD_DIRECTORY/acmx2"

echo "Configuring ACMXVK with MoltenVK..."
cmake -S "$PROJECT_DIRECTORY/ACMXVK" -B "$BUILD_DIRECTORY/acmxvk" \
    "${COMMON_CMAKE_OPTIONS[@]}" \
    -DACMXVK_USE_MOLTENVK=ON \
    -DAUDIO=ON \
    -DMIDI=ON \
    -DWEBP=ON \
    -DTIFF=ON \
    -DWITH_CUDA=OFF \
    -DWITH_OPENCV_DNN=ON
cmake --build "$BUILD_DIRECTORY/acmxvk" --parallel "$CPU_COUNT"
cmake --install "$BUILD_DIRECTORY/acmxvk"

echo "Configuring the Qt interface..."
cmake -S "$PROJECT_DIRECTORY/ACMX2/interface" -B "$BUILD_DIRECTORY/interface" \
    "${COMMON_CMAKE_OPTIONS[@]}"
cmake --build "$BUILD_DIRECTORY/interface" --parallel "$CPU_COUNT"
cmake --install "$BUILD_DIRECTORY/interface"

cat <<EOF

macOS build complete.
  Build directory: $BUILD_DIRECTORY
  Install prefix:  $INSTALL_PREFIX
  libmx2 source:   $LIBMX2_DIRECTORY
  MXVK source:     $MXVK_DIRECTORY

Add the local prefix to PATH before launching installed executables:
  export PATH="$INSTALL_PREFIX/bin:\$PATH"

Persist that PATH setting for future zsh sessions:
  echo 'export PATH="$INSTALL_PREFIX/bin:\$PATH"' >> ~/.zshenv

The script builds libmx2 into the local install prefix before configuring ACMX2.
EOF
