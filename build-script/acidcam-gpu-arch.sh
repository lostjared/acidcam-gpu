#!/bin/bash


# Arch installs CUDA to /opt/cuda
PATH="/opt/cuda/bin:${PATH}"
LD_LIBRARY_PATH="/opt/cuda/lib64:/usr/lib:${LD_LIBRARY_PATH}"
CUDACXX="/opt/cuda/bin/nvcc"

# Limit parallel jobs to avoid OOM during CUDA compilation
BUILDJOBS=4

# ---- Build & install libmx2 ----
mkdir -p /opt/src
cd /opt/src
git clone --depth=1 https://github.com/lostjared/libmx2.git
cd /opt/src/libmx2/libmx
cmake -S . -B build -G Ninja \
      -DEXAMPLES=OFF -DOPENGL=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
    && cmake --build build -j${BUILDJOBS} \
    && cmake --install build \
    && ldconfig

# ---- Clone acidcam-gpu repo ----
cd /opt/src
git clone --depth=1 https://github.com/lostjared/acidcam-gpu.git

# ---- Build & install MXWrite ----
cd /opt/src/acidcam-gpu/MXWrite
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
    && cmake --build build -j${BUILDJOBS} \
    && cmake --install build \
    && ldconfig

# ---- Build & install acidcam-gpu library (CUDA heavy) ----
cd /opt/src/acidcam-gpu/acidcam-gpu
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_CUDA_ARCHITECTURES="75" \
    && cmake --build build -j${BUILDJOBS} \
    && cmake --install build \
    && ldconfig

# ---- Build ACMX2 ----
cd /opt/src/acidcam-gpu/ACMX2
cmake -S . -B build -G Ninja \
      -DAUDIO=ON \
      -DMIDI=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
    && cmake --build build -j${BUILDJOBS} \
    && cmake --install build \
    && ldconfig

# ---- Build ACMX2 Qt interface ----
cd /opt/src/acidcam-gpu/ACMX2/interface
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
    && cmake --build build -j${BUILDJOBS} \
    && cmake --install build \
    && ldconfig

# ---- Build ACMX2 MIDI Map tool ----
cd /opt/src/acidcam-gpu/ACMX2/interface/midi-map
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
    && cmake --build build -j${BUILDJOBS} \
    && cmake --install build \
    && ldconfig


# ---- Download shader packs & models ----
cd /opt/src
mkdir -p files/shaders \
    && cd files/shaders \
    && curl -L https://lostsidedead.biz/acmx2/shaders.zip -o shaders.zip \
    && unzip -o shaders.zip \
    && cd .. \
    && curl -L https://lostsidedead.biz/acmx2/models.zip -o models.zip \
    && unzip -o models.zip


echo "Shaders installed to: /opt/src/files/shaders"
echo "Models installed to: /opt/src/files/models"

# ---- Fix ownership of installed binaries ----
if [ -n "${SUDO_USER}" ]; then
     chown -R "${SUDO_USER}:${SUDO_USER}" /opt/src/acidcam-gpu/ACMX2/interface
fi

echo ""
echo "=== Installation Complete ==="
echo "Binaries installed to: /usr/bin/"
echo "  acmx2, acmx2_interface, midi-map, audio_transfer"
echo "Data directory: /usr/share/acmx2/data/"
echo "Desktop files: /usr/share/applications/"
echo "Shaders: /opt/src/files/shaders"
echo "Models:  /opt/src/files/models"

