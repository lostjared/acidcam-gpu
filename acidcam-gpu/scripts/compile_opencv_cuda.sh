#!/bin/bash
# Optimized for Arch Distrobox
# Fixes: CUDA 13.x Hardware Structs, FFmpeg 7/8 API, and Qt6/CMake Syntax Error
set -e

OPENCV_VERSION="4.12.0"
BUILD_WORKSPACE="$HOME/opencv_build"
INSTALL_PREFIX="/usr/local"

echo "--- STEP 1: Updating Dependencies ---"
sudo pacman -Syu --noconfirm
# Added gtk3 and libcanberra for the new UI backend
sudo pacman -S --needed --noconfirm base-devel cmake git cuda cudnn \
    libjpeg-turbo libpng libtiff v4l-utils tbb openblas ffmpeg gstreamer \
    gtk3 libcanberra

echo "--- STEP 2: Workspace Setup & Patching ---"
mkdir -p "$BUILD_WORKSPACE" && cd "$BUILD_WORKSPACE"
[ ! -d "opencv" ] && git clone https://github.com/opencv/opencv.git
[ ! -d "opencv_contrib" ] && git clone https://github.com/opencv/opencv_contrib.git

cd "$BUILD_WORKSPACE/opencv" && git checkout "$OPENCV_VERSION"
cd "$BUILD_WORKSPACE/opencv_contrib" && git checkout "$OPENCV_VERSION"

# --- Patching CUDA 13.x compatibility ---
CORE_CUDA="$BUILD_WORKSPACE/opencv/modules/core/src/cuda_info.cpp"
sed -i 's/deviceProps().get(device_id_)->clockRate/0/g' "$CORE_CUDA"
sed -i 's/deviceProps().get(device_id_)->kernelExecTimeoutEnabled/0/g' "$CORE_CUDA"
sed -i 's/deviceProps().get(device_id_)->computeMode/0/g' "$CORE_CUDA"
sed -i 's/deviceProps().get(device_id_)->maxTexture1DLinear/0/g' "$CORE_CUDA"
sed -i 's/deviceProps().get(device_id_)->memoryClockRate/0/g' "$CORE_CUDA"
sed -i 's/prop.clockRate/0/g' "$CORE_CUDA"
sed -i 's/prop.deviceOverlap/0/g' "$CORE_CUDA"
sed -i 's/prop.kernelExecTimeoutEnabled/0/g' "$CORE_CUDA"
sed -i 's/prop.computeMode/0/g' "$CORE_CUDA"

# --- Patching FFmpeg 7/8 compatibility ---
FFMPEG_FILE="$BUILD_WORKSPACE/opencv/modules/videoio/src/cap_ffmpeg_impl.hpp"
sed -i 's/avcodec_close( \(.*\) );/avcodec_free_context( \&\1 );/g' "$FFMPEG_FILE"
sed -i '/data = av_stream_get_side_data(video_st, AV_PKT_DATA_DISPLAYMATRIX, NULL);/c\data = NULL;' "$FFMPEG_FILE"

echo "--- STEP 3: Configuring CMake (GTK Backend) ---"
# NOTE: Removed 'rm -rf build' to allow resuming from 66%
mkdir -p "$BUILD_WORKSPACE/build"
cd "$BUILD_WORKSPACE/build"

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D WITH_TBB=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D CUDA_ARCH_BIN=7.5 \
      -D OPENCV_EXTRA_MODULES_PATH="$BUILD_WORKSPACE/opencv_contrib/modules" \
      -D WITH_GTK=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D WITH_MPI=OFF \
      -D BUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,features2d,flann,ml,objdetect,photo,video,dnn,cudaarithm,cudabgsegm,cudafeatures2d,cudafilters,cudaimgproc,cudalegacy,cudaobjdetect,cudaoptflow,cudastereo,cudawarping,cudev \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-error=narrowing" \
      -D BUILD_opencv_python3=OFF \
      -D BUILD_opencv_java=OFF \
      -D BUILD_opencv_js=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      ../opencv

echo "--- STEP 4: Resuming Compilation ---"
make -j$(nproc)

echo "--- STEP 5: Installation ---"
sudo make install
sudo ldconfig

echo "✅ OpenCV build finalized! You can now re-link acidcam-gpu."
