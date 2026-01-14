cmake_minimum_required(VERSION 3.10)
project(ACMX2 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenGL_GL_PREFERENCE GLVND)
include(CheckIncludeFileCXX)
include(CheckLibraryExists)
include(CheckCXXSourceCompiles)
find_package(PkgConfig REQUIRED)
if(NOT PKG_CONFIG_FOUND)
    message(FATAL_ERROR "pkg-config not found. Please install pkg-config.")
endif()
find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(SDL2_mixer REQUIRED)
find_package(libmx2 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio cudaimgproc)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV development libraries with CUDA support.")
endif()
message(STATUS "OpenCV found: version ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
if(NOT OpenCV_CUDA_VERSION)
    message(WARNING "OpenCV CUDA support may not be available. Some features might not work.")
endif()
find_package(MXWrite REQUIRED)
if(NOT MXWrite_FOUND)
    message(FATAL_ERROR "MXWrite not found. Please ensure MXWrite is installed and CMAKE_PREFIX_PATH is set correctly.")
endif()
message(STATUS "MXWrite found: ${MXWrite_INCLUDE_DIRS}")
find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. Please install CUDA Toolkit.")
endif()
message(STATUS "CUDA Toolkit found: version ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
find_package(acidcam-gpu REQUIRED CONFIG)
if(NOT acidcam-gpu_FOUND)
    message(FATAL_ERROR "acidcam-gpu not found. Please ensure acidcam-gpu is installed and CMAKE_PREFIX_PATH is set correctly.")
endif()
message(STATUS "acidcam-gpu found")
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswscale
)
if(NOT FFMPEG_FOUND)
    message(FATAL_ERROR "FFMPEG libraries not found. Please install ffmpeg development libraries (libavcodec, libavformat, libavutil, libswscale).")
endif()
message(STATUS "FFMPEG found: ${FFMPEG_LIBRARIES}")

set(CMAKE_REQUIRED_INCLUDES 
    ${libmx2_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${MXWrite_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    /usr/local/include
    /usr/local/include/opencv4
    /opt/cuda/include
)
check_include_file_cxx("vector" HAVE_VECTOR)
check_include_file_cxx("string" HAVE_STRING)
check_include_file_cxx("fstream" HAVE_FSTREAM)
check_include_file_cxx("filesystem" HAVE_FILESYSTEM)
check_include_file_cxx("thread" HAVE_THREAD)
check_include_file_cxx("mutex" HAVE_MUTEX)
check_include_file_cxx("atomic" HAVE_ATOMIC)
check_include_file_cxx("optional" HAVE_OPTIONAL)
check_include_file_cxx("string_view" HAVE_STRING_VIEW)
check_include_file_cxx("deque" HAVE_DEQUE)
check_include_file_cxx("queue" HAVE_QUEUE)
check_include_file_cxx("condition_variable" HAVE_CONDITION_VARIABLE)
check_include_file_cxx("chrono" HAVE_CHRONO)
check_include_file_cxx("algorithm" HAVE_ALGORITHM)
check_include_file_cxx("unordered_map" HAVE_UNORDERED_MAP)
if(NOT HAVE_FILESYSTEM)
    message(FATAL_ERROR "C++17 <filesystem> header not found. Please ensure your compiler supports C++17.")
endif()
if(NOT HAVE_OPTIONAL OR NOT HAVE_STRING_VIEW)
    message(FATAL_ERROR "C++17 headers (<optional>, <string_view>) not found. Please ensure your compiler supports C++17/C++20.")
endif()
set(CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS} /usr/local/include/opencv4)
check_include_file_cxx("opencv2/opencv.hpp" HAVE_OPENCV_HPP)
check_include_file_cxx("opencv2/core/cuda.hpp" HAVE_OPENCV_CUDA_HPP)
check_include_file_cxx("opencv2/cudaimgproc.hpp" HAVE_OPENCV_CUDAIMGPROC_HPP)
if(NOT HAVE_OPENCV_HPP)
    message(FATAL_ERROR "OpenCV headers not found. Please install OpenCV development libraries.")
endif()
if(NOT HAVE_OPENCV_CUDA_HPP OR NOT HAVE_OPENCV_CUDAIMGPROC_HPP)
    message(WARNING "OpenCV CUDA headers not found. CUDA-accelerated image processing may not be available.")
endif()
check_include_file_cxx("glm/gtc/matrix_transform.hpp" HAVE_GLM)
if(NOT HAVE_GLM)
    message(FATAL_ERROR "GLM headers not found. Please install GLM (OpenGL Mathematics library).")
endif()
option(AUDIO "Audio support" OFF)
if(AUDIO)
    find_library(RTAUDIO_LIBRARY NAMES rtaudio REQUIRED)
    find_path(RTAUDIO_INCLUDE_DIR NAMES rtaudio/RtAudio.h PATHS /usr/include /usr/local/include)
    if(NOT RTAUDIO_LIBRARY)
        message(FATAL_ERROR "RtAudio library not found. Please install rtaudio or disable AUDIO option.")
    endif()
    if(NOT RTAUDIO_INCLUDE_DIR)
        message(FATAL_ERROR "RtAudio headers not found. Please install rtaudio development files or disable AUDIO option.")
    endif()
    set(CMAKE_REQUIRED_INCLUDES ${RTAUDIO_INCLUDE_DIR})
    check_include_file_cxx("rtaudio/RtAudio.h" HAVE_RTAUDIO_H)
    if(NOT HAVE_RTAUDIO_H)
        check_include_file_cxx("RtAudio.h" HAVE_RTAUDIO_H_ALT)
        if(NOT HAVE_RTAUDIO_H_ALT)
            message(FATAL_ERROR "RtAudio.h header not found. Please install rtaudio development files.")
        endif()
    endif()
    set(AUDIO_INCLUDE ${RTAUDIO_INCLUDE_DIR})
    set(AUDIO_LIBS ${RTAUDIO_LIBRARY})
    set(AUDIO_SRC "audio.cpp")
    message(STATUS "AUDIO ENABLED")
    message(STATUS "AUDIO LIBRARY: ${RTAUDIO_LIBRARY}")
    message(STATUS "AUDIO INCLUDE: ${RTAUDIO_INCLUDE_DIR}")
else()
    set(AUDIO_INCLUDE "")
    set(AUDIO_LIBS "")
    set(AUDIO_SRC "")
    message(STATUS "AUDIO DISABLED")
endif()
set(CMAKE_REQUIRED_FLAGS "-std=c++20")
check_cxx_source_compiles("
    #include <concepts>
    #include <ranges>
    int main() { return 0; }
" HAVE_CXX20_FEATURES)
if(NOT HAVE_CXX20_FEATURES)
    message(WARNING "Full C++20 support not detected. Some features may be limited.")
endif()
set(CMAKE_REQUIRED_FLAGS "")
check_cxx_source_compiles("
    #include <thread>
    #include <mutex>
    int main() {
        std::mutex m;
        std::thread t([&m]{ std::lock_guard<std::mutex> lock(m); });
        t.join();
        return 0;
    }
" HAVE_THREADING)
if(NOT HAVE_THREADING)
    message(FATAL_ERROR "Threading support not available. Please ensure your compiler supports std::thread and std::mutex.")
endif()
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
if(NOT OpenGL_FOUND)
    message(FATAL_ERROR "OpenGL not found. Please install OpenGL development libraries.")
endif()
message(STATUS "OpenGL found: ${OPENGL_opengl_LIBRARY}")
message(STATUS "")
message(STATUS "=== ACMX2 Configuration Summary ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Audio Support: ${AUDIO}")
message(STATUS "FFMPEG Libraries: ${FFMPEG_LIBRARIES}")
message(STATUS "FFMPEG Include Dirs: ${FFMPEG_INCLUDE_DIRS}")
message(STATUS "MXWrite: ${MXWrite_INCLUDE_DIRS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===================================")
find_package(Threads REQUIRED)
add_executable(audio_transfer audio_transfer.cpp)
target_compile_options(audio_transfer PRIVATE -O3 -Wall -pedantic)
target_include_directories(audio_transfer PRIVATE 
    ${libmx2_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS} 
    ${SDL2_TTF_INCLUDE_DIRS}
)
target_link_libraries(audio_transfer PRIVATE 
    MXWrite::mxwrite
    PkgConfig::FFMPEG
)
add_executable(acmx2 acmx.cpp ${AUDIO_SRC})
set_target_properties(acmx2 PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
if(AUDIO)
    target_compile_definitions(acmx2 PRIVATE AUDIO_ENABLED)
endif()
target_compile_options(acmx2 PRIVATE -O3 -Wall -pedantic)
target_include_directories(acmx2 PRIVATE 
    ${libmx2_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS} 
    ${SDL2_TTF_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${AUDIO_INCLUDE}/rtaudio
    ${CMAKE_INSTALL_PREFIX}/include
    ${CMAKE_PREFIX_PATH}/include
    ${MXWrite_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    /usr/local/include
    /usr/local/include/opencv4
    /opt/cuda/include
)
target_link_libraries(acmx2 PRIVATE 
    libmx2::mx 
    ${OpenCV_LIBRARIES} 
    ${AUDIO_LIBS}
    MXWrite::mxwrite
    acidcam-gpu::acidcam-gpu
    CUDA::cudart
    OpenGL::OpenGL
    Threads::Threads
)
install(TARGETS acmx2 DESTINATION bin)
install(TARGETS audio_transfer DESTINATION bin)
