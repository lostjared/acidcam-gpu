cmake_minimum_required(VERSION 3.18)
project(acidcam-gpu 
    VERSION 1.0.0
    DESCRIPTION "GPU-accelerated video filter library using CUDA"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_INCLUDEDIR "include")
set(CMAKE_INSTALL_LIBDIR "lib")
set(CMAKE_INSTALL_BINDIR "bin")

include(CheckIncludeFileCXX)
include(CheckLibraryExists)
include(CheckCXXSourceCompiles)
include(GNUInstallDirs)

find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. This project requires CUDA.")
endif()
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
    message(WARNING "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} detected. Version 11.0+ recommended.")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio cudaimgproc)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV with CUDA support.")
endif()
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

if(NOT OpenCV_CUDA_VERSION)
    message(WARNING "OpenCV CUDA support may not be available.")
endif()

find_package(PkgConfig REQUIRED)
if(NOT PKG_CONFIG_FOUND)
    message(FATAL_ERROR "pkg-config not found. Please install pkg-config.")
endif()

find_package(MXWrite REQUIRED CONFIG)
if(NOT MXWrite_FOUND)
    message(FATAL_ERROR "MXWrite not found. Please ensure MXWrite is installed and CMAKE_PREFIX_PATH is set.")
endif()
message(STATUS "MXWrite found: ${MXWrite_DIR}")

pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswscale
)
if(NOT FFMPEG_FOUND)
    message(FATAL_ERROR "FFMPEG libraries not found. Please install ffmpeg development libraries.")
endif()
message(STATUS "FFMPEG libraries: ${FFMPEG_LIBRARIES}")
message(STATUS "FFMPEG include dirs: ${FFMPEG_INCLUDE_DIRS}")

set(CMAKE_REQUIRED_INCLUDES 
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${MXWrite_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)

check_include_file_cxx("opencv2/opencv.hpp" HAVE_OPENCV_HPP)
check_include_file_cxx("opencv2/core/cuda.hpp" HAVE_OPENCV_CUDA_HPP)
check_include_file_cxx("opencv2/cudaimgproc.hpp" HAVE_OPENCV_CUDAIMGPROC_HPP)

if(NOT HAVE_OPENCV_HPP)
    message(FATAL_ERROR "OpenCV main header not found.")
endif()

if(NOT HAVE_OPENCV_CUDA_HPP OR NOT HAVE_OPENCV_CUDAIMGPROC_HPP)
    message(FATAL_ERROR "OpenCV CUDA headers not found. Please install OpenCV with CUDA support.")
endif()

check_include_file_cxx("string" HAVE_STRING)
check_include_file_cxx("vector" HAVE_VECTOR)
check_include_file_cxx("format" HAVE_FORMAT)
check_include_file_cxx("filesystem" HAVE_FILESYSTEM)
check_include_file_cxx("regex" HAVE_REGEX)

if(NOT HAVE_FORMAT)
    message(WARNING "C++20 <format> not available. Some features may not work.")
endif()

if(NOT HAVE_FILESYSTEM)
    message(FATAL_ERROR "C++17 <filesystem> not found. Please use a compiler with C++17 support.")
endif()

find_package(Threads REQUIRED)

message(STATUS "")
message(STATUS "=== acidcam-gpu Configuration Summary ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "FFMPEG: ${FFMPEG_LIBRARIES}")
message(STATUS "MXWrite: ${MXWrite_DIR}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
message(STATUS "========================================")

add_library(acidcam-gpu SHARED
    src/filters.cu
)

set_target_properties(acidcam-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_compile_options(acidcam-gpu PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -ffast-math -fomit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

target_include_directories(acidcam-gpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
        ${MXWrite_INCLUDE_DIRS}
        ${FFMPEG_INCLUDE_DIRS}
)

target_link_libraries(acidcam-gpu
    PUBLIC
        ${OpenCV_LIBS}
        CUDA::cudart
        MXWrite::mxwrite
    PRIVATE
        Threads::Threads
)

add_executable(acidcam
    app/main_cv.cu
)

set_target_properties(acidcam PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(acidcam PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -ffast-math -fomit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

target_include_directories(acidcam
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(acidcam
    PRIVATE
        acidcam-gpu
        ${OpenCV_LIBS}
        CUDA::cudart
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ac-gpu
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

set_target_properties(acidcam-gpu acidcam PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    BUILD_WITH_INSTALL_RPATH FALSE
)

install(TARGETS acidcam-gpu
    EXPORT acidcam-gpuTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS acidcam
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT acidcam-gpuTargets
    FILE acidcam-gpuTargets.cmake
    NAMESPACE acidcam-gpu::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/acidcam-gpu
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/acidcam-gpuConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/acidcam-gpuConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/acidcam-gpuConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/acidcam-gpu
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/acidcam-gpuConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/acidcam-gpuConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/acidcam-gpu
)