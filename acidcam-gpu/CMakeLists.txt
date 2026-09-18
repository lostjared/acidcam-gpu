cmake_minimum_required(VERSION 3.18)

project(acidcam-gpu
    VERSION 1.1.0
    DESCRIPTION "GPU-accelerated video filter library using CUDA"
    LANGUAGES CXX CUDA
)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(CheckIncludeFileCXX)
include(CheckLibraryExists)
include(CheckCXXSourceCompiles)

find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
    message(WARNING "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} detected. Version 11.0+ recommended.")
endif()

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    videoio
    cudaimgproc
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

set(ACIDCAM_GPU_LOCAL_MXWRITE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../MXWrite")
option(EXAMPLES "build examples" OFF)
if(NOT EXISTS "${ACIDCAM_GPU_LOCAL_MXWRITE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "The local MXWrite source tree was not found at ${ACIDCAM_GPU_LOCAL_MXWRITE_DIR}. "
        "acidcam-gpu requires the sibling MXWrite checkout and will not use an installed copy."
    )
endif()

add_subdirectory(
    "${ACIDCAM_GPU_LOCAL_MXWRITE_DIR}"
    "${CMAKE_BINARY_DIR}/acidcam_gpu_mxwrite_build"
    EXCLUDE_FROM_ALL
)

if(NOT TARGET mxwrite)
    message(FATAL_ERROR "The local MXWrite project did not define the expected mxwrite target.")
endif()

message(STATUS "Using local MXWrite source: ${ACIDCAM_GPU_LOCAL_MXWRITE_DIR}")

if(NOT WIN32)
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
        libavcodec
        libavformat
        libavutil
        libswscale
    )

    message(STATUS "FFMPEG libraries: ${FFMPEG_LIBRARIES}")
    message(STATUS "FFMPEG include dirs: ${FFMPEG_INCLUDE_DIRS}")
endif()

set(CMAKE_REQUIRED_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(NOT WIN32)
    list(APPEND CMAKE_REQUIRED_INCLUDES ${FFMPEG_INCLUDE_DIRS})
endif()

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
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "MXWrite: ${ACIDCAM_GPU_LOCAL_MXWRITE_DIR}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")

add_library(acidcam-gpu STATIC
    src/filters.cu
)

set_target_properties(acidcam-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(MSVC)
    target_compile_options(acidcam-gpu PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:/O2>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>
    )
else()
    target_compile_options(acidcam-gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-ffast-math;-fomit-frame-pointer>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>
    )
endif()

target_include_directories(acidcam-gpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

if(NOT WIN32)
    target_include_directories(acidcam-gpu PRIVATE ${FFMPEG_INCLUDE_DIRS})
endif()

target_link_libraries(acidcam-gpu
    PUBLIC
        ${OpenCV_LIBS}
        CUDA::cudart
    PRIVATE
        Threads::Threads
)

if(EXAMPLES)

add_executable(acidcam
    app/main_cv.cu
)

set_target_properties(acidcam PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


if(MSVC)
    target_compile_options(acidcam PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:/O2>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>
    )
else()
    target_compile_options(acidcam PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-ffast-math;-fomit-frame-pointer>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>
    )
endif()

target_include_directories(acidcam
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(acidcam
    PRIVATE
        acidcam-gpu
        mxwrite
        ${OpenCV_LIBS}
        CUDA::cudart
)

endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ac-gpu
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)

if(UNIX AND NOT APPLE)
    set_target_properties(acidcam-gpu acidcam PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
endif()

install(TARGETS acidcam-gpu
    EXPORT acidcam-gpuTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(EXAMPLES)
install(TARGETS acidcam
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
endif()

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

if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE
        @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    )
endif()
