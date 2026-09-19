cmake_minimum_required(VERSION 3.20)

project(ACMX2 VERSION 2.101.1 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenGL_GL_PREFERENCE GLVND)

include(GNUInstallDirs)
include(CheckIncludeFileCXX)
include(CheckLibraryExists)
include(CheckCXXSourceCompiles)

set(ACMX2_USE_PKG_CONFIG_DEFAULT ON)

if(MSVC)
    set(ACMX2_USE_PKG_CONFIG_DEFAULT OFF)
endif()

option(
    ACMX2_USE_PKG_CONFIG
    "Use pkg-config for FFmpeg and optional libraries"
    ${ACMX2_USE_PKG_CONFIG_DEFAULT}
)

if(ACMX2_USE_PKG_CONFIG)
    find_package(PkgConfig REQUIRED)
endif()

set(ACMX2_CUDA_DEFAULT ON)

if(APPLE OR WIN32)
    set(ACMX2_CUDA_DEFAULT OFF)
endif()

option(WITH_CUDA "Enable CUDA / acidcam-gpu GPU acceleration" ${ACMX2_CUDA_DEFAULT})
option(WITH_OPENCV_DNN "Enable OpenCV DNN/CUDA" OFF)
option(WEBP "Save HDR snapshots as WebP instead of 16-bit PNG" OFF)
option(TIFF "Enable saving HDR snapshots as 16-bit TIFF (key '4')" OFF)
option(AUDIO "Audio support" OFF)
option(MIDI "MIDI Support" OFF)

# acidcam-gpu is a CUDA separable-compilation static library. Consumers must
# enable CUDA and perform the final device-link step, otherwise MSVC leaves
# __cudaRegisterLinkedBinary_* unresolved at the executable link.
if(WITH_CUDA AND NOT APPLE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(MSVC)
    find_package(SDL2 CONFIG REQUIRED)
    find_package(SDL2_image CONFIG REQUIRED)
    find_package(SDL2_ttf CONFIG REQUIRED)
    find_package(SDL2_mixer CONFIG REQUIRED)
else()
    find_package(SDL2 REQUIRED)
    find_package(SDL2_image REQUIRED)
    find_package(SDL2_ttf REQUIRED)
    find_package(SDL2_mixer REQUIRED)
endif()

if(NOT TARGET SDL2::SDL2)
    if(TARGET SDL2::SDL2-static)
        add_library(SDL2::SDL2 ALIAS SDL2::SDL2-static)
    else()
        message(FATAL_ERROR "SDL2 was found, but no SDL2::SDL2 target is available.")
    endif()
endif()

if(NOT TARGET SDL2_image::SDL2_image)
    if(TARGET SDL2_image::SDL2_image-static)
        add_library(SDL2_image::SDL2_image ALIAS SDL2_image::SDL2_image-static)
    else()
        message(FATAL_ERROR
            "SDL2_image was found, but no SDL2_image::SDL2_image target is available."
        )
    endif()
endif()

if(NOT TARGET SDL2_ttf::SDL2_ttf)
    if(TARGET SDL2_ttf::SDL2_ttf-static)
        add_library(SDL2_ttf::SDL2_ttf ALIAS SDL2_ttf::SDL2_ttf-static)
    else()
        message(FATAL_ERROR
            "SDL2_ttf was found, but no SDL2_ttf::SDL2_ttf target is available."
        )
    endif()
endif()

if(NOT TARGET SDL2_mixer::SDL2_mixer)
    if(TARGET SDL2_mixer::SDL2_mixer-static)
        add_library(SDL2_mixer::SDL2_mixer ALIAS SDL2_mixer::SDL2_mixer-static)
    else()
        message(FATAL_ERROR
            "SDL2_mixer was found, but no SDL2_mixer::SDL2_mixer target is available."
        )
    endif()
endif()

find_package(libmx2 CONFIG REQUIRED)

if(NOT TARGET libmx2::mx)
    message(FATAL_ERROR
        "libmx2 was found, but it did not export the expected libmx2::mx target."
    )
endif()

if(NOT TARGET libmx2::mxgl)
    message(FATAL_ERROR
        "libmx2 was found, but it did not export the expected libmx2::mxgl target."
    )
endif()

if(WITH_CUDA)
    if(WITH_OPENCV_DNN)
        find_package(
            OpenCV REQUIRED
            COMPONENTS
                core
                imgproc
                highgui
                videoio
                cudaimgproc
                dnn
        )
    else()
        find_package(
            OpenCV REQUIRED
            COMPONENTS
                core
                imgproc
                highgui
                videoio
                cudaimgproc
        )
    endif()
else()
    if(WITH_OPENCV_DNN)
        find_package(
            OpenCV REQUIRED
            COMPONENTS
                core
                imgproc
                highgui
                videoio
                dnn
        )
    else()
        find_package(
            OpenCV REQUIRED
            COMPONENTS
                core
                imgproc
                highgui
                videoio
        )
    endif()
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
        "OpenCV not found. Please install OpenCV development libraries."
    )
endif()

message(STATUS "OpenCV found: version ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")

if(WITH_CUDA AND NOT OpenCV_CUDA_VERSION)
    message(WARNING
        "OpenCV CUDA support may not be available. Some features might not work."
    )
endif()

set(
    ACMX2_LOCAL_MXWRITE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../MXWrite"
)

if(NOT EXISTS "${ACMX2_LOCAL_MXWRITE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "The local MXWrite source tree was not found at "
        "${ACMX2_LOCAL_MXWRITE_DIR}. "
        "ACMX2 requires the sibling MXWrite checkout and will not use an "
        "installed copy."
    )
endif()

add_subdirectory(
    "${ACMX2_LOCAL_MXWRITE_DIR}"
    "${CMAKE_BINARY_DIR}/acmx2_mxwrite_build"
    EXCLUDE_FROM_ALL
)

if(NOT TARGET mxwrite)
    message(FATAL_ERROR
        "The local MXWrite project did not define the expected mxwrite target."
    )
endif()

set(ACMX2_MXWRITE_TARGET mxwrite)
set(MXWrite_INCLUDE_DIRS "${ACMX2_LOCAL_MXWRITE_DIR}")

message(STATUS "Using local MXWrite source: ${ACMX2_LOCAL_MXWRITE_DIR}")

if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)

    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR
            "CUDA Toolkit not found. Please install CUDA Toolkit or configure "
            "with -DWITH_CUDA=OFF."
        )
    endif()

    message(STATUS "CUDA Toolkit found: version ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")

    find_package(acidcam-gpu REQUIRED CONFIG)

    if(NOT acidcam-gpu_FOUND)
        message(FATAL_ERROR
            "acidcam-gpu not found. Please ensure acidcam-gpu is installed and "
            "CMAKE_PREFIX_PATH is set correctly, or configure with "
            "-DWITH_CUDA=OFF."
        )
    endif()

    message(STATUS "acidcam-gpu found")
else()
    message(STATUS "CUDA / acidcam-gpu support DISABLED (WITH_CUDA=OFF)")
    set(CUDAToolkit_INCLUDE_DIRS "")
endif()

if(ACMX2_USE_PKG_CONFIG)
    pkg_check_modules(
        FFMPEG
        REQUIRED
        IMPORTED_TARGET
        libavcodec
        libavformat
        libavutil
        libswscale
        libswresample
    )

    set(ACMX2_FFMPEG_TARGET PkgConfig::FFMPEG)

    # Some Windows pkg-config files omit libswresample from the imported
    # target even though ACMX2 uses its API directly.
    find_library(
        ACMX2_SWRESAMPLE_LIBRARY
        NAMES swresample
        REQUIRED
    )
    target_link_libraries(
        ${ACMX2_FFMPEG_TARGET}
        INTERFACE
            "${ACMX2_SWRESAMPLE_LIBRARY}"
    )
else()
    find_path(
        FFMPEG_INCLUDE_DIR
        NAMES libavcodec/avcodec.h
        REQUIRED
    )

    set(ACMX2_FFMPEG_COMPONENTS
        avcodec
        avformat
        avutil
        swscale
        swresample
    )
    set(ACMX2_FFMPEG_LIBRARIES)
    foreach(ACMX2_FFMPEG_COMPONENT IN LISTS ACMX2_FFMPEG_COMPONENTS)
    find_library(ACMX2_FFMPEG_${ACMX2_FFMPEG_COMPONENT}_LIBRARY
        NAMES ${ACMX2_FFMPEG_COMPONENT}
        REQUIRED
    )
   set(ACMX2_FFMPEG_LIBRARY_VAR "ACMX2_FFMPEG_${ACMX2_FFMPEG_COMPONENT}_LIBRARY")
   list(APPEND ACMX2_FFMPEG_LIBRARIES "${${ACMX2_FFMPEG_LIBRARY_VAR}}")
   endforeach()

    set(FFMPEG_FOUND TRUE)
    set(FFMPEG_INCLUDE_DIRS "${FFMPEG_INCLUDE_DIR}")
    set(FFMPEG_LIBRARIES "${ACMX2_FFMPEG_LIBRARIES}")

    add_library(acmx2_ffmpeg INTERFACE)

    target_include_directories(
        acmx2_ffmpeg
        INTERFACE
            "${FFMPEG_INCLUDE_DIR}"
    )

    target_link_libraries(
        acmx2_ffmpeg
        INTERFACE
            ${ACMX2_FFMPEG_LIBRARIES}
    )

    set(ACMX2_FFMPEG_TARGET acmx2_ffmpeg)
endif()

if(NOT FFMPEG_FOUND)
    message(FATAL_ERROR
        "FFmpeg libraries not found. Install FFmpeg development files or set "
        "ACMX2_USE_PKG_CONFIG=ON."
    )
endif()

message(STATUS "FFmpeg found: ${FFMPEG_LIBRARIES}")

set(
    CMAKE_REQUIRED_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${MXWrite_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)

if(WITH_CUDA AND NOT WIN32)
    list(APPEND CMAKE_REQUIRED_INCLUDES /opt/cuda/include)
endif()

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
    message(FATAL_ERROR
        "C++17 <filesystem> header not found. Please ensure your compiler "
        "supports C++17."
    )
endif()

if(NOT HAVE_OPTIONAL OR NOT HAVE_STRING_VIEW)
    message(FATAL_ERROR
        "C++17 headers (<optional>, <string_view>) not found. Please ensure "
        "your compiler supports C++17/C++20."
    )
endif()

set(CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS})

check_include_file_cxx("opencv2/opencv.hpp" HAVE_OPENCV_HPP)

if(WITH_CUDA)
    check_include_file_cxx(
        "opencv2/core/cuda.hpp"
        HAVE_OPENCV_CUDA_HPP
    )

    check_include_file_cxx(
        "opencv2/cudaimgproc.hpp"
        HAVE_OPENCV_CUDAIMGPROC_HPP
    )
endif()

if(NOT HAVE_OPENCV_HPP)
    message(FATAL_ERROR
        "OpenCV headers not found. Please install OpenCV development libraries."
    )
endif()

if(
    WITH_CUDA
    AND
    (
        NOT HAVE_OPENCV_CUDA_HPP
        OR
        NOT HAVE_OPENCV_CUDAIMGPROC_HPP
    )
)
    message(WARNING
        "OpenCV CUDA headers not found. CUDA-accelerated image processing may "
        "not be available."
    )
endif()

if(AUDIO)
    find_library(
        RTAUDIO_LIBRARY
        NAMES rtaudio
        REQUIRED
    )

    find_path(
        RTAUDIO_INCLUDE_DIR
        NAMES RtAudio.h
        PATH_SUFFIXES rtaudio
    )

    if(NOT RTAUDIO_LIBRARY)
        message(FATAL_ERROR
            "RtAudio library not found. Please install rtaudio or disable "
            "AUDIO option."
        )
    endif()

    if(NOT RTAUDIO_INCLUDE_DIR)
        message(FATAL_ERROR
            "RtAudio headers not found. Please install rtaudio development "
            "files or disable AUDIO option."
        )
    endif()

    set(CMAKE_REQUIRED_INCLUDES ${RTAUDIO_INCLUDE_DIR})

    check_include_file_cxx("rtaudio/RtAudio.h" HAVE_RTAUDIO_H)

    if(NOT HAVE_RTAUDIO_H)
        check_include_file_cxx("RtAudio.h" HAVE_RTAUDIO_H_ALT)

        if(NOT HAVE_RTAUDIO_H_ALT)
            message(FATAL_ERROR
                "RtAudio.h header not found. Please install rtaudio "
                "development files."
            )
        endif()
    endif()

    set(AUDIO_INCLUDE ${RTAUDIO_INCLUDE_DIR})
    set(AUDIO_LIBS ${RTAUDIO_LIBRARY})
    set(AUDIO_SRC "audio.cpp" "file_audio.cpp")

    message(STATUS "AUDIO ENABLED")
    message(STATUS "AUDIO LIBRARY: ${RTAUDIO_LIBRARY}")
    message(STATUS "AUDIO INCLUDE: ${RTAUDIO_INCLUDE_DIR}")
else()
    set(AUDIO_INCLUDE "")
    set(AUDIO_LIBS "")
    set(AUDIO_SRC "")

    message(STATUS "AUDIO DISABLED")
endif()

if(MIDI)
    if(MSVC AND VCPKG_TARGET_TRIPLET)
        # vcpkg prepends the debug library directory while configuring a
        # Release build.  Prefer the matching release archive explicitly.
        find_library(
            RTMIDI_LIBRARY
            NAMES rtmidi
            PATHS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib"
            NO_DEFAULT_PATH
            REQUIRED
        )
    else()
        find_library(RTMIDI_LIBRARY NAMES rtmidi REQUIRED)
    endif()

    find_path(
        RTMIDI_INCLUDE_DIR
        NAMES rtmidi/RtMidi.h
        PATHS
            /usr/include
            /usr/local/include
    )

    if(NOT RTMIDI_LIBRARY)
        message(FATAL_ERROR
            "RtMidi library not found. Please install rtmidi or disable "
            "MIDI option."
        )
    endif()

    if(NOT RTMIDI_INCLUDE_DIR)
        message(FATAL_ERROR
            "RtMidi headers not found. Please install rtmidi development "
            "files or disable MIDI option."
        )
    endif()

    set(MIDI_INCLUDE ${RTMIDI_INCLUDE_DIR})
    set(MIDI_LIBS ${RTMIDI_LIBRARY})

    message(STATUS "MIDI ENABLED")
    message(STATUS "MIDI LIBRARY: ${RTMIDI_LIBRARY}")
    message(STATUS "MIDI INCLUDE: ${RTMIDI_INCLUDE_DIR}")
else()
    set(MIDI_INCLUDE "")
    set(MIDI_LIBS "")

    message(STATUS "MIDI DISABLED")
endif()

if(NOT MSVC)
    set(CMAKE_REQUIRED_FLAGS "-std=c++20")
endif()

check_cxx_source_compiles(
    "
    #include <concepts>
    #include <ranges>

    int main() {
        return 0;
    }
    "
    HAVE_CXX20_FEATURES
)

if(NOT HAVE_CXX20_FEATURES)
    message(WARNING
        "Full C++20 support not detected. Some features may be limited."
    )
endif()

set(CMAKE_REQUIRED_FLAGS "")

check_cxx_source_compiles(
    "
    #include <thread>
    #include <mutex>

    int main() {
        std::mutex m;

        std::thread t([&m] {
            std::lock_guard<std::mutex> lock(m);
        });

        t.join();

        return 0;
    }
    "
    HAVE_THREADING
)

if(NOT HAVE_THREADING)
    message(FATAL_ERROR
        "Threading support not available. Please ensure your compiler supports "
        "std::thread and std::mutex."
    )
endif()

find_package(OpenGL REQUIRED)

if(NOT OpenGL_FOUND)
    message(FATAL_ERROR
        "OpenGL not found. Please install OpenGL development libraries."
    )
endif()

if(TARGET OpenGL::GL)
    set(ACMX2_OPENGL_TARGET OpenGL::GL)
elseif(TARGET OpenGL::OpenGL)
    set(ACMX2_OPENGL_TARGET OpenGL::OpenGL)
else()
    message(FATAL_ERROR
        "No usable OpenGL imported target found "
        "(expected OpenGL::GL or OpenGL::OpenGL)."
    )
endif()

message(STATUS "OpenGL target selected: ${ACMX2_OPENGL_TARGET}")

message(STATUS "")
message(STATUS "=== ACMX2 Configuration Summary ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

if(WITH_CUDA)
    message(STATUS
        "CUDA Support: ENABLED (version ${CUDAToolkit_VERSION})"
    )
else()
    message(STATUS "CUDA Support: DISABLED")
endif()

message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Audio Support: ${AUDIO}")
message(STATUS "MIDI Support: ${MIDI}")
message(STATUS "FFMPEG Libraries: ${FFMPEG_LIBRARIES}")
message(STATUS "FFMPEG Include Dirs: ${FFMPEG_INCLUDE_DIRS}")
message(STATUS "MXWrite: ${MXWrite_INCLUDE_DIRS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===================================")

find_package(Threads REQUIRED)

add_executable(
    audio_transfer
    audio_transfer.cpp
)

if(MSVC)
    target_compile_options(
        audio_transfer
        PRIVATE
            /O2
            /W4
            /permissive-
    )
else()
    target_compile_options(
        audio_transfer
        PRIVATE
            -O3
            -Wall
            -pedantic
    )
endif()

target_include_directories(
    audio_transfer
    PRIVATE
        ${MXWrite_INCLUDE_DIRS}
        ${FFMPEG_INCLUDE_DIRS}
)

target_link_libraries(
    audio_transfer
    PRIVATE
        ${ACMX2_MXWRITE_TARGET}
        ${ACMX2_FFMPEG_TARGET}
)

if(WITH_OPENCV_DNN)
    set(ACMX2_DNN_SRC "dnn.cpp")

    if(ACMX2_USE_PKG_CONFIG)
        pkg_check_modules(
            YAMLCPP
            REQUIRED
            IMPORTED_TARGET
            yaml-cpp
        )

        set(ACMX2_YAMLCPP_TARGET PkgConfig::YAMLCPP)
    else()
        find_package(yaml-cpp CONFIG REQUIRED)

        if(TARGET yaml-cpp::yaml-cpp)
            set(ACMX2_YAMLCPP_TARGET yaml-cpp::yaml-cpp)
        elseif(TARGET yaml-cpp)
            set(ACMX2_YAMLCPP_TARGET yaml-cpp)
        else()
            message(FATAL_ERROR
                "yaml-cpp was found, but no usable CMake target was provided."
            )
        endif()
    endif()
else()
    set(ACMX2_DNN_SRC "")
endif()

add_executable(
    acmx2
    acmx.cpp
    program.cpp
    ${AUDIO_SRC}
    ${ACMX2_DNN_SRC}
)

set_target_properties(
    acmx2
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(UNIX)
    set_target_properties(
        acmx2
        PROPERTIES
            INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

target_compile_definitions(
    acmx2
    PRIVATE
        WITH_GL
)

if(WIN32)
    target_compile_definitions(
        acmx2
        PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
    )

    target_sources(
        acmx2
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/win-icon.rc"
    )
endif()

if(WITH_OPENCV_DNN)
    target_compile_definitions(
        acmx2
        PRIVATE
            ACMX2_WITH_DNN
    )

    target_link_libraries(
        acmx2
        PRIVATE
            ${ACMX2_YAMLCPP_TARGET}
    )
endif()

if(WITH_CUDA)
    target_compile_definitions(
        acmx2
        PRIVATE
            ACMX2_WITH_CUDA
    )
endif()

if(WEBP)
    if(ACMX2_USE_PKG_CONFIG)
        pkg_check_modules(
            WEBP
            REQUIRED
            IMPORTED_TARGET
            libwebp
        )

        set(ACMX2_WEBP_TARGET PkgConfig::WEBP)
    else()
        find_package(WebP CONFIG REQUIRED)

        set(WEBP_FOUND TRUE)

        if(TARGET WebP::webp)
            set(ACMX2_WEBP_TARGET WebP::webp)
        elseif(TARGET WebP::webpdecoder)
            set(ACMX2_WEBP_TARGET WebP::webpdecoder)
        else()
            message(FATAL_ERROR
                "WebP was found, but no usable CMake target was provided."
            )
        endif()
    endif()

    if(NOT WEBP_FOUND)
        message(FATAL_ERROR
            "libwebp was not found. Install the WebP development package or "
            "configure with -DWEBP=OFF."
        )
    endif()

    target_compile_definitions(
        acmx2
        PRIVATE
            ACMX2_WITH_WEBP
    )

    target_link_libraries(
        acmx2
        PRIVATE
            ${ACMX2_WEBP_TARGET}
    )

    message(STATUS
        "WebP HDR snapshot support: ENABLED (libwebp ${WEBP_VERSION})"
    )
else()
    message(STATUS "WebP HDR snapshot support: DISABLED")
endif()

if(TIFF)
    if(ACMX2_USE_PKG_CONFIG)
        pkg_check_modules(
            TIFF
            REQUIRED
            IMPORTED_TARGET
            libtiff-4
        )

        set(ACMX2_TIFF_TARGET PkgConfig::TIFF)
    else()
        find_package(TIFF CONFIG REQUIRED)

        set(TIFF_FOUND TRUE)

        if(TARGET TIFF::TIFF)
            set(ACMX2_TIFF_TARGET TIFF::TIFF)
        else()
            message(FATAL_ERROR
                "TIFF was found, but no usable CMake target was provided."
            )
        endif()
    endif()

    if(NOT TIFF_FOUND)
        message(FATAL_ERROR
            "libtiff was not found. Install the TIFF development package or "
            "configure with -DTIFF=OFF."
        )
    endif()

    target_compile_definitions(
        acmx2
        PRIVATE
            ACMX2_WITH_TIFF
    )

    target_link_libraries(
        acmx2
        PRIVATE
            ${ACMX2_TIFF_TARGET}
    )

    message(STATUS
        "16-bit TIFF HDR snapshot support: ENABLED "
        "(libtiff ${TIFF_VERSION})"
    )
else()
    message(STATUS "16-bit TIFF HDR snapshot support: DISABLED")
endif()

if(AUDIO)
    target_compile_definitions(
        acmx2
        PRIVATE
            AUDIO_ENABLED
    )
    set_property(TARGET acmx2 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(MIDI)
    target_compile_definitions(
        acmx2
        PRIVATE
            MIDI_ENABLED
    )
endif()

if(MSVC)
    target_compile_options(
        acmx2
        PRIVATE
            /O2
            /W4
            /permissive-
    )
else()
    target_compile_options(
        acmx2
        PRIVATE
            -O3
            -Wall
            -pedantic
    )
endif()

target_include_directories(
    acmx2
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${AUDIO_INCLUDE}
        ${MIDI_INCLUDE}
        ${MXWrite_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${FFMPEG_INCLUDE_DIRS}
)

if(WITH_CUDA)
    if(NOT CUDAToolkit_INCLUDE_DIRS)
        message(FATAL_ERROR
            "CUDA toolkit include directories were not found."
        )
    endif()
endif()

if(WIN32 AND TARGET SDL2::SDL2main)
    target_link_libraries(
        acmx2
        PRIVATE
            SDL2::SDL2main
    )
endif()

if(MINGW)
    if(TARGET SDL2::SDL2main OR SDL2MAIN_LIBRARY)
        if(SDL2MAIN_LIBRARY AND NOT TARGET SDL2::SDL2main)
            target_link_libraries(
                acmx2
                PRIVATE
                    ${SDL2MAIN_LIBRARY}
            )
        endif()
    else()
        message(FATAL_ERROR
            "SDL2main is required for the MinGW ACMX2 executable, "
            "but it was not found."
        )
    endif()
endif()

target_link_libraries(
    acmx2
    PRIVATE
        libmx2::mxgl
        libmx2::mx
        SDL2::SDL2
        SDL2_image::SDL2_image
        SDL2_ttf::SDL2_ttf
        SDL2_mixer::SDL2_mixer
        ${OpenCV_LIBRARIES}
        ${AUDIO_LIBS}
        ${MIDI_LIBS}
        ${ACMX2_MXWRITE_TARGET}
        ${ACMX2_OPENGL_TARGET}
        Threads::Threads
        ${ACMX2_FFMPEG_TARGET}
)

if(WIN32 AND MIDI)
    # RtMidi's Windows backend uses the WinMM MIDI API.
    target_link_libraries(acmx2 PRIVATE winmm)
endif()

if(MINGW)
    target_link_libraries(
        acmx2
        PRIVATE
            opengl32
    )

    target_link_options(
        acmx2
        PRIVATE
            -mwindows
    )

    set_target_properties(
        acmx2
        PROPERTIES
            WIN32_EXECUTABLE TRUE
    )
elseif(WIN32)
    target_link_libraries(
        acmx2
        PRIVATE
            opengl32
    )

    set_target_properties(
        acmx2
        PROPERTIES
            WIN32_EXECUTABLE TRUE
    )
elseif(NOT APPLE)
    target_link_libraries(
        acmx2
        PRIVATE
            GL
    )
endif()

if(WITH_CUDA)
    target_link_libraries(
        acmx2
        PRIVATE
            acidcam-gpu::acidcam-gpu
            CUDA::cudart
            opencv_cudawarping
            opencv_cudaimgproc
            opencv_cudaarithm
    )
endif()

install(
    TARGETS acmx2
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    TARGETS audio_transfer
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/data/win-icon.png
    DESTINATION
        ${CMAKE_INSTALL_DATADIR}/acmx2
    RENAME
        acmx2.png
)

install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/data
    DESTINATION
        ${CMAKE_INSTALL_DATADIR}/acmx2
)

if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE
        @ONLY
    )

    add_custom_target(
        uninstall
        COMMAND
            ${CMAKE_COMMAND}
            -P
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    )
endif()
