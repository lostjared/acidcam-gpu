cmake_minimum_required(VERSION 3.20)

# Keep native Apple Silicon builds on arm64 unless the caller explicitly asks
# for another architecture or a universal binary.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin"
   AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$"
   AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "macOS target architectures")
endif()

project(ACMXVK VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(VALIDATION "Enable Vulkan validation layers" OFF)
option(AUDIO "Enable RtAudio input and audio-reactive shader metrics" OFF)
option(ACMXVK_USE_MOLTENVK "Enable MoltenVK portability support on Apple platforms" ${APPLE})

set(ACMXVK_MXWRITE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../MXWrite" CACHE PATH
    "MXWrite source directory containing the ACMXVK encoder API"
)
option(ACMXVK_USE_BUNDLED_MXWRITE
    "Build the repository MXWrite source instead of using an installed package"
    ON
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The installed MXVK export currently expects its dependency targets to have
# already been loaded by the consuming project.
if(NOT APPLE)
    find_package(CUDAToolkit QUIET)
endif()
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# MXWrite installations exist with both a combined PkgConfig::FFMPEG target
# and individual FFmpeg targets in their exported link interface. Define both
# forms before loading MXWrite so either package generation can be consumed.
if(NOT TARGET PkgConfig::FFMPEG)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
        libavcodec>=58.0.0
        libavformat>=58.0.0
        libavutil>=56.0.0
        libswscale>=5.0.0
        libswresample>=3.0.0
    )
endif()
if(NOT TARGET PkgConfig::AVCODEC)
    pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec>=58.0.0)
endif()
if(NOT TARGET PkgConfig::AVFORMAT)
    pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat>=58.0.0)
endif()
if(NOT TARGET PkgConfig::AVUTIL)
    pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil>=56.0.0)
endif()
if(NOT TARGET PkgConfig::SWSCALE)
    pkg_check_modules(SWSCALE REQUIRED IMPORTED_TARGET libswscale>=5.0.0)
endif()
if(NOT TARGET PkgConfig::SWRESAMPLE)
    pkg_check_modules(SWRESAMPLE REQUIRED IMPORTED_TARGET libswresample>=3.0.0)
endif()

if(ACMXVK_USE_BUNDLED_MXWRITE)
    if(NOT EXISTS "${ACMXVK_MXWRITE_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "ACMXVK_USE_BUNDLED_MXWRITE is ON, but a compatible MXWrite source "
            "was not found. Set ACMXVK_MXWRITE_SOURCE_DIR to the MXVK/MXWrite "
            "directory, or set ACMXVK_USE_BUNDLED_MXWRITE=OFF."
        )
    endif()
    file(READ "${ACMXVK_MXWRITE_SOURCE_DIR}/mxwrite.hpp"
        ACMXVK_MXWRITE_HEADER_CONTENT
    )
    if(NOT ACMXVK_MXWRITE_HEADER_CONTENT MATCHES "get_bytes_written"
       OR NOT ACMXVK_MXWRITE_HEADER_CONTENT MATCHES "video_encoder_options")
        message(FATAL_ERROR
            "${ACMXVK_MXWRITE_SOURCE_DIR}/mxwrite.hpp is older than the API "
            "required by ACMXVK. Use the MXWrite source from the current MXVK."
        )
    endif()
    message(STATUS "Using MXWrite source: ${ACMXVK_MXWRITE_SOURCE_DIR}")
    add_subdirectory(
        "${ACMXVK_MXWRITE_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/mxwrite"
        EXCLUDE_FROM_ALL
    )
    if(NOT TARGET mxwrite)
        message(FATAL_ERROR "The repository MXWrite source did not create target mxwrite")
    endif()
    if(NOT TARGET MXWrite::mxwrite)
        add_library(MXWrite::mxwrite ALIAS mxwrite)
    endif()
else()
    find_package(MXWrite CONFIG REQUIRED)
endif()
find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_ttf CONFIG REQUIRED)
find_package(Vulkan 1.4 REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

find_package(mxvk CONFIG REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

get_target_property(ACMXVK_MXVK_LIBRARIES mxvk::mxvk INTERFACE_LINK_LIBRARIES)

if(APPLE AND ACMXVK_MXVK_LIBRARIES MATCHES "MoltenVK")
    message(FATAL_ERROR
        "The installed MXVK package links MoltenVK directly. Rebuild and "
        "reinstall the current MXVK so SDL and volk both use Vulkan::Vulkan."
    )
endif()

if(ACMXVK_MXVK_LIBRARIES MATCHES "SDL3_mixer::")
    find_package(SDL3_mixer CONFIG REQUIRED)
endif()

if(ACMXVK_MXVK_LIBRARIES MATCHES "JPEG::")
    find_package(JPEG REQUIRED)
endif()

if(ACMXVK_MXVK_LIBRARIES MATCHES "CUDA::")
    if(APPLE)
        message(FATAL_ERROR
            "The installed MXVK package was built with CUDA. Rebuild MXVK "
            "without CUDA before using it with MoltenVK."
        )
    else()
        find_package(CUDAToolkit REQUIRED)
    endif()
endif()

set(ACMXVK_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(ACMXVK_SPRITE_VERTEX_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/sprite.vert.spv"
)
set(ACMXVK_ECHO_CACHE_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/echo_cache.frag.spv"
)
set(ACMXVK_FLIP_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/flip.frag.spv"
)
set(ACMXVK_CUSTOM_UNIFORM_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/custom_uniform.frag.spv"
)
set(ACMXVK_AUDIO_REACTIVE_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/audio_reactive.frag.spv"
)
set(ACMXVK_AUDIO_SPECTRUM_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/audio_spectrum.frag.spv"
)
set(ACMXVK_AUDIO_HISTORY_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/audio_history.frag.spv"
)

add_custom_command(
    OUTPUT "${ACMXVK_SPRITE_VERTEX_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sprite.vert"
            -o "${ACMXVK_SPRITE_VERTEX_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sprite.vert"
    COMMENT "Compiling ACMXVK sprite vertex shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_ECHO_CACHE_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/echo_cache.frag"
            -o "${ACMXVK_ECHO_CACHE_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/echo_cache.frag"
    COMMENT "Compiling ACMXVK history-cache test shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_FLIP_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/flip.frag"
            -o "${ACMXVK_FLIP_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/flip.frag"
    COMMENT "Compiling ACMXVK final-output flip shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_CUSTOM_UNIFORM_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/custom_uniform.frag"
            -o "${ACMXVK_CUSTOM_UNIFORM_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/custom_uniform.frag"
    COMMENT "Compiling ACMXVK custom-uniform test shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_AUDIO_REACTIVE_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/audio_reactive.frag"
            -o "${ACMXVK_AUDIO_REACTIVE_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/audio_reactive.frag"
    COMMENT "Compiling ACMXVK audio-reactive test shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_AUDIO_SPECTRUM_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/audio_spectrum.frag"
            -o "${ACMXVK_AUDIO_SPECTRUM_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/audio_spectrum.frag"
    COMMENT "Compiling ACMXVK FFT spectrum test shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_AUDIO_HISTORY_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/audio_history.frag"
            -o "${ACMXVK_AUDIO_HISTORY_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/audio_history.frag"
    COMMENT "Compiling ACMXVK FFT history test shader"
    VERBATIM
)

add_custom_target(acmxvk_shaders DEPENDS
    "${ACMXVK_SPRITE_VERTEX_SHADER}"
    "${ACMXVK_ECHO_CACHE_SHADER}"
    "${ACMXVK_FLIP_SHADER}"
    "${ACMXVK_CUSTOM_UNIFORM_SHADER}"
    "${ACMXVK_AUDIO_REACTIVE_SHADER}"
    "${ACMXVK_AUDIO_SPECTRUM_SHADER}"
    "${ACMXVK_AUDIO_HISTORY_SHADER}"
)

add_executable(acmxvk acmx.cpp)
add_dependencies(acmxvk acmxvk_shaders)
target_include_directories(acmxvk PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(acmxvk PRIVATE mxvk::mxvk MXWrite::mxwrite ${OpenCV_LIBS})
target_compile_options(acmxvk PRIVATE -Wall -Wextra -Wpedantic)
if(APPLE AND ACMXVK_USE_MOLTENVK)
    target_compile_definitions(acmxvk PRIVATE
        MXVK_USE_MOLTENVK
        VK_ENABLE_BETA_EXTENSIONS
    )
endif()
if(AUDIO)
    find_library(RTAUDIO_LIBRARY NAMES rtaudio REQUIRED)
    find_path(RTAUDIO_INCLUDE_DIR NAMES rtaudio/RtAudio.h REQUIRED)
    target_sources(acmxvk PRIVATE audio.cpp audio.hpp file_audio.cpp file_audio.hpp)
    target_include_directories(acmxvk PRIVATE "${RTAUDIO_INCLUDE_DIR}")
    target_link_libraries(acmxvk PRIVATE
        "${RTAUDIO_LIBRARY}"
        PkgConfig::AVCODEC
        PkgConfig::AVFORMAT
        PkgConfig::AVUTIL
        PkgConfig::SWRESAMPLE
    )
    target_compile_definitions(acmxvk PRIVATE AUDIO_ENABLED)
endif()
if(VALIDATION)
    target_compile_definitions(acmxvk PRIVATE ENABLE_VALIDATION)
endif()
target_compile_definitions(acmxvk PRIVATE
    ACMXVK_BUILD_SPRITE_VERTEX_SHADER="${ACMXVK_SPRITE_VERTEX_SHADER}"
    ACMXVK_INSTALL_SPRITE_VERTEX_SHADER="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders/sprite.vert.spv"
    ACMXVK_BUILD_ECHO_CACHE_SHADER="${ACMXVK_ECHO_CACHE_SHADER}"
    ACMXVK_INSTALL_ECHO_CACHE_SHADER="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders/echo_cache.frag.spv"
    ACMXVK_BUILD_FLIP_SHADER="${ACMXVK_FLIP_SHADER}"
    ACMXVK_INSTALL_FLIP_SHADER="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders/flip.frag.spv"
)

install(TARGETS acmxvk RUNTIME DESTINATION bin)
install(FILES
        "${ACMXVK_SPRITE_VERTEX_SHADER}"
        "${ACMXVK_ECHO_CACHE_SHADER}"
        "${ACMXVK_FLIP_SHADER}"
        "${ACMXVK_CUSTOM_UNIFORM_SHADER}"
        "${ACMXVK_AUDIO_REACTIVE_SHADER}"
        "${ACMXVK_AUDIO_SPECTRUM_SHADER}"
        "${ACMXVK_AUDIO_HISTORY_SHADER}"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/acmxvk/shaders")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/playlists/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/acmxvk/playlists")

if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        @ONLY
    )
    add_custom_target(uninstall
        COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        COMMENT "Uninstalling ACMXVK"
    )
endif()
