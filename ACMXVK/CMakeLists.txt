cmake_minimum_required(VERSION 3.20)

# Keep native Apple Silicon builds on arm64 unless the caller explicitly asks
# for another architecture or a universal binary.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin"
   AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$"
   AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "macOS target architectures")
endif()

project(ACMXVK VERSION 0.1.0 LANGUAGES CXX)

if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)
endif()

include(GNUInstallDirs)
include(CTest)

option(VALIDATION "Enable Vulkan validation layers" OFF)
option(AUDIO "Enable RtAudio input and audio-reactive shader metrics" OFF)
option(MIDI "Enable RtMidi input support" OFF)
option(WEBP "Enable lossless WebP snapshots" OFF)
option(TIFF "Enable lossless TIFF snapshots" OFF)
option(WITH_CUDA "Enable acidcam-gpu CUDA filters" OFF)
option(WITH_OPENCV_DNN "Enable OpenCV DNN effects" OFF)
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
       OR NOT ACMXVK_MXWRITE_HEADER_CONTENT MATCHES "video_encoder_options"
       OR NOT ACMXVK_MXWRITE_HEADER_CONTENT MATCHES
           "write_hdr_rgba16_at_pts")
        message(FATAL_ERROR
            "${ACMXVK_MXWRITE_SOURCE_DIR}/mxwrite.hpp is older than the API "
            "required by ACMXVK HDR increment 5. Use the MXWrite source from "
            "the current MXVK."
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
if(WITH_OPENCV_DNN)
    find_package(OpenCV REQUIRED COMPONENTS
        core
        imgproc
        imgcodecs
        highgui
        videoio
        dnn
    )
else()
    find_package(OpenCV REQUIRED)
endif()

if(WEBP)
    pkg_check_modules(ACMXVK_WEBP REQUIRED IMPORTED_TARGET libwebp)
endif()
if(TIFF)
    pkg_check_modules(ACMXVK_TIFF REQUIRED IMPORTED_TARGET libtiff-4)
endif()

find_package(mxvk 0.33.1 CONFIG REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

get_target_property(ACMXVK_MXVK_INCLUDE_DIRECTORIES
    mxvk::mxvk INTERFACE_INCLUDE_DIRECTORIES)
set(ACMXVK_MXVK_HDR_INPUT_API OFF)
foreach(ACMXVK_MXVK_INCLUDE_DIRECTORY IN LISTS
        ACMXVK_MXVK_INCLUDE_DIRECTORIES)
    set(ACMXVK_MXVK_SPRITE_HEADER
        "${ACMXVK_MXVK_INCLUDE_DIRECTORY}/mxvk/mxvk_sprite.hpp")
    set(ACMXVK_MXVK_CAPTURE_HEADER
        "${ACMXVK_MXVK_INCLUDE_DIRECTORY}/mxvk/mxvk_ff_capture.hpp")
    set(ACMXVK_MXVK_WINDOW_HEADER
        "${ACMXVK_MXVK_INCLUDE_DIRECTORY}/mxvk/mxvk.hpp")
    set(ACMXVK_MXVK_MODEL_HEADER
        "${ACMXVK_MXVK_INCLUDE_DIRECTORY}/mxvk/mxvk_abstract_model.hpp")
    if(EXISTS "${ACMXVK_MXVK_SPRITE_HEADER}"
       AND EXISTS "${ACMXVK_MXVK_CAPTURE_HEADER}"
       AND EXISTS "${ACMXVK_MXVK_WINDOW_HEADER}"
       AND EXISTS "${ACMXVK_MXVK_MODEL_HEADER}")
        file(READ "${ACMXVK_MXVK_SPRITE_HEADER}"
            ACMXVK_MXVK_SPRITE_HEADER_CONTENT)
        file(READ "${ACMXVK_MXVK_CAPTURE_HEADER}"
            ACMXVK_MXVK_CAPTURE_HEADER_CONTENT)
        file(READ "${ACMXVK_MXVK_WINDOW_HEADER}"
            ACMXVK_MXVK_WINDOW_HEADER_CONTENT)
        file(READ "${ACMXVK_MXVK_MODEL_HEADER}"
            ACMXVK_MXVK_MODEL_HEADER_CONTENT)
        if(ACMXVK_MXVK_SPRITE_HEADER_CONTENT MATCHES
               "createEmptySpriteRgba16"
           AND ACMXVK_MXVK_SPRITE_HEADER_CONTENT MATCHES
               "enableHistoryTextureRgba16Float"
           AND ACMXVK_MXVK_CAPTURE_HEADER_CONTENT MATCHES "readRgba16"
           AND ACMXVK_MXVK_WINDOW_HEADER_CONTENT MATCHES
               "setHdrRenderIntermediatesEnabled"
           AND ACMXVK_MXVK_WINDOW_HEADER_CONTENT MATCHES
               "setFrameReadbackRgba16Enabled"
           AND ACMXVK_MXVK_WINDOW_HEADER_CONTENT MATCHES
               "setPostProcessingPresentFragmentShader"
           AND ACMXVK_MXVK_MODEL_HEADER_CONTENT MATCHES
               "setColorAttachmentFormat")
            set(ACMXVK_MXVK_HDR_INPUT_API ON)
            break()
        endif()
    endif()
endforeach()
if(NOT ACMXVK_MXVK_HDR_INPUT_API)
    message(FATAL_ERROR
        "The selected MXVK package predates ACMXVK HDR increment 6. Rebuild "
        "and reinstall the current MXVK source tree, then configure ACMXVK "
        "again.")
endif()

get_target_property(ACMXVK_MXVK_LIBRARIES mxvk::mxvk INTERFACE_LINK_LIBRARIES)

set(ACMXVK_MXVK_CUDA_AVAILABLE OFF)
if(NOT APPLE AND ACMXVK_MXVK_LIBRARIES MATCHES "CUDA::")
    set(ACMXVK_MXVK_CUDA_AVAILABLE ON)
    set(ACMXVK_OPENCV_COMPONENTS
        core
        imgproc
        imgcodecs
        highgui
        videoio
        cudaarithm
    )
    if(WITH_OPENCV_DNN)
        list(APPEND ACMXVK_OPENCV_COMPONENTS dnn)
    endif()
    find_package(OpenCV REQUIRED COMPONENTS
        ${ACMXVK_OPENCV_COMPONENTS}
    )
endif()

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

if(WITH_CUDA)
    if(APPLE)
        message(FATAL_ERROR
            "WITH_CUDA is not available with MoltenVK. Configure ACMXVK with "
            "-DWITH_CUDA=OFF on Apple platforms."
        )
    endif()
    if(NOT ACMXVK_MXVK_LIBRARIES MATCHES "CUDA::")
        message(FATAL_ERROR
            "WITH_CUDA requires an MXVK installation built with CUDA support."
        )
    endif()
    find_package(CUDAToolkit REQUIRED)
    find_package(acidcam-gpu CONFIG REQUIRED)
endif()

set(ACMXVK_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(ACMXVK_TEST_SHADER_MANIFEST
    "${ACMXVK_SHADER_OUTPUT_DIR}/library.json"
)
set(ACMXVK_SPRITE_VERTEX_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/sprite.vert.spv"
)
set(ACMXVK_ECHO_CACHE_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/echo_cache.frag.spv"
)
set(ACMXVK_FLIP_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/flip.frag.spv"
)
set(ACMXVK_PASSTHROUGH_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/passthrough.frag.spv"
)
set(ACMXVK_HUMAN_COMPOSITE_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/human_composite.frag.spv"
)
set(ACMXVK_MODEL_VERTEX_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/model.vert.spv"
)
set(ACMXVK_MODEL_FRAGMENT_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/model.frag.spv"
)
set(ACMXVK_CUSTOM_UNIFORM_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/custom_uniform.frag.spv"
)
set(ACMXVK_MIDI_SLIDER_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/midi_slider.frag.spv"
)
set(ACMXVK_AUDIO_REACTIVE_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/audio_reactive.frag.spv"
)
set(ACMXVK_AUDIO_SPECTRUM_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/audio_spectrum.frag.spv"
)
set(ACMXVK_COMPUTE_TEST_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/compute_test.comp.spv"
)
set(ACMXVK_HDR_COMPUTE_TEST_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/compute_test_hdr.comp.spv"
)
set(ACMXVK_HDR_TRANSFER_SHADERS)
set(ACMXVK_HDR_TRANSFER_NAMES pq_decode pq_encode hlg_decode hlg_encode)
set(ACMXVK_HDR_TRANSFER_DEFINES
    ACMXVK_HDR_PQ_DECODE ACMXVK_HDR_PQ_ENCODE
    ACMXVK_HDR_HLG_DECODE ACMXVK_HDR_HLG_ENCODE)
foreach(ACMXVK_HDR_TRANSFER_INDEX RANGE 0 3)
    list(GET ACMXVK_HDR_TRANSFER_NAMES ${ACMXVK_HDR_TRANSFER_INDEX}
        ACMXVK_HDR_TRANSFER_NAME)
    list(GET ACMXVK_HDR_TRANSFER_DEFINES ${ACMXVK_HDR_TRANSFER_INDEX}
        ACMXVK_HDR_TRANSFER_DEFINE)
    set(ACMXVK_HDR_TRANSFER_OUTPUT
        "${ACMXVK_SHADER_OUTPUT_DIR}/hdr_${ACMXVK_HDR_TRANSFER_NAME}.frag.spv")
    add_custom_command(
        OUTPUT "${ACMXVK_HDR_TRANSFER_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${ACMXVK_SHADER_OUTPUT_DIR}"
        COMMAND "${GLSLC_EXECUTABLE}"
                -D${ACMXVK_HDR_TRANSFER_DEFINE}=1
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hdr_transfer.frag"
                -o "${ACMXVK_HDR_TRANSFER_OUTPUT}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hdr_transfer.frag"
        COMMENT "Compiling ACMXVK HDR ${ACMXVK_HDR_TRANSFER_NAME} shader"
        VERBATIM
    )
    list(APPEND ACMXVK_HDR_TRANSFER_SHADERS
        "${ACMXVK_HDR_TRANSFER_OUTPUT}")
endforeach()
set(ACMXVK_HDR_PREVIEW_SHADERS)
foreach(ACMXVK_HDR_PREVIEW_NAME pq hlg)
    string(TOUPPER "${ACMXVK_HDR_PREVIEW_NAME}"
        ACMXVK_HDR_PREVIEW_DEFINE_SUFFIX)
    set(ACMXVK_HDR_PREVIEW_OUTPUT
        "${ACMXVK_SHADER_OUTPUT_DIR}/hdr_preview_${ACMXVK_HDR_PREVIEW_NAME}.frag.spv")
    add_custom_command(
        OUTPUT "${ACMXVK_HDR_PREVIEW_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${ACMXVK_SHADER_OUTPUT_DIR}"
        COMMAND "${GLSLC_EXECUTABLE}"
                -DACMXVK_HDR_PREVIEW_${ACMXVK_HDR_PREVIEW_DEFINE_SUFFIX}=1
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hdr_preview.frag"
                -o "${ACMXVK_HDR_PREVIEW_OUTPUT}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hdr_preview.frag"
        COMMENT "Compiling ACMXVK ${ACMXVK_HDR_PREVIEW_NAME} HDR preview shader"
        VERBATIM
    )
    list(APPEND ACMXVK_HDR_PREVIEW_SHADERS
        "${ACMXVK_HDR_PREVIEW_OUTPUT}")
endforeach()
set(ACMXVK_MANDALA_FRACTAL_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/mandala_fractal.comp.spv"
)
set(ACMXVK_AUDIO_HISTORY_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/audio_history.frag.spv"
)
set(ACMXVK_CROSSFADE_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/xfade"
)
set(ACMXVK_CROSSFADE_OUTPUT_DIR
    "${ACMXVK_SHADER_OUTPUT_DIR}/xfade"
)
file(GLOB ACMXVK_CROSSFADE_SHADER_SOURCES CONFIGURE_DEPENDS
    "${ACMXVK_CROSSFADE_SOURCE_DIR}/xfade_*.glsl"
)
set(ACMXVK_CROSSFADE_SHADERS)
foreach(ACMXVK_CROSSFADE_SOURCE IN LISTS ACMXVK_CROSSFADE_SHADER_SOURCES)
    get_filename_component(ACMXVK_CROSSFADE_NAME
        "${ACMXVK_CROSSFADE_SOURCE}" NAME_WE
    )
    set(ACMXVK_CROSSFADE_OUTPUT
        "${ACMXVK_CROSSFADE_OUTPUT_DIR}/${ACMXVK_CROSSFADE_NAME}.frag.spv"
    )
    add_custom_command(
        OUTPUT "${ACMXVK_CROSSFADE_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${ACMXVK_CROSSFADE_OUTPUT_DIR}"
        COMMAND "${GLSLC_EXECUTABLE}" -fshader-stage=fragment
                "${ACMXVK_CROSSFADE_SOURCE}"
                -o "${ACMXVK_CROSSFADE_OUTPUT}"
        DEPENDS "${ACMXVK_CROSSFADE_SOURCE}"
        COMMENT "Compiling ACMXVK crossfade ${ACMXVK_CROSSFADE_NAME}"
        VERBATIM
    )
    list(APPEND ACMXVK_CROSSFADE_SHADERS "${ACMXVK_CROSSFADE_OUTPUT}")
endforeach()
set(ACMXVK_OVERLAY_FONT
    "${CMAKE_CURRENT_SOURCE_DIR}/../ACMX2/data/font.ttf"
    CACHE FILEPATH "TrueType font used by ACMXVK overlays"
)
if(NOT EXISTS "${ACMXVK_OVERLAY_FONT}")
    message(FATAL_ERROR
        "ACMXVK overlay font was not found at ${ACMXVK_OVERLAY_FONT}. "
        "Set ACMXVK_OVERLAY_FONT to a readable TrueType font."
    )
endif()
set(ACMXVK_BUILD_RESOURCE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(ACMXVK_BUILD_OVERLAY_FONT
    "${ACMXVK_BUILD_RESOURCE_DIRECTORY}/data/font.ttf"
)
configure_file(
    "${ACMXVK_OVERLAY_FONT}"
    "${ACMXVK_BUILD_OVERLAY_FONT}"
    COPYONLY
)

add_custom_command(
    OUTPUT "${ACMXVK_COMPUTE_TEST_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_test.comp"
            -o "${ACMXVK_COMPUTE_TEST_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_test.comp"
    COMMENT "Compiling ACMXVK compute-pass test shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_HDR_COMPUTE_TEST_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}" -DACMXVK_HDR_COMPUTE=1
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_test.comp"
            -o "${ACMXVK_HDR_COMPUTE_TEST_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_test.comp"
    COMMENT "Compiling ACMXVK RGBA16F compute-pass test shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_MANDALA_FRACTAL_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mandala_fractal.comp"
            -o "${ACMXVK_MANDALA_FRACTAL_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mandala_fractal.comp"
    COMMENT "Compiling ACMXVK slider-controlled mandala fractal shader"
    VERBATIM
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
    OUTPUT "${ACMXVK_PASSTHROUGH_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/passthrough.frag"
            -o "${ACMXVK_PASSTHROUGH_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/passthrough.frag"
    COMMENT "Compiling ACMXVK shader-bypass pass"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_HUMAN_COMPOSITE_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/human_composite.frag"
            -o "${ACMXVK_HUMAN_COMPOSITE_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/human_composite.frag"
    COMMENT "Compiling ACMXVK human foreground composite shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_MODEL_VERTEX_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/model.vert"
            -o "${ACMXVK_MODEL_VERTEX_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/model.vert"
    COMMENT "Compiling ACMXVK 3D model vertex shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ACMXVK_MODEL_FRAGMENT_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/model.frag"
            -o "${ACMXVK_MODEL_FRAGMENT_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/model.frag"
    COMMENT "Compiling ACMXVK 3D model fragment shader"
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
    OUTPUT
        "${ACMXVK_MIDI_SLIDER_SHADER}"
        "${ACMXVK_TEST_SHADER_MANIFEST}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ACMXVK_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/midi_slider.frag"
            -o "${ACMXVK_MIDI_SLIDER_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/library.json"
            "${ACMXVK_TEST_SHADER_MANIFEST}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/midi_slider.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/library.json"
    COMMENT "Compiling ACMXVK MIDI slider test shader"
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
    "${ACMXVK_PASSTHROUGH_SHADER}"
    "${ACMXVK_HUMAN_COMPOSITE_SHADER}"
    "${ACMXVK_MODEL_VERTEX_SHADER}"
    "${ACMXVK_MODEL_FRAGMENT_SHADER}"
    "${ACMXVK_CUSTOM_UNIFORM_SHADER}"
    "${ACMXVK_MIDI_SLIDER_SHADER}"
    "${ACMXVK_TEST_SHADER_MANIFEST}"
    "${ACMXVK_AUDIO_REACTIVE_SHADER}"
    "${ACMXVK_AUDIO_SPECTRUM_SHADER}"
    "${ACMXVK_COMPUTE_TEST_SHADER}"
    "${ACMXVK_HDR_COMPUTE_TEST_SHADER}"
    ${ACMXVK_HDR_TRANSFER_SHADERS}
    ${ACMXVK_HDR_PREVIEW_SHADERS}
    "${ACMXVK_MANDALA_FRACTAL_SHADER}"
    "${ACMXVK_AUDIO_HISTORY_SHADER}"
    ${ACMXVK_CROSSFADE_SHADERS}
)

add_executable(acmxvk
    acmx.cpp
    main_window.cpp
    main_window.hpp
    app/interface_client.cpp
    app/interface_client.hpp
    app/camera_probe.hpp
    app/media_helpers.cpp
    app/media_helpers.hpp
    app/media_utils.cpp
    app/media_utils.hpp
    app/options.cpp
    app/options.hpp
    app/output_paths.cpp
    app/output_paths.hpp
    app/playlist.cpp
    app/playlist.hpp
    app/resource_paths.cpp
    app/resource_paths.hpp
    app/shader_library.cpp
    app/shader_library.hpp
    app/snapshot_writer.cpp
    app/snapshot_writer.hpp
    input_validation.cpp
    input_validation.hpp
)
if(APPLE)
    target_sources(acmxvk PRIVATE app/camera_probe_macos.mm)
    find_library(ACMXVK_AVFOUNDATION_FRAMEWORK AVFoundation REQUIRED)
    find_library(ACMXVK_COREMEDIA_FRAMEWORK CoreMedia REQUIRED)
    find_library(ACMXVK_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_link_libraries(acmxvk PRIVATE
        "${ACMXVK_AVFOUNDATION_FRAMEWORK}"
        "${ACMXVK_COREMEDIA_FRAMEWORK}"
        "${ACMXVK_FOUNDATION_FRAMEWORK}"
    )
else()
    target_sources(acmxvk PRIVATE app/camera_probe.cpp)
endif()
add_dependencies(acmxvk acmxvk_shaders)
target_include_directories(acmxvk PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(acmxvk PRIVATE
    mxvk::mxvk
    MXWrite::mxwrite
    PkgConfig::AVFORMAT
    PkgConfig::AVUTIL
    ${OpenCV_LIBS}
)
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
if(MIDI)
    find_library(RTMIDI_LIBRARY NAMES rtmidi REQUIRED)
    find_path(RTMIDI_INCLUDE_DIR NAMES rtmidi/RtMidi.h REQUIRED)
    target_sources(acmxvk PRIVATE midi.cpp midi.hpp)
    target_include_directories(acmxvk PRIVATE "${RTMIDI_INCLUDE_DIR}")
    target_link_libraries(acmxvk PRIVATE "${RTMIDI_LIBRARY}")
    target_compile_definitions(acmxvk PRIVATE MIDI_ENABLED)
endif()
if(WEBP)
    target_link_libraries(acmxvk PRIVATE PkgConfig::ACMXVK_WEBP)
    target_compile_definitions(acmxvk PRIVATE ACMXVK_WITH_WEBP)
    message(STATUS
        "WebP snapshot support: ENABLED (libwebp ${ACMXVK_WEBP_VERSION})"
    )
else()
    message(STATUS "WebP snapshot support: DISABLED")
endif()
if(TIFF)
    target_link_libraries(acmxvk PRIVATE PkgConfig::ACMXVK_TIFF)
    target_compile_definitions(acmxvk PRIVATE ACMXVK_WITH_TIFF)
    message(STATUS
        "TIFF snapshot support: ENABLED (libtiff ${ACMXVK_TIFF_VERSION})"
    )
else()
    message(STATUS "TIFF snapshot support: DISABLED")
endif()
if(WITH_OPENCV_DNN)
    target_sources(acmxvk PRIVATE edge_dnn.cpp edge_dnn.hpp)
    target_compile_definitions(acmxvk PRIVATE ACMXVK_WITH_DNN)
    message(STATUS "OpenCV DNN effects: ENABLED (independent of acidcam-gpu)")
    if(NOT WITH_CUDA)
        message(STATUS "acidcam-gpu filters: DISABLED; libacidcam-gpu will not be linked")
    endif()
else()
    message(STATUS "OpenCV DNN effects: DISABLED")
endif()
if(ACMXVK_MXVK_CUDA_AVAILABLE)
    target_link_libraries(acmxvk PRIVATE opencv_cudaarithm)
    target_compile_definitions(acmxvk PRIVATE ACMXVK_WITH_MXVK_CUDA)
endif()
if(WITH_CUDA)
    target_sources(acmxvk PRIVATE gpu_filters.cpp gpu_filters.hpp)
    target_link_libraries(acmxvk PRIVATE
        acidcam-gpu::acidcam-gpu
        CUDA::cudart
    )
    target_compile_definitions(acmxvk PRIVATE ACMXVK_WITH_CUDA)
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
    ACMXVK_BUILD_PASSTHROUGH_SHADER="${ACMXVK_PASSTHROUGH_SHADER}"
    ACMXVK_INSTALL_PASSTHROUGH_SHADER="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders/passthrough.frag.spv"
    ACMXVK_BUILD_HDR_TRANSFER_DIRECTORY="${ACMXVK_SHADER_OUTPUT_DIR}"
    ACMXVK_INSTALL_HDR_TRANSFER_DIRECTORY="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders"
    ACMXVK_BUILD_HUMAN_COMPOSITE_SHADER="${ACMXVK_HUMAN_COMPOSITE_SHADER}"
    ACMXVK_INSTALL_HUMAN_COMPOSITE_SHADER="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders/human_composite.frag.spv"
    ACMXVK_BUILD_MODEL_VERTEX_SHADER="${ACMXVK_MODEL_VERTEX_SHADER}"
    ACMXVK_INSTALL_MODEL_VERTEX_SHADER="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders/model.vert.spv"
    ACMXVK_BUILD_MODEL_FRAGMENT_SHADER="${ACMXVK_MODEL_FRAGMENT_SHADER}"
    ACMXVK_INSTALL_MODEL_FRAGMENT_SHADER="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders/model.frag.spv"
    ACMXVK_BUILD_DEFAULT_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/models/cube.obj"
    ACMXVK_INSTALL_DEFAULT_MODEL="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/models/cube.obj"
    ACMXVK_BUILD_RESOURCE_DIRECTORY="${ACMXVK_BUILD_RESOURCE_DIRECTORY}"
    ACMXVK_INSTALL_RESOURCE_DIRECTORY="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk"
    ACMXVK_BUILD_OVERLAY_FONT="${ACMXVK_BUILD_OVERLAY_FONT}"
    ACMXVK_INSTALL_OVERLAY_FONT="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/data/font.ttf"
    ACMXVK_BUILD_CROSSFADE_DIRECTORY="${ACMXVK_CROSSFADE_OUTPUT_DIR}"
    ACMXVK_INSTALL_CROSSFADE_DIRECTORY="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders/xfade"
)

if(BUILD_TESTING)
    if(WIN32)
        add_executable(acmxvk_interface_client_windows_tests
            interface_client_windows_tests.cpp
            app/interface_client.cpp
            app/interface_client.hpp
            interface_control.hpp
        )
        target_compile_options(acmxvk_interface_client_windows_tests PRIVATE
            -Wall -Wextra -Wpedantic
        )
        add_test(NAME acmxvk_interface_client_windows
            COMMAND acmxvk_interface_client_windows_tests
        )
    endif()
    add_executable(acmxvk_input_validation_tests
        input_validation_tests.cpp
        input_validation.cpp
        input_validation.hpp
    )
    target_compile_options(acmxvk_input_validation_tests PRIVATE
        -Wall -Wextra -Wpedantic
    )
    add_test(NAME acmxvk_input_validation
        COMMAND acmxvk_input_validation_tests
    )
    add_executable(acmxvk_resource_paths_tests
        resource_paths_tests.cpp
        app/resource_paths.cpp
        app/resource_paths.hpp
        app/options.hpp
    )
    target_compile_options(acmxvk_resource_paths_tests PRIVATE
        -Wall -Wextra -Wpedantic
    )
    add_test(NAME acmxvk_resource_paths
        COMMAND acmxvk_resource_paths_tests
    )
    add_executable(acmxvk_playlist_tests
        playlist_tests.cpp
        app/playlist.cpp
        app/playlist.hpp
        app/options.hpp
        input_validation.cpp
        input_validation.hpp
    )
    target_compile_options(acmxvk_playlist_tests PRIVATE
        -Wall -Wextra -Wpedantic
    )
    add_test(NAME acmxvk_playlist
        COMMAND acmxvk_playlist_tests
    )
    add_executable(acmxvk_output_paths_tests
        output_paths_tests.cpp
        app/output_paths.cpp
        app/output_paths.hpp
        app/snapshot_writer.hpp
        app/options.hpp
    )
    target_compile_options(acmxvk_output_paths_tests PRIVATE
        -Wall -Wextra -Wpedantic
    )
    add_test(NAME acmxvk_output_paths
        COMMAND acmxvk_output_paths_tests
    )
    add_test(NAME acmxvk_shader_library_build
        COMMAND acmxvk
            --build "${CMAKE_CURRENT_SOURCE_DIR}/tests/shader_library/library.json"
            --builddir "${CMAKE_CURRENT_BINARY_DIR}/shader-library-build-test"
            --glslc "${GLSLC_EXECUTABLE}"
    )
    add_test(NAME acmxvk_shader_library_fix
        COMMAND "${CMAKE_COMMAND}"
            -DACMXVK_EXECUTABLE=$<TARGET_FILE:acmxvk>
            -DGLSLC_EXECUTABLE=${GLSLC_EXECUTABLE}
            -DSOURCE_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}/tests/shader_library_fix
            -DOUTPUT_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/shader-library-fix-test
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_shader_library_fix.cmake"
    )
endif()

install(TARGETS acmxvk RUNTIME DESTINATION bin)
install(FILES
        "${ACMXVK_SPRITE_VERTEX_SHADER}"
        "${ACMXVK_ECHO_CACHE_SHADER}"
        "${ACMXVK_FLIP_SHADER}"
        "${ACMXVK_PASSTHROUGH_SHADER}"
        "${ACMXVK_HUMAN_COMPOSITE_SHADER}"
        "${ACMXVK_MODEL_VERTEX_SHADER}"
        "${ACMXVK_MODEL_FRAGMENT_SHADER}"
        "${ACMXVK_CUSTOM_UNIFORM_SHADER}"
        "${ACMXVK_MIDI_SLIDER_SHADER}"
        "${ACMXVK_AUDIO_REACTIVE_SHADER}"
        "${ACMXVK_AUDIO_SPECTRUM_SHADER}"
        "${ACMXVK_COMPUTE_TEST_SHADER}"
        "${ACMXVK_HDR_COMPUTE_TEST_SHADER}"
        ${ACMXVK_HDR_TRANSFER_SHADERS}
        ${ACMXVK_HDR_PREVIEW_SHADERS}
        "${ACMXVK_MANDALA_FRACTAL_SHADER}"
        "${ACMXVK_AUDIO_HISTORY_SHADER}"
        "${ACMXVK_TEST_SHADER_MANIFEST}"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/acmxvk/shaders")
install(FILES ${ACMXVK_CROSSFADE_SHADERS}
        DESTINATION "${CMAKE_INSTALL_DATADIR}/acmxvk/shaders/xfade")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/playlists/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/acmxvk/playlists")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/midi-examples/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/acmxvk/midi-examples")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/models/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/acmxvk/models")
install(FILES "${ACMXVK_OVERLAY_FONT}"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/acmxvk/data"
        RENAME font.ttf)

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
