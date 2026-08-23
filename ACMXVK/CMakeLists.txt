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
option(ACMXVK_USE_MOLTENVK "Link ACMXVK with MoltenVK on Apple platforms" ${APPLE})

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
find_package(MXWrite CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_ttf CONFIG REQUIRED)
if(APPLE AND ACMXVK_USE_MOLTENVK AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    find_package(Vulkan 1.4 REQUIRED OPTIONAL_COMPONENTS MoltenVK)
else()
    find_package(Vulkan 1.4 REQUIRED)
endif()
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

# FindVulkan did not expose Vulkan::MoltenVK until CMake 3.24. Create the same
# dependency target for older CMake releases and Vulkan SDK layouts so the
# installed MXVK export can be loaded reliably.
if(APPLE AND ACMXVK_USE_MOLTENVK AND NOT TARGET Vulkan::MoltenVK)
    set(ACMXVK_MOLTENVK_HINTS /opt/homebrew/lib /usr/local/lib)
    if(DEFINED ENV{VULKAN_SDK})
        list(PREPEND ACMXVK_MOLTENVK_HINTS
            "$ENV{VULKAN_SDK}/lib"
            "$ENV{VULKAN_SDK}/macOS/lib"
        )
    endif()

    if(Vulkan_MoltenVK_LIBRARY AND NOT ACMXVK_MOLTENVK_LIBRARY)
        set(ACMXVK_MOLTENVK_LIBRARY "${Vulkan_MoltenVK_LIBRARY}")
    else()
        find_library(ACMXVK_MOLTENVK_LIBRARY
            NAMES MoltenVK
            HINTS ${ACMXVK_MOLTENVK_HINTS}
        )
    endif()
    if(NOT ACMXVK_MOLTENVK_LIBRARY)
        message(FATAL_ERROR
            "MoltenVK was not found. Source the Vulkan SDK environment or set "
            "Vulkan_MoltenVK_LIBRARY/ACMXVK_MOLTENVK_LIBRARY to libMoltenVK."
        )
    endif()

    add_library(Vulkan::MoltenVK INTERFACE IMPORTED)
    set_property(TARGET Vulkan::MoltenVK PROPERTY
        INTERFACE_LINK_LIBRARIES "${ACMXVK_MOLTENVK_LIBRARY}"
    )
    if(Vulkan_INCLUDE_DIRS)
        set_property(TARGET Vulkan::MoltenVK PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIRS}"
        )
    endif()
endif()

find_package(mxvk CONFIG REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

get_target_property(ACMXVK_MXVK_LIBRARIES mxvk::mxvk INTERFACE_LINK_LIBRARIES)

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

add_custom_target(acmxvk_shaders DEPENDS
    "${ACMXVK_SPRITE_VERTEX_SHADER}"
    "${ACMXVK_ECHO_CACHE_SHADER}"
    "${ACMXVK_FLIP_SHADER}"
    "${ACMXVK_CUSTOM_UNIFORM_SHADER}"
    "${ACMXVK_AUDIO_REACTIVE_SHADER}"
    "${ACMXVK_AUDIO_SPECTRUM_SHADER}"
)

add_executable(acmxvk acmx.cpp)
add_dependencies(acmxvk acmxvk_shaders)
target_include_directories(acmxvk PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(acmxvk PRIVATE mxvk::mxvk MXWrite::mxwrite ${OpenCV_LIBS})
target_compile_options(acmxvk PRIVATE -Wall -Wextra -Wpedantic)
if(APPLE AND ACMXVK_USE_MOLTENVK)
    find_library(ACMXVK_METAL_FRAMEWORK Metal REQUIRED)
    find_library(ACMXVK_QUARTZ_CORE_FRAMEWORK QuartzCore REQUIRED)
    find_library(ACMXVK_COCOA_FRAMEWORK Cocoa REQUIRED)
    find_library(ACMXVK_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ACMXVK_IOKIT_FRAMEWORK IOKit REQUIRED)
    target_link_libraries(acmxvk PRIVATE
        Vulkan::MoltenVK
        "${ACMXVK_METAL_FRAMEWORK}"
        "${ACMXVK_QUARTZ_CORE_FRAMEWORK}"
        "${ACMXVK_COCOA_FRAMEWORK}"
        "${ACMXVK_FOUNDATION_FRAMEWORK}"
        "${ACMXVK_IOKIT_FRAMEWORK}"
    )
    target_compile_definitions(acmxvk PRIVATE
        MXVK_USE_MOLTENVK
        VK_ENABLE_BETA_EXTENSIONS
    )
endif()
if(AUDIO)
    find_library(RTAUDIO_LIBRARY NAMES rtaudio REQUIRED)
    find_path(RTAUDIO_INCLUDE_DIR NAMES rtaudio/RtAudio.h REQUIRED)
    target_sources(acmxvk PRIVATE audio.cpp audio.hpp)
    target_include_directories(acmxvk PRIVATE "${RTAUDIO_INCLUDE_DIR}")
    target_link_libraries(acmxvk PRIVATE "${RTAUDIO_LIBRARY}")
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
