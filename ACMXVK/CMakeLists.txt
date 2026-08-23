cmake_minimum_required(VERSION 3.20)

project(ACMXVK VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(VALIDATION "Enable Vulkan validation layers" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The installed MXVK export currently expects its dependency targets to have
# already been loaded by the consuming project.
find_package(MXWrite CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_ttf CONFIG REQUIRED)
find_package(Vulkan 1.4 REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(glm CONFIG REQUIRED)
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
    find_package(CUDAToolkit REQUIRED)
endif()

if(ACMXVK_MXVK_LIBRARIES MATCHES "opencv_")
    find_package(OpenCV REQUIRED)
endif()

set(ACMXVK_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(ACMXVK_SPRITE_VERTEX_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/sprite.vert.spv"
)
set(ACMXVK_ECHO_CACHE_SHADER
    "${ACMXVK_SHADER_OUTPUT_DIR}/echo_cache.frag.spv"
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

add_custom_target(acmxvk_shaders DEPENDS
    "${ACMXVK_SPRITE_VERTEX_SHADER}"
    "${ACMXVK_ECHO_CACHE_SHADER}"
)

add_executable(acmxvk acmx.cpp)
add_dependencies(acmxvk acmxvk_shaders)
target_link_libraries(acmxvk PRIVATE mxvk::mxvk MXWrite::mxwrite)
target_compile_options(acmxvk PRIVATE -Wall -Wextra -Wpedantic)
if(VALIDATION)
    target_compile_definitions(acmxvk PRIVATE ENABLE_VALIDATION)
endif()
target_compile_definitions(acmxvk PRIVATE
    ACMXVK_BUILD_SPRITE_VERTEX_SHADER="${ACMXVK_SPRITE_VERTEX_SHADER}"
    ACMXVK_INSTALL_SPRITE_VERTEX_SHADER="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders/sprite.vert.spv"
    ACMXVK_BUILD_ECHO_CACHE_SHADER="${ACMXVK_ECHO_CACHE_SHADER}"
    ACMXVK_INSTALL_ECHO_CACHE_SHADER="${CMAKE_INSTALL_FULL_DATADIR}/acmxvk/shaders/echo_cache.frag.spv"
)

install(TARGETS acmxvk RUNTIME DESTINATION bin)
install(FILES
        "${ACMXVK_SPRITE_VERTEX_SHADER}"
        "${ACMXVK_ECHO_CACHE_SHADER}"
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
