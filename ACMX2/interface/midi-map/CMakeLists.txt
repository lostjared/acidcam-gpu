cmake_minimum_required(VERSION 3.16)
project(midi-map LANGUAGES CXX)

if(WIN32)
    enable_language(RC)
endif()

# CMAKE_BUILD_TYPE is only meaningful for single-configuration generators
# (for example Ninja or Unix Makefiles).  Visual Studio selects the
# configuration at build time, so do not force one into its cache.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Package managers such as vcpkg provide RtMidi's CMake target.  Retain a
# library/header fallback for distributions that only install the raw files.
find_package(RtMidi CONFIG QUIET)
if(TARGET RtMidi::rtmidi)
    set(RTMIDI_TARGET RtMidi::rtmidi)
elseif(TARGET rtmidi)
    set(RTMIDI_TARGET rtmidi)
else()
    find_library(RTMIDI_LIBRARY NAMES rtmidi RtMidi rtmidi-static REQUIRED)
    find_path(RTMIDI_INCLUDE_DIR NAMES rtmidi/RtMidi.h REQUIRED)

    add_library(RtMidi::rtmidi UNKNOWN IMPORTED)
    set_target_properties(RtMidi::rtmidi PROPERTIES
        IMPORTED_LOCATION "${RTMIDI_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${RTMIDI_INCLUDE_DIR}"
    )
    set(RTMIDI_TARGET RtMidi::rtmidi)
endif()

add_executable(midi-map
    main.cpp
    midi_window.cpp
)

if(MSVC)
    target_compile_options(midi-map PRIVATE /W4 /permissive-)
else()
    target_compile_options(midi-map PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

target_link_libraries(midi-map PRIVATE
    Qt6::Widgets
    ${RTMIDI_TARGET}
)

if(WIN32)
    target_sources(midi-map PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/win-icon.rc
    )
    set_target_properties(midi-map PROPERTIES WIN32_EXECUTABLE TRUE)
endif()

install(TARGETS midi-map DESTINATION bin)

if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    )
endif()
