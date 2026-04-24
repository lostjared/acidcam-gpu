cmake_minimum_required(VERSION 3.16)
project(midi-map LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_library(RTMIDI_LIBRARY NAMES rtmidi REQUIRED)
find_path(RTMIDI_INCLUDE_DIR NAMES rtmidi/RtMidi.h PATHS /usr/include /usr/local/include)
if(NOT RTMIDI_LIBRARY)
    message(FATAL_ERROR "RtMidi library not found. Please install rtmidi.")
endif()
if(NOT RTMIDI_INCLUDE_DIR)
    message(FATAL_ERROR "RtMidi headers not found. Please install rtmidi development files.")
endif()

add_executable(midi-map
    main.cpp
    midi_window.cpp
)

target_compile_options(midi-map PRIVATE -O3 -Wall -pedantic)

target_include_directories(midi-map PRIVATE
    ${RTMIDI_INCLUDE_DIR}
)

target_link_libraries(midi-map PRIVATE
    Qt6::Widgets
    ${RTMIDI_LIBRARY}
)

if(MINGW)
    target_sources(midi-map PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/win-icon.rc
        ${CMAKE_CURRENT_SOURCE_DIR}/win-icon.ico
    )
    target_link_options(midi-map PRIVATE -mwindows)
    set_target_properties(midi-map PROPERTIES WIN32_EXECUTABLE TRUE)
endif()

install(TARGETS midi-map DESTINATION bin)
