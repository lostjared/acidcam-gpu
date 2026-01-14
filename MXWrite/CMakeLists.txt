cmake_minimum_required(VERSION 3.10)
project(MXWrite LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(GNUInstallDirs)
include(CheckIncludeFileCXX)

find_package(PkgConfig REQUIRED)

pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec>=58.0.0
    libavformat>=58.0.0
    libavutil>=56.0.0
    libswscale>=5.0.0
)

if(NOT FFMPEG_FOUND)
    message(FATAL_ERROR "FFmpeg libraries not found. Please install FFmpeg development packages.")
endif()

set(CMAKE_REQUIRED_INCLUDES ${FFMPEG_INCLUDE_DIRS})
check_include_file_cxx("libavcodec/avcodec.h" HAVE_AVCODEC_H)
check_include_file_cxx("libavformat/avformat.h" HAVE_AVFORMAT_H)
check_include_file_cxx("libavutil/imgutils.h" HAVE_AVUTIL_IMGUTILS_H)
check_include_file_cxx("libavutil/mathematics.h" HAVE_AVUTIL_MATHEMATICS_H)
check_include_file_cxx("libavutil/opt.h" HAVE_AVUTIL_OPT_H)
check_include_file_cxx("libswscale/swscale.h" HAVE_SWSCALE_H)

if(NOT HAVE_AVCODEC_H)
    message(FATAL_ERROR "Required header libavcodec/avcodec.h not found")
endif()
if(NOT HAVE_AVFORMAT_H)
    message(FATAL_ERROR "Required header libavformat/avformat.h not found")
endif()
if(NOT HAVE_AVUTIL_IMGUTILS_H)
    message(FATAL_ERROR "Required header libavutil/imgutils.h not found")
endif()
if(NOT HAVE_AVUTIL_MATHEMATICS_H)
    message(FATAL_ERROR "Required header libavutil/mathematics.h not found")
endif()
if(NOT HAVE_AVUTIL_OPT_H)
    message(FATAL_ERROR "Required header libavutil/opt.h not found")
endif()
if(NOT HAVE_SWSCALE_H)
    message(FATAL_ERROR "Required header libswscale/swscale.h not found")
endif()


check_include_file_cxx("string" HAVE_STRING_H)
check_include_file_cxx("chrono" HAVE_CHRONO_H)
check_include_file_cxx("thread" HAVE_THREAD_H)
check_include_file_cxx("mutex" HAVE_MUTEX_H)
check_include_file_cxx("queue" HAVE_QUEUE_H)
check_include_file_cxx("string_view" HAVE_STRING_VIEW_H)

if(NOT HAVE_STRING_H)
    message(FATAL_ERROR "Required C++ header <string> not found")
endif()
if(NOT HAVE_CHRONO_H)
    message(FATAL_ERROR "Required C++ header <chrono> not found")
endif()
if(NOT HAVE_THREAD_H)
    message(FATAL_ERROR "Required C++ header <thread> not found")
endif()
if(NOT HAVE_MUTEX_H)
    message(FATAL_ERROR "Required C++ header <mutex> not found")
endif()
if(NOT HAVE_QUEUE_H)
    message(FATAL_ERROR "Required C++ header <queue> not found")
endif()
if(NOT HAVE_STRING_VIEW_H)
    message(FATAL_ERROR "Required C++ header <string_view> not found (C++17 required)")
endif()


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

message(STATUS "Found FFmpeg:")
message(STATUS "  libavcodec version: ${FFMPEG_libavcodec_VERSION}")
message(STATUS "  libavformat version: ${FFMPEG_libavformat_VERSION}")
message(STATUS "  libavutil version: ${FFMPEG_libavutil_VERSION}")
message(STATUS "  libswscale version: ${FFMPEG_libswscale_VERSION}")
message(STATUS "  Include directories: ${FFMPEG_INCLUDE_DIRS}")
message(STATUS "  Library directories: ${FFMPEG_LIBRARY_DIRS}")

add_library(mxwrite STATIC mxwrite.cpp)
target_include_directories(mxwrite PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${FFMPEG_INCLUDE_DIRS}
)
target_link_libraries(mxwrite PUBLIC PkgConfig::FFMPEG Threads::Threads)
target_compile_options(mxwrite PRIVATE -O3 -Wall -Wextra -pedantic)
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MXWriteConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/MXWriteConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MXWriteConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MXWrite
)

install(FILES mxwrite.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS mxwrite
    EXPORT MXWriteTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES 
    "${CMAKE_CURRENT_BINARY_DIR}/MXWriteConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MXWriteConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MXWrite
)

install(EXPORT MXWriteTargets
    NAMESPACE MXWrite::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MXWrite
)

export(EXPORT MXWriteTargets
    NAMESPACE MXWrite::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/MXWriteTargets.cmake"
)