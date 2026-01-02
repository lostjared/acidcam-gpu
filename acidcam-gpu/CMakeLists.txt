cmake_minimum_required(VERSION 3.18)
project(acidcam-gpu 
    VERSION 1.0.0
    DESCRIPTION "GPU-accelerated video filter library using CUDA"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set install directories for CMake package config
set(CMAKE_INSTALL_INCLUDEDIR "include")
set(CMAKE_INSTALL_LIBDIR "lib")
set(CMAKE_INSTALL_BINDIR "bin")

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(MXWrite REQUIRED)


pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswscale
)

message(STATUS "FFMPEG LIBRARIES: ${FFMPEG_LIBRARIES}")
message(STATUS "FFMPEG INCLUDE DIRS: ${FFMPEG_INCLUDE_DIRS}")
message(STATUS "MXWrite: ${MXWrite_LIBRARIES}")



add_library(acidcam-gpu SHARED
    src/filters.cu
)

set_target_properties(acidcam-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(acidcam-gpu PRIVATE 
    -O3               
    -march=native     
    -ffast-math       
    -fomit-frame-pointer
)

target_include_directories(acidcam-gpu
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_INSTALL_PREFIX}/include
    ${CMAKE_PREFIX_PATH}/include
    ${FFMPEG_INCLUDE_DIRS}
    ${MXWrite_INCLUDE_DIRS}
)

target_link_libraries(acidcam-gpu
    PUBLIC
    ${OpenCV_LIBS}
    MXWrite::mxwrite
)

add_executable(acidcam
    app/main_cv.cu
)

target_compile_options(acidcam PRIVATE 
    -O3               
    -march=native     
    -ffast-math       
    -fomit-frame-pointer
)

set_target_properties(acidcam PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
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
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include/acidcam-gpu
    FILES_MATCHING PATTERN "*.hpp"
)

set_target_properties(acidcam-gpu acidcam PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

install(TARGETS acidcam-gpu
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(TARGETS acidcam
    RUNTIME DESTINATION bin
)

# Install CMake configuration files
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/acidcam-gpuConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/acidcam-gpuConfig.cmake"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/acidcam-gpuConfigVersion.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/acidcam-gpuConfigVersion.cmake"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/acidcam-gpuTargets.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/acidcam-gpuTargets.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/acidcam-gpuConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/acidcam-gpuConfigVersion.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/acidcam-gpuTargets.cmake"
    DESTINATION lib/cmake/acidcam-gpu
)
