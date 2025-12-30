cmake_minimum_required(VERSION 3.18)
project(acidcam-gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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