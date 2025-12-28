cmake_minimum_required(VERSION 3.18)
project(acidcam-gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)

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
)

target_link_libraries(acidcam-gpu
    PRIVATE
    ${OpenCV_LIBS}
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