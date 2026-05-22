cmake_minimum_required(VERSION 3.10)
project(ACMX2 VERSION 2.18.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenGL_GL_PREFERENCE GLVND)
include(CheckIncludeFileCXX)
include(CheckLibraryExists)
include(CheckCXXSourceCompiles)
find_package(PkgConfig REQUIRED)
if(NOT PKG_CONFIG_FOUND)
    message(FATAL_ERROR "pkg-config not found. Please install pkg-config.")
endif()
option(WITH_CUDA "Enable CUDA / acidcam-gpu GPU acceleration" ON)
option(WITH_OPENCV_DNN "Enable OpenCV DNN/CUDA" OFF)
option(WEBP "Save HDR snapshots as WebP instead of 16-bit PNG" OFF)
option(TIFF "Enable saving HDR snapshots as 16-bit TIFF (key '4')" OFF)

find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(SDL2_mixer REQUIRED)
find_package(libmx2 REQUIRED)
if(WITH_CUDA)
    if(WITH_OPENCV_DNN)
        find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio cudaimgproc dnn)
    else()
        find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio cudaimgproc)
    endif()
else()
    if(WITH_OPENCV_DNN)
        find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)
    else()
        find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
    endif()
endif()
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV development libraries.")
endif()
message(STATUS "OpenCV found: version ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
if(WITH_CUDA AND NOT OpenCV_CUDA_VERSION)
    message(WARNING "OpenCV CUDA support may not be available. Some features might not work.")
endif()
set(ACMX2_LOCAL_MXWRITE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../MXWrite")
set(ACMX2_MXWRITE_TARGET "")

if(EXISTS "${ACMX2_LOCAL_MXWRITE_DIR}/CMakeLists.txt")
    add_subdirectory("${ACMX2_LOCAL_MXWRITE_DIR}" "${CMAKE_BINARY_DIR}/acmx2_mxwrite_build")
    set(ACMX2_MXWRITE_TARGET mxwrite)
    set(MXWrite_INCLUDE_DIRS "${ACMX2_LOCAL_MXWRITE_DIR}")
    set(MXWrite_FOUND TRUE)
    message(STATUS "MXWrite source build enabled from: ${ACMX2_LOCAL_MXWRITE_DIR}")
else()
    find_package(MXWrite REQUIRED)
    if(NOT MXWrite_FOUND)
        message(FATAL_ERROR "MXWrite not found. Please ensure MXWrite is installed and CMAKE_PREFIX_PATH is set correctly.")
    endif()
    set(ACMX2_MXWRITE_TARGET MXWrite::mxwrite)
    if(NOT MXWrite_INCLUDE_DIRS)
        get_target_property(MXWrite_INCLUDE_DIRS ${ACMX2_MXWRITE_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    endif()
    message(STATUS "MXWrite found: ${MXWrite_INCLUDE_DIRS}")
endif()
if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDA Toolkit not found. Please install CUDA Toolkit or configure with -DWITH_CUDA=OFF.")
    endif()
    message(STATUS "CUDA Toolkit found: version ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    find_package(acidcam-gpu REQUIRED CONFIG)
    if(NOT acidcam-gpu_FOUND)
        message(FATAL_ERROR "acidcam-gpu not found. Please ensure acidcam-gpu is installed and CMAKE_PREFIX_PATH is set correctly, or configure with -DWITH_CUDA=OFF.")
    endif()
    message(STATUS "acidcam-gpu found")
else()
    message(STATUS "CUDA / acidcam-gpu support DISABLED (WITH_CUDA=OFF)")
    set(CUDAToolkit_INCLUDE_DIRS "")
endif()
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswscale
    libswresample
)
if(NOT FFMPEG_FOUND)
    message(FATAL_ERROR "FFMPEG libraries not found. Please install ffmpeg development libraries (libavcodec, libavformat, libavutil, libswscale).")
endif()
message(STATUS "FFMPEG found: ${FFMPEG_LIBRARIES}")

set(CMAKE_REQUIRED_INCLUDES 
    ${libmx2_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${MXWrite_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    /usr/local/include
    /usr/local/include/opencv4
)
if(WITH_CUDA)
    list(APPEND CMAKE_REQUIRED_INCLUDES /opt/cuda/include)
endif()
check_include_file_cxx("vector" HAVE_VECTOR)
check_include_file_cxx("string" HAVE_STRING)
check_include_file_cxx("fstream" HAVE_FSTREAM)
check_include_file_cxx("filesystem" HAVE_FILESYSTEM)
check_include_file_cxx("thread" HAVE_THREAD)
check_include_file_cxx("mutex" HAVE_MUTEX)
check_include_file_cxx("atomic" HAVE_ATOMIC)
check_include_file_cxx("optional" HAVE_OPTIONAL)
check_include_file_cxx("string_view" HAVE_STRING_VIEW)
check_include_file_cxx("deque" HAVE_DEQUE)
check_include_file_cxx("queue" HAVE_QUEUE)
check_include_file_cxx("condition_variable" HAVE_CONDITION_VARIABLE)
check_include_file_cxx("chrono" HAVE_CHRONO)
check_include_file_cxx("algorithm" HAVE_ALGORITHM)
check_include_file_cxx("unordered_map" HAVE_UNORDERED_MAP)
if(NOT HAVE_FILESYSTEM)
    message(FATAL_ERROR "C++17 <filesystem> header not found. Please ensure your compiler supports C++17.")
endif()
if(NOT HAVE_OPTIONAL OR NOT HAVE_STRING_VIEW)
    message(FATAL_ERROR "C++17 headers (<optional>, <string_view>) not found. Please ensure your compiler supports C++17/C++20.")
endif()
set(CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS} /usr/local/include/opencv4)
check_include_file_cxx("opencv2/opencv.hpp" HAVE_OPENCV_HPP)
if(WITH_CUDA)
    check_include_file_cxx("opencv2/core/cuda.hpp" HAVE_OPENCV_CUDA_HPP)
    check_include_file_cxx("opencv2/cudaimgproc.hpp" HAVE_OPENCV_CUDAIMGPROC_HPP)
endif()
if(NOT HAVE_OPENCV_HPP)
    message(FATAL_ERROR "OpenCV headers not found. Please install OpenCV development libraries.")
endif()
if(WITH_CUDA AND (NOT HAVE_OPENCV_CUDA_HPP OR NOT HAVE_OPENCV_CUDAIMGPROC_HPP))
    message(WARNING "OpenCV CUDA headers not found. CUDA-accelerated image processing may not be available.")
endif()
option(AUDIO "Audio support" OFF)
if(AUDIO)
    find_library(RTAUDIO_LIBRARY NAMES rtaudio REQUIRED)
    find_path(RTAUDIO_INCLUDE_DIR NAMES rtaudio/RtAudio.h PATHS /usr/include /usr/local/include)
    if(NOT RTAUDIO_LIBRARY)
        message(FATAL_ERROR "RtAudio library not found. Please install rtaudio or disable AUDIO option.")
    endif()
    if(NOT RTAUDIO_INCLUDE_DIR)
        message(FATAL_ERROR "RtAudio headers not found. Please install rtaudio development files or disable AUDIO option.")
    endif()
    set(CMAKE_REQUIRED_INCLUDES ${RTAUDIO_INCLUDE_DIR})
    check_include_file_cxx("rtaudio/RtAudio.h" HAVE_RTAUDIO_H)
    if(NOT HAVE_RTAUDIO_H)
        check_include_file_cxx("RtAudio.h" HAVE_RTAUDIO_H_ALT)
        if(NOT HAVE_RTAUDIO_H_ALT)
            message(FATAL_ERROR "RtAudio.h header not found. Please install rtaudio development files.")
        endif()
    endif()
    set(AUDIO_INCLUDE ${RTAUDIO_INCLUDE_DIR})
    set(AUDIO_LIBS ${RTAUDIO_LIBRARY})
    set(AUDIO_SRC "audio.cpp" "file_audio.cpp")
    message(STATUS "AUDIO ENABLED")
    message(STATUS "AUDIO LIBRARY: ${RTAUDIO_LIBRARY}")
    message(STATUS "AUDIO INCLUDE: ${RTAUDIO_INCLUDE_DIR}")
else()
    set(AUDIO_INCLUDE "")
    set(AUDIO_LIBS "")
    set(AUDIO_SRC "")
    message(STATUS "AUDIO DISABLED")
endif()
option(MIDI "MIDI Support" OFF)
if(MIDI)
    find_library(RTMIDI_LIBRARY NAMES rtmidi REQUIRED)
    find_path(RTMIDI_INCLUDE_DIR NAMES rtmidi/RtMidi.h PATHS /usr/include /usr/local/include)
    if(NOT RTMIDI_LIBRARY)
        message(FATAL_ERROR "RtMidi library not found. Please install rtmidi or disable MIDI option.")
    endif()
    if(NOT RTMIDI_INCLUDE_DIR)
        message(FATAL_ERROR "RtMidi headers not found. Please install rtmidi development files or disable MIDI option.")
    endif()
    set(MIDI_INCLUDE ${RTMIDI_INCLUDE_DIR})
    set(MIDI_LIBS ${RTMIDI_LIBRARY})
    message(STATUS "MIDI ENABLED")
    message(STATUS "MIDI LIBRARY: ${RTMIDI_LIBRARY}")
    message(STATUS "MIDI INCLUDE: ${RTMIDI_INCLUDE_DIR}")
else()
    set(MIDI_INCLUDE "")
    set(MIDI_LIBS "")
    message(STATUS "MIDI DISABLED")
endif()
set(CMAKE_REQUIRED_FLAGS "-std=c++20")
check_cxx_source_compiles("
    #include <concepts>
    #include <ranges>
    int main() { return 0; }
" HAVE_CXX20_FEATURES)
if(NOT HAVE_CXX20_FEATURES)
    message(WARNING "Full C++20 support not detected. Some features may be limited.")
endif()
set(CMAKE_REQUIRED_FLAGS "")
check_cxx_source_compiles("
    #include <thread>
    #include <mutex>
    int main() {
        std::mutex m;
        std::thread t([&m]{ std::lock_guard<std::mutex> lock(m); });
        t.join();
        return 0;
    }
" HAVE_THREADING)
if(NOT HAVE_THREADING)
    message(FATAL_ERROR "Threading support not available. Please ensure your compiler supports std::thread and std::mutex.")
endif()
find_package(OpenGL REQUIRED)
if(NOT OpenGL_FOUND)
    message(FATAL_ERROR "OpenGL not found. Please install OpenGL development libraries.")
endif()
if(TARGET OpenGL::GL)
    set(ACMX2_OPENGL_TARGET OpenGL::GL)
elseif(TARGET OpenGL::OpenGL)
    set(ACMX2_OPENGL_TARGET OpenGL::OpenGL)
else()
    message(FATAL_ERROR "No usable OpenGL imported target found (expected OpenGL::GL or OpenGL::OpenGL).")
endif()
message(STATUS "OpenGL target selected: ${ACMX2_OPENGL_TARGET}")
message(STATUS "")
message(STATUS "=== ACMX2 Configuration Summary ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(WITH_CUDA)
    message(STATUS "CUDA Support: ENABLED (version ${CUDAToolkit_VERSION})")
else()
    message(STATUS "CUDA Support: DISABLED")
endif()
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Audio Support: ${AUDIO}")
message(STATUS "FFMPEG Libraries: ${FFMPEG_LIBRARIES}")
message(STATUS "FFMPEG Include Dirs: ${FFMPEG_INCLUDE_DIRS}")
message(STATUS "MXWrite: ${MXWrite_INCLUDE_DIRS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===================================")
find_package(Threads REQUIRED)
add_executable(audio_transfer audio_transfer.cpp)
target_compile_options(audio_transfer PRIVATE -O3 -Wall -pedantic)
target_include_directories(audio_transfer PRIVATE 
    ${libmx2_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS} 
    ${SDL2_TTF_INCLUDE_DIRS}
    ${MXWrite_INCLUDE_DIRS}
)
target_link_libraries(audio_transfer PRIVATE 
    ${ACMX2_MXWRITE_TARGET}
    PkgConfig::FFMPEG
)
if(WITH_OPENCV_DNN)
    set(ACMX2_DNN_SRC "dnn.cpp")
    pkg_check_modules(YAMLCPP REQUIRED IMPORTED_TARGET yaml-cpp)
else()
    set(ACMX2_DNN_SRC "")
endif()
add_executable(acmx2 acmx.cpp program.cpp ${AUDIO_SRC} ${ACMX2_DNN_SRC})
set_target_properties(acmx2 PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_compile_definitions(acmx2 PRIVATE WITH_GL)
if(WITH_CUDA)
    if(WITH_OPENCV_DNN)
    target_compile_definitions(acmx2 PRIVATE ACMX2_WITH_DNN)
    target_link_libraries(acmx2 PRIVATE PkgConfig::YAMLCPP)
    endif()
    target_compile_definitions(acmx2 PRIVATE ACMX2_WITH_CUDA)
endif()
if(WEBP)
    pkg_check_modules(WEBP REQUIRED IMPORTED_TARGET libwebp)
    if(NOT WEBP_FOUND)
        message(FATAL_ERROR "libwebp not found via pkg-config. Install libwebp development package or configure with -DWEBP=OFF.")
    endif()
    target_compile_definitions(acmx2 PRIVATE ACMX2_WITH_WEBP)
    target_link_libraries(acmx2 PRIVATE PkgConfig::WEBP)
    message(STATUS "WebP HDR snapshot support: ENABLED (libwebp ${WEBP_VERSION})")
else()
    message(STATUS "WebP HDR snapshot support: DISABLED")
endif()
if(TIFF)
    pkg_check_modules(TIFF REQUIRED IMPORTED_TARGET libtiff-4)
    if(NOT TIFF_FOUND)
        message(FATAL_ERROR "libtiff-4 not found via pkg-config. Install libtiff development package or configure with -DTIFF=OFF.")
    endif()
    target_compile_definitions(acmx2 PRIVATE ACMX2_WITH_TIFF)
    target_link_libraries(acmx2 PRIVATE PkgConfig::TIFF)
    message(STATUS "16-bit TIFF HDR snapshot support: ENABLED (libtiff ${TIFF_VERSION})")
else()
    message(STATUS "16-bit TIFF HDR snapshot support: DISABLED")
endif()
if(AUDIO)
    target_compile_definitions(acmx2 PRIVATE AUDIO_ENABLED)
endif()
if(MIDI)
    target_compile_definitions(acmx2 PRIVATE MIDI_ENABLED)
endif()
target_compile_options(acmx2 PRIVATE -O3 -Wall -pedantic)
target_include_directories(acmx2 PRIVATE 
    ${libmx2_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS} 
    ${SDL2_TTF_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${AUDIO_INCLUDE}/rtaudio
    ${MIDI_INCLUDE}
    ${CMAKE_INSTALL_PREFIX}/include
    ${CMAKE_PREFIX_PATH}/include
    ${MXWrite_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${FFMPEG_INCLUDE_DIRS}
    /usr/local/include
    /usr/local/include/opencv4
)
if(WITH_CUDA)
    target_include_directories(acmx2 PRIVATE /opt/cuda/include)
endif()
target_link_libraries(acmx2 PRIVATE 
    libmx2::mx 
    libmx2::mxgl
    ${OpenCV_LIBRARIES} 
    ${AUDIO_LIBS}
    ${MIDI_LIBS}
    ${ACMX2_MXWRITE_TARGET}
    ${ACMX2_OPENGL_TARGET}
    Threads::Threads
    PkgConfig::FFMPEG
    ${FFMPEG_LIBRARIES}
)
if(MINGW)
    target_sources(acmx2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/interface/win-icon.rc
        ${CMAKE_CURRENT_SOURCE_DIR}/interface/win-icon.ico
    )
    target_link_libraries(acmx2 PRIVATE
        SDL2::SDL2main
        opengl32
    )
    target_link_options(acmx2 PRIVATE -mwindows)
    set_target_properties(acmx2 PROPERTIES WIN32_EXECUTABLE TRUE)
elseif(NOT APPLE)
    target_link_libraries(acmx2 PRIVATE GL)
endif()
if(WITH_CUDA)
    target_link_libraries(acmx2 PRIVATE
        acidcam-gpu::acidcam-gpu
        CUDA::cudart
        opencv_cudawarping
        opencv_cudaimgproc
        opencv_cudaarithm
    )
endif()
install(TARGETS acmx2 DESTINATION bin)
install(TARGETS audio_transfer DESTINATION bin)
# acmx2 is a command-line application; only the Qt interface (acmx2_interface)
# installs a .desktop file into share/applications.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/win-icon.png DESTINATION share/acmx2 RENAME acmx2.png)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION share/acmx2)

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
