# acidcam-gpu CMake Package Configuration

This directory now includes complete CMake configuration files for using the acidcam-gpu library in other CMake projects.

## Files Created

### Configuration Files (.cmake.in templates)

1. **acidcam-gpuConfig.cmake.in**
   - Main package configuration file
   - Finds dependencies (OpenCV, MXWrite, FFmpeg)
   - Imports the acidcam-gpu target
   - Sets backward compatibility variables

2. **acidcam-gpuConfigVersion.cmake.in**
   - Version information (1.0.0)
   - Creates imported library target
   - Sets version variables
   - Links required dependencies

3. **acidcam-gpuTargets.cmake.in**
   - Defines the imported library target: `acidcam-gpu::acidcam-gpu`
   - Sets library location and include directories
   - Marks as CUDA-aware target

### Documentation

4. **CMAKE_USAGE.md**
   - Complete guide for using acidcam-gpu in other projects
   - Installation instructions
   - Usage examples
   - Troubleshooting tips

5. **example-CMakeLists.txt**
   - Working example of a CMake project that uses acidcam-gpu
   - Shows proper linking and compilation flags
   - Demonstrates best practices

### Updated CMakeLists.txt

The main CMakeLists.txt has been updated to:
- Add project version (1.0.0) and description
- Define standard install directories
- Configure and install the CMake package files to `lib/cmake/acidcam-gpu/`

## Installation

To install acidcam-gpu with CMake configurations:

```bash
cd /home/jared/acidcam-gpu/acidcam-gpu
mkdir build && cd build
cmake ..
make
sudo make install
```

This will install:
- Library: `/usr/local/lib/libacidcam-gpu.so`
- Headers: `/usr/local/include/acidcam-gpu/`
- CMake configs: `/usr/local/lib/cmake/acidcam-gpu/`

## Using in Your Project

In your CMakeLists.txt:

```cmake
find_package(acidcam-gpu REQUIRED CONFIG)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE acidcam-gpu::acidcam-gpu)
```

See CMAKE_USAGE.md for detailed instructions.

## CMake Version

- Minimum required: CMake 3.18 (for CUDA support)
- Tested with: CMake 3.x+

## Target Details

The `acidcam-gpu::acidcam-gpu` imported target provides:
- **INTERFACE_INCLUDE_DIRECTORIES**: Points to the installed header files
- **IMPORTED_LOCATION**: Path to libacidcam-gpu.so
- **INTERFACE_LINK_LIBRARIES**: Automatically links OpenCV, MXWrite, and FFmpeg

## Key Improvements

✓ Standard CMake package format (Config mode)
✓ Version information available to consumer projects
✓ Proper dependency handling through CMake
✓ Target-based linking (modern CMake style)
✓ Works with `find_package()`
✓ Backward compatible with pkg-config
✓ Supports custom installation paths

## Next Steps

1. Build and install the library
2. Other projects can use: `find_package(acidcam-gpu REQUIRED CONFIG)`
3. Link with: `target_link_libraries(target acidcam-gpu::acidcam-gpu)`
