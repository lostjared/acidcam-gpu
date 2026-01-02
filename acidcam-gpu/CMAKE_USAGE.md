# Using acidcam-gpu in Your CMake Projects

This guide explains how to use the acidcam-gpu library in your own CMake projects.

## Installation

First, build and install acidcam-gpu:

```bash
cd acidcam-gpu
mkdir build
cd build
cmake ..
make
sudo make install
```

By default, it installs to `/usr/local`. To install to a custom location:

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
sudo make install
```

## Using in Your CMake Project

### Method 1: Using find_package()

This is the recommended approach. In your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyProject)

# Find the acidcam-gpu package
find_package(acidcam-gpu REQUIRED CONFIG)

# Create your executable or library
add_executable(my_app main.cpp)

# Link against acidcam-gpu
target_link_libraries(my_app PRIVATE acidcam-gpu::acidcam-gpu)
```

If you installed acidcam-gpu to a custom location:

```cmake
find_package(acidcam-gpu REQUIRED CONFIG 
    HINTS /path/to/install/lib/cmake/acidcam-gpu)
```

### Method 2: Using CMake Install Prefix Path

Set the `CMAKE_PREFIX_PATH` when configuring your project:

```bash
cmake .. -DCMAKE_PREFIX_PATH=/usr/local
```

## Include Paths

The acidcam-gpu headers will automatically be added to your target's include directories when you use `target_link_libraries()` with `acidcam-gpu::acidcam-gpu`.

To access the headers directly:

```cpp
#include "ac-gpu/ac-gpu.hpp"
```

## Available Targets

The following targets are available:

- `acidcam-gpu::acidcam-gpu` - The shared library with all headers

## Checking Installation

To verify that acidcam-gpu is installed correctly:

```bash
ls /usr/local/lib/cmake/acidcam-gpu/
# Should show: acidcam-gpuConfig.cmake, acidcam-gpuConfigVersion.cmake, acidcam-gpuTargets.cmake
```

## Example Project

See `example-CMakeLists.txt` in this directory for a complete example of how to use the library.

## Troubleshooting

### "Could not find acidcam-gpu"

Make sure:
1. acidcam-gpu is installed: `ls /usr/local/lib/cmake/acidcam-gpu/`
2. CMAKE_PREFIX_PATH includes the install directory
3. You're using the correct find_package syntax: `find_package(acidcam-gpu REQUIRED CONFIG)`

### Missing Dependencies

acidcam-gpu requires:
- OpenCV
- CUDA Toolkit
- FFmpeg development libraries
- MXWrite library

Install these before building acidcam-gpu.

### Library Not Found at Runtime

If you get runtime errors about `libacidcam-gpu.so` not being found:

1. Add the library path to `LD_LIBRARY_PATH`:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   ```

2. Or set `RPATH` when building your project:
   ```cmake
   set(CMAKE_INSTALL_RPATH "/usr/local/lib")
   ```

## Version Information

The current version of acidcam-gpu is **1.0.0**.

You can check the version in your CMake code:

```cmake
find_package(acidcam-gpu REQUIRED CONFIG)
message(STATUS "Found acidcam-gpu version: ${acidcam-gpu_VERSION}")
```
