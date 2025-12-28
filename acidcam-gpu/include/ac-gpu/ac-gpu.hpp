#ifndef __AC_GPU_HPP__
#define __AC_GPU_HPP__

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>   
#include <thread>  
#include <string>

namespace ac_gpu {

    struct Filters {
        size_t index;
        std::string name;
    };

    extern Filters filters[];
    __global__ void filterKernel(int filterIndex, unsigned char* data, int width, int height, size_t step, float alpha, bool isNegative);
}


extern "C" void launch_filter(int index, unsigned char* data, int width, int height, size_t step, float alpha, bool isNegative);

#endif