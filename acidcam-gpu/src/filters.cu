#include "ac-gpu/ac-gpu.hpp"
#include <cuda_runtime.h>
#include <string>

namespace ac_gpu {

    Filters filters[] = { 0, "SelfAlphaBlend" };

    __device__ void applySelfAlphaBlend(int x, int y, unsigned char* data, int width, int height, size_t step, float alpha, bool isNegative) {
        if (x < width && y < height) {
            int idx = y * step + x * 4; 
            unsigned char b = data[idx];
            unsigned char g = data[idx + 1];
            unsigned char r = data[idx + 2];
            data[idx]     = (unsigned char)(b + (b * alpha)); 
            data[idx + 1] = (unsigned char)(g + (g * alpha)); 
            data[idx + 2] = (unsigned char)(r + (r * alpha)); 
            if (isNegative) {
                data[idx]     = 255 - data[idx];
                data[idx + 1] = 255 - data[idx + 1];
                data[idx + 2] = 255 - data[idx + 2];
            }
            data[idx + 3] = 255;
        }
    }

    
    __global__ void filterKernel(int filterIndex, unsigned char* data, int width, int height, size_t step, float alpha, bool isNegative) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        switch(filterIndex) {
            case 0:
                applySelfAlphaBlend(x, y, data, width, height, step, alpha, isNegative);
                break;
            case 1:
                break;
        }
    }
}

void launch_filter(int index, unsigned char* data, int width, int height, size_t step, float alpha, bool isNegative) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                    (height + blockSize.y - 1) / blockSize.y);

    ac_gpu::filterKernel<<<gridSize, blockSize>>>(index, data, width, height, step, alpha, isNegative);        
    cudaDeviceSynchronize(); 
}