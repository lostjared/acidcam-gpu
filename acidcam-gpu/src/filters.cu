#include "ac-gpu/ac-gpu.hpp"
#include <cuda_runtime.h>
#include <string>

namespace ac_gpu {

    Filters filters[] = { {0, "SelfAlphaBlend"} };

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

    __device__ void sort_window_25(unsigned char* window) {
        for (int i = 0; i < 25; i++) {
            for (int j = i + 1; j < 25; j++) {
                if (window[i] > window[j]) {
                    unsigned char tmp = window[i];
                    window[i] = window[j];
                    window[j] = tmp;
                }
            }
        }
    }

    __global__ void medianBlur5x5Kernel(unsigned char* data, int width, int height, size_t step) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return;

        for (int c = 0; c < 3; ++c) { 
            unsigned char window[25];
            int count = 0;
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    window[count++] = data[(y + dy) * step + (x + dx) * 4 + c];
                }
            }
            sort_window_25(window);
            data[y * step + x * 4 + c] = window[12]; 
        }
    }

    __global__ void medianBlendKernel(unsigned char* currentFrame, unsigned char** allFrames, int numFrames, int width, int height, size_t step, bool isNegative) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        int idx = y * step + x * 4;
        int sumB = 0; 
        int sumG = 0; 
        int sumR = 0; 
        for (int j = 0; j < numFrames; ++j) {
            unsigned char* framePtr = allFrames[j];
            sumB += framePtr[idx];     
            sumG += framePtr[idx + 1]; 
            sumR += framePtr[idx + 2]; 
        }
        
        int valB = 1 + sumB;
        int valG = 1 + sumG;
        int valR = 1 + sumR;
        
        unsigned char newB = currentFrame[idx]     ^ (unsigned char)valB;
        unsigned char newG = currentFrame[idx + 1] ^ (unsigned char)valG;
        unsigned char newR = currentFrame[idx + 2] ^ (unsigned char)valR;
        
        currentFrame[idx]     = newG; 
        currentFrame[idx + 1] = newR; 
        currentFrame[idx + 2] = newB; 
        
        
        if (isNegative) {
            currentFrame[idx]     = 255 - currentFrame[idx];
            currentFrame[idx + 1] = 255 - currentFrame[idx + 1];
            currentFrame[idx + 2] = 255 - currentFrame[idx + 2];
        }
        
        
        currentFrame[idx + 3] = 255; 
    }

    __global__ void filterKernel(int filterIndex, unsigned char* data, int width, int height, size_t step, float alpha, bool isNegative) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x >= width || y >= height) return;
        
        switch(filterIndex) {
            case 0:
                applySelfAlphaBlend(x, y, data, width, height, step, alpha, isNegative);
                break;
            default:
                break;
        }
    }

    __global__ void squareBlockResizeVerticalKernel(unsigned char* currentFrame, unsigned char** allFrames, int numFrames, int width, int height, size_t step, int square_size, int collection_index) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        int idx = y * step + x * 4;
        unsigned char* historyFrame = allFrames[collection_index];
        currentFrame[idx]     = (unsigned char)((currentFrame[idx]     * 0.5f) + (historyFrame[idx]     * 0.5f));
        currentFrame[idx + 1] = (unsigned char)((currentFrame[idx + 1] * 0.5f) + (historyFrame[idx + 1] * 0.5f));
        currentFrame[idx + 2] = (unsigned char)((currentFrame[idx + 2] * 0.5f) + (historyFrame[idx + 2] * 0.5f));
        currentFrame[idx + 3] = 255;
    }

} 

extern "C" void launch_median_blur(unsigned char* data, int width, int height, size_t step) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    ac_gpu::medianBlur5x5Kernel<<<gridSize, blockSize>>>(data, width, height, step);
    cudaDeviceSynchronize();
}

extern "C" void launch_filter(int index, unsigned char* data, int width, int height, size_t step, float alpha, bool isNegative) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    ac_gpu::filterKernel<<<gridSize, blockSize>>>(index, data, width, height, step, alpha, isNegative);        
    cudaDeviceSynchronize(); 
}

extern "C" void launch_median_blend(unsigned char* currentFrame, unsigned char** devicePtrArray, int numFrames, int width, int height, size_t step, int div_value) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    ac_gpu::medianBlendKernel<<<gridSize, blockSize>>>(currentFrame, devicePtrArray, numFrames, width, height, step, (div_value != 0));
    cudaDeviceSynchronize();
}


extern "C" void launch_square_block_resize_vertical(unsigned char* currentFrame, unsigned char** devicePtrArray, int numFrames, int width, int height, size_t step, int square_size, int collection_index) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    ac_gpu::squareBlockResizeVerticalKernel<<<gridSize, blockSize>>>(currentFrame, devicePtrArray, numFrames, width, height, step, square_size, collection_index);
    cudaDeviceSynchronize();
}
