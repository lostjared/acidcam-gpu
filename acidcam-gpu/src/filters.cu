#include "ac-gpu/ac-gpu.hpp"
#include <cuda_runtime.h>
#include <string>

namespace ac_gpu {
    
    Filter filters[] = { 
        {0, "SelfAlphaBlend"}, 
        {1, "MedianBlend"}, 
        {2, "MedianBlurBlend"},
        {3, "SquareBlockResize"} 
    };

    struct FilterParams {
        float alpha;
        bool isNegative;
        int numFrames;
        int square_size;
        int start_index;
        int start_dir;
    };

    __device__ void setAlpha(unsigned char* data, int idx, bool isNegative) {
        if (isNegative) {
            data[idx]     = 255 - data[idx];
            data[idx + 1] = 255 - data[idx + 1];
            data[idx + 2] = 255 - data[idx + 2];
        }
        data[idx + 3] = 255;
    }

    __device__ void processSelfAlphaBlend(int x, int y, unsigned char* data, size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];
        
        data[idx]     = (unsigned char)(b + (b * params.alpha));
        data[idx + 1] = (unsigned char)(g + (g * params.alpha));
        data[idx + 2] = (unsigned char)(r + (r * params.alpha));
        
        setAlpha(data, idx, params.isNegative);
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

    __device__ void processMedianBlur(int x, int y, unsigned char* data, int width, int height, size_t step) {
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

    __device__ void processMedianBlend(int x, int y, unsigned char* currentFrame, unsigned char** allFrames,
                                        size_t step, const FilterParams& params) {
        int idx = y * step + x * 4;
        int sumB = 0, sumG = 0, sumR = 0;
        
        for (int j = 0; j < params.numFrames; ++j) {
            unsigned char* framePtr = allFrames[j];
            if (framePtr == nullptr) continue; 
            sumB += framePtr[idx];
            sumG += framePtr[idx + 1];
            sumR += framePtr[idx + 2];
        }
        
        unsigned char newB = currentFrame[idx]     ^ (unsigned char)(1 + sumB);
        unsigned char newG = currentFrame[idx + 1] ^ (unsigned char)(1 + sumG);
        unsigned char newR = currentFrame[idx + 2] ^ (unsigned char)(1 + sumR);
        
        currentFrame[idx]     = newG;
        currentFrame[idx + 1] = newR;
        currentFrame[idx + 2] = newB;
        
        setAlpha(currentFrame, idx, params.isNegative);
    }

    __device__ void processSquareBlockResize(int x, int y, unsigned char* currentFrame, unsigned char** allFrames,
                                              size_t step, const FilterParams& params) {
        int block_row = y / params.square_size;
        int period = 2 * (params.numFrames - 1);
        if (period <= 0) period = 1;
        
        int start_pos = (params.start_dir == 1) ? params.start_index : (2 * (params.numFrames - 1)) - params.start_index;
        int pos = (start_pos + block_row) % period;
        int frame_index = (pos < params.numFrames) ? pos : period - pos;
        
        if (frame_index < 0) frame_index = 0;
        if (frame_index >= params.numFrames) frame_index = params.numFrames - 1;
        
        int idx = y * step + x * 4;
        unsigned char* historyFrame = allFrames[frame_index];
        
        if (historyFrame == nullptr) return;
        
        currentFrame[idx]     = (unsigned char)(0.5f * currentFrame[idx]     + 0.5f * historyFrame[idx]);
        currentFrame[idx + 1] = (unsigned char)(0.5f * currentFrame[idx + 1] + 0.5f * historyFrame[idx + 1]);
        currentFrame[idx + 2] = (unsigned char)(0.5f * currentFrame[idx + 2] + 0.5f * historyFrame[idx + 2]);
        currentFrame[idx + 3] = 255;
    }


    __global__ void unifiedFilterKernel(Filter *filters, size_t count, unsigned char* data, unsigned char** allFrames,
                                         int width, int height, size_t step, FilterParams params) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x >= width || y >= height) return;
        
        for(int i = 0;  i < count; ++i) {
                switch (filters[i].index) {
                    case 0:
                        processSelfAlphaBlend(x, y, data, step, params);
                        break;
                    case 1: 
                        processMedianBlend(x, y, data, allFrames, step, params);
                        break;
                    case 2: 
                        processMedianBlend(x, y, data, allFrames, step, params);
                        break;
                    case 3:
                        processSquareBlockResize(x, y, data, allFrames, step, params);
                        break;
                }
        }
    }
}

extern "C" void launch_filter(ac_gpu::Filter *f, size_t c, unsigned char* data, unsigned char** allFrames,
                               int numFrames, int width, int height, size_t step,
                               float alpha, bool isNegative, int square_size,
                               int start_index, int start_dir) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    ac_gpu::FilterParams params;
    params.alpha = alpha;
    params.isNegative = isNegative;
    params.numFrames = numFrames;
    params.square_size = square_size;
    params.start_index = start_index;
    params.start_dir = start_dir;
    ac_gpu::unifiedFilterKernel<<<gridSize, blockSize>>>(f, c, data, allFrames, width, height, step, params);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}
