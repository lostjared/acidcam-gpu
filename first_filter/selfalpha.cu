#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>   
#include <thread>  


__global__ void selfAlphaBlendKernel(unsigned char* data, int width, int height, size_t step, float alpha, bool isNegative) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * step + x * 4; // BGRA

        unsigned char b = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char r = data[idx + 2];

        data[idx]     = (unsigned char)(b + (unsigned char)(b * alpha)); 
        data[idx + 1] = (unsigned char)(g + (unsigned char)(g * alpha)); 
        data[idx + 2] = (unsigned char)(r + (unsigned char)(r * alpha)); 

        // 2. swapColors (Blue and Red swap)
        unsigned char tempB = data[idx];
        data[idx] = data[idx + 2];
        data[idx + 2] = tempB;

        // 3. invert
        if (isNegative) {
            data[idx]     = 255 - data[idx];
            data[idx + 1] = 255 - data[idx + 1];
            data[idx + 2] = 255 - data[idx + 2];
        }
        data[idx + 3] = 255; 
    }
}

int main(int argc, char **argv) {
    if(argc != 2) return 1;

    cv::Mat h_img = cv::imread(argv[1]);
    if (h_img.empty()) return -1;
    cv::cvtColor(h_img, h_img, cv::COLOR_BGR2BGRA);

    int width = h_img.cols, height = h_img.rows;
    size_t size = h_img.total() * h_img.elemSize(), step = h_img.step;

    unsigned char *d_img;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, h_img.data, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16), grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Timing Variables
    float alpha = 0.1f, alpha_inc = 0.05f;
    int direction = 1;
    bool isNegative = true;

    // Define 60 FPS (16.66ms per frame)
    const std::chrono::milliseconds frame_duration(1000 / 60);

    while(1) {
        // Mark start of frame
        auto start_time = std::chrono::steady_clock::now();

        // GPU Processing
        selfAlphaBlendKernel<<<grid, block>>>(d_img, width, height, step, alpha, isNegative);
        
        // Alpha oscillation logic
        if(direction == 1) {
            alpha += alpha_inc;
            if(alpha > 10.0f) { alpha = 10.0f; direction = 2; }
        } else {
            alpha -= alpha_inc;
            if(alpha <= 0.1f) { alpha = 0.1f; direction = 1; }
        }

        // cudaMemcpy is a "blocking" call, so it waits for the kernel to finish
        cudaMemcpy(h_img.data, d_img, size, cudaMemcpyDeviceToHost);
        cv::imshow("Consistent 60 FPS", h_img);

        // Calculate how long the work took
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Sleep if we finished the work faster than 16.6ms
        if (elapsed < frame_duration) {
            std::this_thread::sleep_for(frame_duration - elapsed);
        }

        if(cv::waitKey(1) == 27) break; 
    }

    cudaFree(d_img);
    return 0;
}
