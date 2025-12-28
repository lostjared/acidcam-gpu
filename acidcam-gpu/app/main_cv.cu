#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include<ac-gpu/ac-gpu.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto frame_duration = std::chrono::milliseconds((int)(1000.0 / fps));
    cv::Mat frame;
    unsigned char* d_data; 
    size_t size = width * height * 4; 
    cudaMalloc(&d_data, size);
    cv::namedWindow("filter", cv::WINDOW_NORMAL);
    cv::resizeWindow("filter", 1280, 720);
    float alpha = 1.0f;
    while (1) {
        auto start_time = std::chrono::steady_clock::now();
        cap >> frame;
        if (frame.empty()) break;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);    
        cudaMemcpy(d_data, frame.data, size, cudaMemcpyHostToDevice);
        static bool dir = true;
        if(dir == true) {
            alpha += 0.1f;
            if(alpha > 10.0f)
                dir = false;
        } else {
            alpha -= 0.1f;
            if(alpha <= 10.0f) 
                dir = true;
        }
        launch_filter(0, d_data, width, height, width * 4, alpha, false);
        cudaDeviceSynchronize();
        cudaMemcpy(frame.data, d_data, size, cudaMemcpyDeviceToHost);
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);
        cv::imshow("filter", frame);
        if (cv::waitKey(1) == 27) break; 
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (elapsed < frame_duration) {
            std::this_thread::sleep_for(frame_duration - elapsed);
        }
    }
    cudaFree(d_data);
    return 0;
}