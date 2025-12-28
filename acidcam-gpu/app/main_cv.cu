#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <ac-gpu/ac-gpu.hpp>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    bool isNegative = false; 

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) return -1;

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto frame_duration = std::chrono::milliseconds((int)(1000.0 / fps));

    int current_filter = 0;
    int max_filter = 1;
    int screenshot_count = 1;


    {
        cv::Mat frame;
        cv::namedWindow("filter", cv::WINDOW_NORMAL);
        cv::resizeWindow("filter", 1280, 720);
        ac_gpu::DynamicFrameBuffer buffer(8); 
        unsigned char** d_ptrList; 
        cudaMalloc(&d_ptrList, buffer.arraySize * sizeof(unsigned char*));
        cv::Mat rgba_out(height, width, CV_8UC4);
        unsigned char* d_workingBuffer = nullptr;
        size_t workingPitch = 0;
        cudaMallocPitch(&d_workingBuffer, &workingPitch, width * 4, height);
        
        float alpha = 1.0f;
        while (true) {
            auto start_time = std::chrono::steady_clock::now();
        
            static bool dir = true;
            if(dir == true) {
                alpha += 0.1f;
                if(alpha >= 6.0f) {
                    alpha = 6.0f;
                    dir = false;
                }
            } else {
                alpha -= 0.1f;
                if(alpha <= 1.0f) {
                    alpha = 1.0f;
                    dir = true;
                }
            }
    
            if (!cap.read(frame)) break; 
            buffer.update(frame);
            
            // Reallocate working buffer if size changed
            if (workingPitch != buffer.framePitch) {
                if (d_workingBuffer) cudaFree(d_workingBuffer);
                cudaMallocPitch(&d_workingBuffer, &workingPitch, buffer.w * 4, buffer.h);
            }
            
            // Copy newest frame to working buffer for processing
            cudaMemcpy2D(d_workingBuffer, workingPitch,
                         buffer.deviceFrames[buffer.arraySize - 1], buffer.framePitch,
                         buffer.w * 4, buffer.h, cudaMemcpyDeviceToDevice);
            
            // Apply median blur random number of times (3-9) like CPU version
            // Blur the WORKING buffer, not the collection buffer
            int r = 3 + std::rand() % 7;
            for (int i = 0; i < r; ++i) {
                launch_median_blur(d_workingBuffer, 
                       buffer.w, buffer.h, workingPitch);
            }
            
            cudaMemcpy(d_ptrList, buffer.deviceFrames.data(), 
            buffer.arraySize * sizeof(unsigned char*), 
            cudaMemcpyHostToDevice);

            // Blend writes to working buffer, collection stays clean
            launch_median_blend(d_workingBuffer, 
                    d_ptrList, 
                    buffer.arraySize, 
                    buffer.w, buffer.h, 
                    workingPitch, 
                    isNegative);
            
            // Copy result from working buffer to output
            cudaMemcpy2D(rgba_out.data, rgba_out.step[0], 
             d_workingBuffer, workingPitch, 
             width * 4, height, 
             cudaMemcpyDeviceToHost);

            cv::cvtColor(rgba_out, frame, cv::COLOR_RGBA2BGR);
            cv::imshow("filter", frame);

            int key = cv::waitKey(1);
            if (key == 27) break; 
            else if (key == 's' || key == 'S') { 
                cv::imwrite("capture_" + std::to_string(screenshot_count++) + ".png", frame);
                std::cout << "Saved screenshot." << std::endl;
            } 
            else if((key == 'w' || key == 'W') && current_filter < max_filter) {
                current_filter++;
            }
            else if((key == 'e' || key == 'E') && current_filter > 0) {
                current_filter--;
            }

            auto end_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            if (elapsed < frame_duration) std::this_thread::sleep_for(frame_duration - elapsed);
        }

        cudaFree(d_ptrList);
        if (d_workingBuffer) cudaFree(d_workingBuffer);
    } 

    cap.release(); 
    return 0;
}