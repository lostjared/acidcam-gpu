#include<opencv2/opencv.hpp>
#include<iostream>
#include<chrono>
#include<thread>
#include<cuda_runtime.h>
#include<ac-gpu/ac-gpu.hpp>
#include<cstdlib>
#include<ctime>
#include<regex>
#include<string>

bool isNumeric(const std::string &text) {
    std::regex re("^-?\\d+(\\.\\d+)?$");
    return std::regex_match(text, re);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " camera_index start_filter" << std::endl;
        return -1;
    }
    bool camera_mode = false;
    int camera_index = 0;
    int filter_index = 0;
    if(isNumeric(argv[1])) {
        camera_mode = true;
        camera_index = std::stoi(argv[1]);
    }
    if(!isNumeric(argv[2])) {
        std::cerr << "Error invalid filter_index\n";
        return -1;
    } else {
        filter_index = std::stoi(argv[2]);
    }
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    cv::VideoCapture cap;
    if(camera_mode == true) {
        cap.open(camera_index, cv::CAP_V4L2);
    } else {
        cap.open(argv[1]);
    }
    if (!cap.isOpened()) return -1;
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 60);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Video resolution: " << width << "x" << height << " @ " << fps << " fps" << std::endl;
    auto frame_duration = std::chrono::milliseconds((int)(1000.0 / fps));
    int current_filter = filter_index;
    const int max_filter = 4;  
    const char* filter_names[] = {"SelfAlphaBlend", "MedianBlur", "MedianBlend", "MedianBlurBLend", "SquareBlockResize"};
    std::cout << "Current filter: " << filter_names[current_filter] << " (" << current_filter << ")" << std::endl;
    int screenshot_count = 1;
    int square_size = 4;
    int square_dir = 1;
    int collection_index = 0;
    int index_dir = 1;
    {
        cv::Mat frame;
        cv::namedWindow("filter", cv::WINDOW_NORMAL);
        cv::resizeWindow("filter", 1920, 1080);
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
            if (workingPitch != buffer.framePitch) {
                if (d_workingBuffer) cudaFree(d_workingBuffer);
                cudaMallocPitch(&d_workingBuffer, &workingPitch, buffer.w * 4, buffer.h);
            }
            cudaMemcpy(d_ptrList, buffer.deviceFrames.data(), buffer.arraySize * sizeof(unsigned char*), cudaMemcpyHostToDevice);
            if(index_dir == 1) {
                collection_index++;
                if(collection_index >= (buffer.arraySize - 1)) {
                    collection_index = buffer.arraySize - 1;
                    index_dir = 0;
                }
            } else {
                collection_index--;
                if(collection_index <= 0) {
                    collection_index = 0;
                    index_dir = 1;
                }
            }
            if(square_dir == 1) {
                square_size += 2;
                if(square_size >= 64) {
                    square_size = 64;
                    square_dir = 0;
                }
            } else {
                square_size -= 2;
                if(square_size <= 2) {
                    square_size = 2;
                    square_dir = 1;
                }
            }
            cudaMemcpy2D(d_workingBuffer, workingPitch,
                         buffer.deviceFrames[buffer.arraySize - 1], buffer.framePitch,
                         buffer.w * 4, buffer.h, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_ptrList, buffer.deviceFrames.data(), 
                       buffer.arraySize * sizeof(unsigned char*), 
                       cudaMemcpyHostToDevice);
            

            launch_filter(current_filter, d_workingBuffer, d_ptrList,
                          buffer.arraySize, buffer.w, buffer.h, workingPitch,
                          alpha, false, square_size, collection_index, index_dir);
            
            cudaMemcpy2D(rgba_out.data, rgba_out.step[0], d_workingBuffer, workingPitch, width * 4, height, cudaMemcpyDeviceToHost);
            cv::cvtColor(rgba_out, frame, cv::COLOR_RGBA2BGR);
            cv::imshow("filter", frame);
            int key = cv::waitKey(1);
            if (key == 27) break; 
            else if (key == 's' || key == 'S') { 
                cv::imwrite("capture_" + std::to_string(screenshot_count++) + ".png", frame);
                std::cout << "Saved screenshot." << std::endl;
            } 
            else if (key == 82 || key == 0 || key == 65362) { 
                if (current_filter < max_filter) {
                    current_filter++;
                    std::cout << "Current filter: " << filter_names[current_filter] << " (" << current_filter << ")" << std::endl;
                }
            }
            else if (key == 84 || key == 1 || key == 65364) { 
                if (current_filter > 0) {
                    current_filter--;
                    std::cout << "Current filter: " << filter_names[current_filter] << " (" << current_filter << ")" << std::endl;
                }
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