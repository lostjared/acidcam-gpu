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

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

bool isNumeric(const std::string &text) {
    std::regex re("^-?\\d+(\\.\\d+)?$");
    return std::regex_match(text, re);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "ac: Usage: " << argv[0] << " camera_index start_filter dynmic_buffer_size" << std::endl;
        return -1;
    }
    bool camera_mode = false;
    int camera_index = 0;
    int filter_index = 0;
    int dynamic_buffer = 10;
    const int max_filter = 3;  

    if(isNumeric(argv[1])) {
        camera_mode = true;
        camera_index = std::stoi(argv[1]);
    }
    if(!isNumeric(argv[2])) {
        std::cerr << "ac: Error invalid filter_index\n";
        return -1;
    } else {
        filter_index = std::stoi(argv[2]);
    }
    if(filter_index > max_filter || filter_index < 0) {
        std::cerr << "ac: Filter out of range..\n";
        return -3;
    }
    if(!isNumeric(argv[3])) {
        std::cerr << "ac: Requires value between 4-32 for sizes of dynamic array buffer.\n";
        return -2;
    } else {
        dynamic_buffer = std::stoi(argv[3]);
        if(dynamic_buffer < 4 || dynamic_buffer > 32) {
            std::cerr << "ac: Requires value between 4-32 for sizes of dynamic array buffer.\n";
            return -2;  
        }
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
    std::cout << "ac: Video resolution: " << width << "x" << height << " @ " << fps << " fps" << std::endl;
    auto frame_duration = std::chrono::milliseconds((int)(1000.0 / fps));
    int current_filter = filter_index;

    const char* filter_names[] = {"SelfAlphaBlend", "MedianBlend", "MedianBlurBLend", "SquareBlockResize"};
    std::cout << "ac: Current filter: " << filter_names[current_filter] << " (" << current_filter << ")" << std::endl;
    int screenshot_count = 1;
    int square_size = 4;
    int square_dir = 1;
    int collection_index = 0;
    int index_dir = 1;
    {
        cv::Mat frame;
        cv::namedWindow("filter", cv::WINDOW_NORMAL);
        cv::resizeWindow("filter", 1920, 1080);
        ac_gpu::DynamicFrameBuffer buffer(dynamic_buffer); 
        unsigned char** d_ptrList; 
        CHECK_CUDA(cudaMalloc(&d_ptrList, buffer.arraySize * sizeof(unsigned char*)));
        cv::Mat rgba_out(height, width, CV_8UC4);
        unsigned char* d_workingBuffer = nullptr;
        size_t workingPitch = 0;
        CHECK_CUDA(cudaMallocPitch(&d_workingBuffer, &workingPitch, width * 4, height));
        float alpha = 1.0f;
        while (1) {
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
            if(current_filter == 2 || current_filter == 3) {
                int r = 3 + rand() % 7;  
                for(int i = 0; i < r; ++i)
                    cv::medianBlur(frame, frame, 3);
            }

            buffer.update(frame);
            
            if (workingPitch != buffer.framePitch || width != buffer.w || height != buffer.h) {
                if (d_workingBuffer) CHECK_CUDA(cudaFree(d_workingBuffer));
                width = buffer.w;
                height = buffer.h;
                CHECK_CUDA(cudaMallocPitch(&d_workingBuffer, &workingPitch, width * 4, height));
                rgba_out = cv::Mat(height, width, CV_8UC4);  // Resize output buffer too
            }
            
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
            CHECK_CUDA(cudaMemcpy2D(d_workingBuffer, workingPitch,
                         buffer.deviceFrames[buffer.arraySize - 1], buffer.framePitch,
                         buffer.w * 4, buffer.h, cudaMemcpyDeviceToDevice));
            
            CHECK_CUDA(cudaMemcpy(d_ptrList, buffer.deviceFrames.data(), 
                       buffer.arraySize * sizeof(unsigned char*), 
                       cudaMemcpyHostToDevice));

            launch_filter(current_filter, d_workingBuffer, d_ptrList,
                          buffer.arraySize, buffer.w, buffer.h, workingPitch,
                          alpha, false, square_size, collection_index, index_dir);
            
            CHECK_CUDA(cudaMemcpy2D(rgba_out.data, rgba_out.step[0], d_workingBuffer, workingPitch, width * 4, height, cudaMemcpyDeviceToHost));
            cv::cvtColor(rgba_out, frame, cv::COLOR_RGBA2BGR);
            cv::imshow("filter", frame);
            int key = cv::waitKey(1);
            if (key == 27) break; 
            else if (key == 's' || key == 'S') { 
                std::string out = "capture_" + std::to_string(screenshot_count++) + ".png";
                cv::imwrite(out, frame);
                std::cout << "ac: Saved screenshot: " << out << std::endl;
            } 
            else if (key == 82 || key == 0 || key == 65362) { 
                if (current_filter < max_filter) {
                    current_filter++;
                    std::cout << "ac: Current filter: " << filter_names[current_filter] << " (" << current_filter << ")" << std::endl;
                }
            }
            else if (key == 84 || key == 1 || key == 65364) { 
                if (current_filter > 0) {
                    current_filter--;
                    std::cout << "ac: Current filter: " << filter_names[current_filter] << " (" << current_filter << ")" << std::endl;
                }
            }

            auto end_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            if (elapsed < frame_duration) std::this_thread::sleep_for(frame_duration - elapsed);
        }

        CHECK_CUDA(cudaFree(d_ptrList));
        if (d_workingBuffer) CHECK_CUDA(cudaFree(d_workingBuffer));
    } 

    cap.release(); 
    return 0;
}