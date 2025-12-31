#include<opencv2/opencv.hpp>
#include<opencv2/core/cuda.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<iostream>
#include<chrono>
#include<thread>
#include<cuda_runtime.h>
#include<ac-gpu/ac-gpu.hpp>
#include<ac-gpu/argz.hpp>
#include<cstdlib>
#include<ctime>
#include<regex>
#include<string>
#include<vector>
#include<mxwrite.hpp>


struct AnimationState {
    float alpha = 1.0f;
    int alpha_dir = 1;
    int square_offset = 0;
    int square_dir = 1;
    int square_dir_size = 1;
    int square_speed = 2;
    int square_size = 0;
} gState;


bool isNumeric(const std::string &text) {
    std::regex re("^-?\\d+(\\.\\d+)?$");
    return std::regex_match(text, re);
}

cv::Size extractResolution(const std::string &text) {
    auto pos = text.find("x");
    if(pos == std::string::npos || text.empty()) {
        throw ac_gpu::ACException("Error could not extract string or variable size");
    }
    std::string left = text.substr(0, pos);
    std::string right = text.substr(pos+1);
    return cv::Size(std::stoi(left), std::stoi(right));
}


void updateAndDraw(cv::Mat& frame, ac_gpu::DynamicFrameBuffer& buffer, 
                    cv::cuda::GpuMat& gpuWorkingBuffer, unsigned char** d_ptrList,
                    ac_gpu::Filter* activeFilters, size_t filterCount, 
                    ac_gpu::Filter** d_list_ptr, bool& changed) {
    
    
    if (gState.alpha_dir == 1) {
        gState.alpha += 0.01f;
        if (gState.alpha >= 3.0f) gState.alpha_dir = 0;
    } else {
        gState.alpha -= 0.01f;
        if (gState.alpha <= 1.0f) gState.alpha_dir = 1;
    }

    static int current_frame_index = 0;
    static int index_dir = 1;
    if(index_dir == 1) {
        current_frame_index++;
        if(current_frame_index >= (buffer.arraySize - 1)) {
            current_frame_index = buffer.arraySize - 1;
            index_dir = 0;
        }
    } else {
        current_frame_index--;
        if(current_frame_index <= 0) {
            current_frame_index = 0;
            index_dir = 1;
        }
    }

    if(gState.square_dir_size == 1) {
        gState.square_size += 2;
         if(gState.square_size >= 64) {
               gState.square_size = 64;
                gState.square_dir_size = 0;
        }
    } else {
        gState.square_size -= 2;
        if(gState.square_size <= 2) {
            gState.square_size = 2;
            gState.square_dir_size = 1;
        }
    }
    
    
    CHECK_CUDA(cudaMemcpy(d_ptrList, buffer.deviceFrames.data(), 
                          buffer.arraySize * sizeof(unsigned char*), 
                          cudaMemcpyHostToDevice));

    
    CHECK_CUDA(cudaMemcpy2D(gpuWorkingBuffer.ptr<unsigned char>(), gpuWorkingBuffer.step,
                            buffer.deviceFrames[buffer.arraySize - 1], buffer.framePitch,
                            buffer.w * 4, buffer.h, cudaMemcpyDeviceToDevice));
    
    launch_filter(
        activeFilters, 
        filterCount, 
        gpuWorkingBuffer.ptr<unsigned char>(), 
        d_ptrList, 
        buffer.arraySize, 
        gpuWorkingBuffer.cols, 
        gpuWorkingBuffer.rows, 
        gpuWorkingBuffer.step, 
        gState.alpha, 
        false, 
        gState.square_size,
        current_frame_index, 
        index_dir,
        d_list_ptr, 
        changed     
    );
}


int main(int argc, char** argv) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    std::cout << "Acid Cam GPU Demo" << std::endl;
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "OpenCV Cuda Support not found." << std::endl;
        std::cerr << "Reason: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "Check: Are NVIDIA drivers installed? Is the GPU seated?" << std::endl;
        return -1; 
    } else {
        std::cout << "🚀 GPU Acceleration Active: " << device_count << " device(s) found." << std::endl;
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    }
    Argz<std::string> argz(argc, argv);
    bool camera_mode = false;
    int camera_index = 0, filter_index = 0, dynamic_buffer = 10;
    unsigned char** d_ptrList = nullptr;
    ac_gpu::Filter* d_filterList = nullptr;
    cv::Size vres(1920, 1080), cres (1920, 1080);
    bool filtersChanged = true; 
    std::string inputArg, filtersArg, bufferArg, cameraArg, output_filename, tally;
    std::string output_crf= "23";
    double output_fps = 60.0;
    int tick_count = 1;
    argz.addOptionSingleValue('i', "input").addOptionDoubleValue(255, "input", "Input video")
    .addOptionSingleValue('c', "camera").addOptionDoubleValue(258, "camera", "Camera ID")
    .addOptionSingleValue('f', "filters").addOptionDoubleValue(256, "filters", "Filter IDs")
    .addOptionSingleValue('b', "buffer").addOptionDoubleValue(257, "buffer", "Buffer 4-32")
    .addOptionSingleValue('r', "resolution").addOptionDoubleValue(260, "resolution", "Window size")
    .addOptionDoubleValue(289, "camera-res", "Camera size")
    .addOptionDoubleValue(290, "output", "Filename")
    .addOptionDoubleValue(293, "speed", "Tick speed")
    .addOptionDoubleValue(291, "crf", "CRF")
    .addOptionDoubleValue(292, "fps", "FPS")
    .addOptionSingle('h', "help");
    try {
        Argument<std::string> a;
        int code = 0;
        while ((code = argz.proc(a)) != -1) {
            switch (code) {
                case 'h': argz.help(std::cout); return 0;
                case 'i': case 255: inputArg = a.arg_value; break;
                case 'c': case 258: cameraArg = a.arg_value; break;
                case 'f': case 256: filtersArg = a.arg_value; break;
                case 'b': case 257: bufferArg = a.arg_value; break;
                case 'r': case 260: vres = extractResolution(a.arg_value); break;
                case 289: cres = extractResolution(a.arg_value); break;
                case 290: output_filename = a.arg_value; break;
                case 291: output_crf = a.arg_value; break;
                case 292: output_fps = std::stod(a.arg_value); break;
                case 293: tick_count = std::stoi(a.arg_value); break;
            }
        }
    }  
    catch(ArgException<std::string> &e) {
        std::cerr << e.text() << "\n";
        return -1;
    }
    
    if (!cameraArg.empty()) { camera_mode = true; camera_index = std::stoi(cameraArg); }
    std::vector<ac_gpu::Filter> vlist;
    std::string list = filtersArg;
    if (list.find(',') != std::string::npos) {
        size_t start = 0;
        while (1) {
            size_t pos = list.find(',', start);
            std::string tok = (pos == std::string::npos) ? list.substr(start) : list.substr(start, pos - start);
            if (!tok.empty()) vlist.emplace_back(ac_gpu::Filter{std::stoi(tok), ac_gpu::filters[std::stoi(tok)].name});
            if (pos == std::string::npos) break;
            start = pos + 1;
        }
        filter_index = vlist[0].index;
    } else {
        filter_index = std::stoi(list);
        vlist.emplace_back(ac_gpu::Filter{filter_index, ac_gpu::filters[filter_index].name});
    }

    dynamic_buffer = std::stoi(bufferArg);

    cv::VideoCapture cap;
    Writer writer;
    int frameCount = 0, screenshot_count = 1;
    double currentFPS = 0.0;
    auto lastFPSUpdate = std::chrono::steady_clock::now();

    try {
        if(camera_mode) cap.open(camera_index, cv::CAP_V4L2);
        else cap.open(inputArg);
        if (!cap.isOpened()) return -1;
        if(camera_mode == true) {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, cres.width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, cres.height);
            cap.set(cv::CAP_PROP_FPS, output_fps);
        }
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;

        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        std::cout << "ac: Resolution: " << width << "x" << height << " @ " << fps << std::endl;

        ac_gpu::DynamicFrameBuffer buffer(dynamic_buffer);
        CHECK_CUDA(cudaMalloc(&d_ptrList, buffer.arraySize * sizeof(unsigned char*)));

        cv::cuda::GpuMat gpuWorkingBuffer(height, width, CV_8UC4);
        cv::cuda::GpuMat gpuDisplayBuffer(height, width, CV_8UC4);
        cv::Mat frame, rgba_out(height, width, CV_8UC4);
        cv::namedWindow("filter", cv::WINDOW_NORMAL);
        cv::resizeWindow("filter", vres.width, vres.height);
        cv::moveWindow("filter", 0, 0);

        if(!output_filename.empty()) 
            writer.open_ts(output_filename, width, height, fps, output_crf.c_str());

        auto frame_duration = std::chrono::milliseconds((int)(1000.0 / fps));

        while (1) {
            auto start_time = std::chrono::steady_clock::now();
            if (!cap.read(frame)) break; 

            buffer.update(frame);

            if (width != buffer.w || height != buffer.h) {
                width = buffer.w; height = buffer.h;
                gpuWorkingBuffer.create(height, width, CV_8UC4);
                gpuDisplayBuffer.create(height, width, CV_8UC4);
                rgba_out = cv::Mat(height, width, CV_8UC4);
            }
            
            updateAndDraw(frame, buffer, gpuWorkingBuffer, d_ptrList, &vlist[0], vlist.size(), &d_filterList, filtersChanged);

            cv::cuda::cvtColor(gpuWorkingBuffer, gpuDisplayBuffer, cv::COLOR_RGBA2BGRA);
            gpuDisplayBuffer.download(frame);

            if(!output_filename.empty()) {
                gpuWorkingBuffer.download(rgba_out);
                writer.write_ts(rgba_out.ptr());
            }

            auto currentTime = std::chrono::steady_clock::now();
            auto elapsedx = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastFPSUpdate);
            if (elapsedx.count() >= 1000) {
                currentFPS = (frameCount * 1000.0) / elapsedx.count();
                frameCount = 0;
                lastFPSUpdate = currentTime;
            }
            frameCount++;

            static int tick = 0;
            if ((tick_count == 1) || (++tick % tick_count == 0)) {
                cv::putText(frame, "FPS: " + std::to_string((int)currentFPS), cv::Point(20, 50), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                cv::imshow("filter", frame);
            }

            int key = cv::waitKey(1);
            if (key == 27) break; 
            else if (key == 's' || key == 'S') {
                std::string out = "acidcam-gpu_" + std::to_string(screenshot_count++) + ".png";
                cv::imwrite(out, frame);
                std::cout << "Saved: " << out << std::endl;
            }
            else if (key == 't') {
                tally += std::to_string(filter_index) + ", ";
            }
            else if (key == 82 || key == 65362) { 
                if (filter_index < ac_gpu::AC_FILTER_MAX - 1) {
                    filter_index++;
                    vlist.clear();
                    vlist.emplace_back(ac_gpu::Filter{filter_index, ac_gpu::filters[filter_index].name});
                    filtersChanged = true;
                    std::cout << "Filter: " << ac_gpu::filters[filter_index].name << " (" << filter_index << ")" << std::endl;
                }
            }
            else if (key == 84 || key == 65364) { 
                if (filter_index > 0) {
                    filter_index--;
                    vlist.clear();
                    vlist.emplace_back(ac_gpu::Filter{filter_index, ac_gpu::filters[filter_index].name});
                    filtersChanged = true;
                    std::cout << "Filter: " << ac_gpu::filters[filter_index].name << " (" << filter_index << ")" << std::endl;
                }
            }

            if(!camera_mode && tick == 1) {
                auto end_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                if (elapsed < frame_duration) std::this_thread::sleep_for(frame_duration - elapsed);
            } 
        } 

        if (d_filterList)
            CHECK_CUDA(cudaFree(d_filterList));
     
        CHECK_CUDA(cudaFree(d_ptrList));
      
        if(!tally.empty()) 
            std::cout << "Total tally: " << tally << "\n";

        if(!output_filename.empty())
            std::cout << "Wrote: " << output_filename << " " << writer.get_frame_count() << " frames " << static_cast<double>(writer.get_frame_count())/fps << " seconds" << std::endl;
    }
    catch(std::exception &e) { 
        std::cerr << e.what() << std::endl; 
    }
    if(!output_filename.empty()) writer.close();
    cap.release(); 
    return 0;
}