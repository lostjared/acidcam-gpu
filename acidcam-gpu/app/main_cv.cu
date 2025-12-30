#include<opencv2/opencv.hpp>
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

static const char* filter_names[] = {
    "SelfAlphaBlend", 
    "MedianBlend", 
    "MedianBlurBlend", 
    "SquareBlockResize", 
    "SelfAlphaScaleRefined",
    "StrangeGlitch",
    "MatrixOutline",
    "AuraTrails",
    "MirrorReverseColor",
    "ImageSquareShrink"
};

struct AnimationState {
    float alpha = 1.0f;
    int alpha_dir = 1;
    int square_offset = 0;
    int square_dir = 1;
    int square_speed = 2;
} gState;

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

void updateAndDraw(cv::Mat& frame, ac_gpu::DynamicFrameBuffer& buffer, 
                    unsigned char* d_workingBuffer, unsigned char** d_ptrList,
                    size_t workingPitch, ac_gpu::Filter* activeFilters, size_t filterCount) {
    
    if (gState.alpha_dir == 1) {
        gState.alpha += 0.01f;
        if (gState.alpha >= 1.0f) gState.alpha_dir = 0;
    } else {
        gState.alpha -= 0.01f;
        if (gState.alpha <= 0.1f) gState.alpha_dir = 1;
    }

    if (gState.square_dir == 1) {
        gState.square_offset += gState.square_speed;
        if (gState.square_offset > (frame.rows / 2) - 1) gState.square_dir = 0;
    } else {
        gState.square_offset -= gState.square_speed;
        if (gState.square_offset <= 1) gState.square_dir = 1;
    }

    
    //buffer.update(frame); 

    
    CHECK_CUDA(cudaMemcpy(d_ptrList, buffer.deviceFrames.data(), 
                          buffer.arraySize * sizeof(unsigned char*), 
                          cudaMemcpyHostToDevice));

    
    CHECK_CUDA(cudaMemcpy2D(d_workingBuffer, workingPitch,
                            buffer.deviceFrames[buffer.arraySize - 1], buffer.framePitch,
                            buffer.w * 4, buffer.h, cudaMemcpyDeviceToDevice));

    launch_filter(
        activeFilters, 
        filterCount, 
        d_workingBuffer, 
        d_ptrList, 
        buffer.arraySize, 
        buffer.w, 
        buffer.h, 
        workingPitch, 
        gState.alpha, 
        false, 
        gState.square_offset, 
        gState.square_offset, 
        gState.square_dir
    );
}

int main(int argc, char** argv) {
    Argz<std::string> argz(argc, argv);
    bool camera_mode = false;
    int camera_index = 0;
    int filter_index = 0;
    int dynamic_buffer = 10;
    unsigned char* d_workingBuffer = nullptr;
    size_t workingPitch = 0;
    unsigned char** d_ptrList = nullptr;
    std::string inputArg;
    std::string filtersArg;
    std::string bufferArg;
    std::string cameraArg;
    argz.addOptionSingleValue('i', "input file (short)");
    argz.addOptionDoubleValue(255, "input", "Input video file");
    argz.addOptionSingleValue('c', "camera index (short)");
    argz.addOptionDoubleValue(258, "camera", "Camera index (prefer this over --input)");
    argz.addOptionSingleValue('f', "filters list (short)");
    argz.addOptionDoubleValue(256, "filters", "Comma-separated filter indices (e.g. 0,1,2)");
    argz.addOptionSingleValue('b', "buffer size (short)");
    argz.addOptionDoubleValue(257, "buffer", "Dynamic buffer size (4-32)");
    argz.addOptionSingle('h', "help");
    try {
        Argument<std::string> a;
        std::vector<std::string> positional;
        int code = 0;
        while ((code = argz.proc(a)) != -1) {
            switch (code) {
                case 'h':
                    argz.help(std::cout);
                    return 0;
                case 'i':
                case 255:
                    inputArg = a.arg_value;
                    break;
                case 'c':
                case 258:
                    cameraArg = a.arg_value;
                    break;
                case 'f':
                case 256:
                    filtersArg = a.arg_value;
                    break;
                case 'b':
                case 257:
                    bufferArg = a.arg_value;
                    break;
            }
        }
    } catch (ArgException<std::string> &e) {
        std::cerr << "ac: Argument error: " << e.text() << std::endl;
        return -1;
    }

    if (filtersArg.empty() || bufferArg.empty() || (inputArg.empty() && cameraArg.empty())) {
        std::cerr << "ac: Usage: --input <file> or --camera <index> --filters <list> --buffer <size>\n";
        return -1;
    }

    if (!cameraArg.empty()) {
        if (!isNumeric(cameraArg)) {
            std::cerr << "ac: camera index must be numeric\n";
            return -1;
        }
        camera_mode = true;
        camera_index = std::stoi(cameraArg);
    } else if (!inputArg.empty() && isNumeric(inputArg)) {
        camera_mode = true;
        camera_index = std::stoi(inputArg);
    }

    std::vector<ac_gpu::Filter> vlist;

    std::string list = filtersArg.empty() ? argv[2] : filtersArg;
    if (list.find(',') != std::string::npos) {
        size_t start = 0;
        while (1) {
            size_t pos = list.find(',', start);
            std::string tok = (pos == std::string::npos) ? list.substr(start) : list.substr(start, pos - start);
            if (tok.empty()) {
                if (pos == std::string::npos) break;
                start = pos + 1;
                continue;
            }
            if (!isNumeric(tok)) {
                std::cerr << "ac: Error invalid filter_index\n";
                return -1;
            }
            int idx = std::stoi(tok);
            if (idx > ac_gpu::AC_FILTER_MAX || idx < 0) {
                std::cerr << "ac: Filter out of range..\n";
                return -3;
            }
            vlist.emplace_back(ac_gpu::Filter{idx, filter_names[idx]});
            if (pos == std::string::npos) break;
            start = pos + 1;
        }
        if (!vlist.empty()) filter_index = (int)vlist[0].index;
    } else {
        if (!isNumeric(list)) {
            std::cerr << "ac: Error invalid filter_index\n";
            return -1;
    
        }
        int idx = std::stoi(list);
        if (idx > ac_gpu::AC_FILTER_MAX-1 || idx < 0) {
            std::cerr << "ac: Filter out of range..\n";
            return -3;
        }
        filter_index = idx;
        vlist.emplace_back(ac_gpu::Filter{filter_index, filter_names[filter_index]});
    }
    if(!bufferArg.empty()) {
        if(!isNumeric(bufferArg)) {
            std::cerr << "ac: Requires value between 4-32 for sizes of dynamic array buffer.\n";
            return -2;
        }
        dynamic_buffer = std::stoi(bufferArg);
    }
    if(dynamic_buffer < 4 || dynamic_buffer > 32) {
        std::cerr << "ac: Requires value between 4-32 for sizes of dynamic array buffer.\n";
        return -2;  
    }
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    cv::VideoCapture cap;
    try {
        if(camera_mode == true) {
            cap.open(camera_index, cv::CAP_V4L2);
        } else {
            cap.open(inputArg.empty() ? argv[1] : inputArg);
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
        ac_gpu::DynamicFrameBuffer buffer(dynamic_buffer);
        CHECK_CUDA(cudaMalloc(&d_ptrList, buffer.arraySize * sizeof(unsigned char*)));
        cv::Mat rgba_out(height, width, CV_8UC4);
        std::cout << "ac: Video resolution: " << width << "x" << height << " @ " << fps << " fps" << std::endl;
        auto frame_duration = std::chrono::milliseconds((int)(1000.0 / fps));
        int current_filter = filter_index;
        int screenshot_count = 1;
        //int square_size = 4;
        //int square_dir = 1;
        //int collection_index = 0;
        //int index_dir = 1;
        cv::Mat frame;
        cv::namedWindow("filter", cv::WINDOW_NORMAL);
        cv::resizeWindow("filter", width, height);
        CHECK_CUDA(cudaMallocPitch(&d_workingBuffer, &workingPitch, width * 4, height));
        while (1) {
            auto start_time = std::chrono::steady_clock::now();
            if (!cap.read(frame)) break; 

            buffer.update(frame);
            
            if(current_filter == 2 || current_filter == 3) {
                int r = 3 + rand() % 7;  
                for(int i = 0; i < r; ++i)
                    cv::medianBlur(frame, frame, 3);
            }

            if (workingPitch != buffer.framePitch || width != buffer.w || height != buffer.h) {
                if (d_workingBuffer) CHECK_CUDA(cudaFree(d_workingBuffer));
                width = buffer.w;
                height = buffer.h;
                CHECK_CUDA(cudaMallocPitch(&d_workingBuffer, &workingPitch, width * 4, height));
                rgba_out = cv::Mat(height, width, CV_8UC4);  
            }
            
            updateAndDraw(  
                frame, 
                buffer, 
                d_workingBuffer, 
                d_ptrList, 
                workingPitch, 
                &vlist[0], 
                vlist.size()
            );
                        
            CHECK_CUDA(cudaMemcpy2D(rgba_out.data, rgba_out.step[0], d_workingBuffer, workingPitch, width * 4, height, cudaMemcpyDeviceToHost));
            cv::cvtColor(rgba_out, frame, cv::COLOR_RGBA2BGR);
            cv::imshow("filter", frame);
            int key = cv::waitKey(1);
            if (key == 27) break; 

            else if (key == 's' || key == 'S') { 
                auto now = std::chrono::system_clock::now();
                std::time_t t = std::chrono::system_clock::to_time_t(now);
                std::tm tm = *std::localtime(&t);
                char timebuf[32];
                std::strftime(timebuf, sizeof(timebuf), "%Y%m%d_%H%M%S", &tm);
                std::string out = "acidcam_gpu-" + std::string(timebuf) + "_" +
                                std::to_string(width) + "x" + std::to_string(height) +
                                "_" + std::to_string(screenshot_count++) + ".png";
                cv::imwrite(out, frame);
                std::cout << "ac: Saved screenshot: " << out << std::endl;
            } 
            else if (key == 82 || key == 0 || key == 65362) { 
                if (current_filter < ac_gpu::AC_FILTER_MAX - 1) {
                    current_filter++;
                    vlist.clear();
                    ac_gpu::Filter f {current_filter, filter_names[current_filter]};
                    vlist.push_back(f);
                    std::cout << "ac: Current filter: " << filter_names[current_filter] << " (" << current_filter << ")" << std::endl;
                }
            }
            else if (key == 84 || key == 1 || key == 65364) { 
                if (current_filter > 0) {
                    current_filter--;
                    vlist.clear();
                    ac_gpu::Filter f {current_filter, filter_names[current_filter]};
                    vlist.push_back(f);
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
    catch(ac_gpu::ACException &e) {
        std::cerr << e.why() << std::endl;
    } 
    catch(std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    cap.release(); 
    return 0;
}