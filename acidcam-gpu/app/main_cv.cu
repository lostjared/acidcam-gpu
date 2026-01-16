#define CURRENT_VERSION "v1.0"
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
#include<format>
#include<filesystem>
#include<fstream>
#include<unistd.h>
#include<fcntl.h>
#include<signal.h>

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
                    ac_gpu::GPUFilter** d_list_ptr, bool& changed) {
    
    
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
    
    
   CHECK_CUDA(cudaMemcpy(d_ptrList, buffer.getDeviceFramePointers(), 
                          buffer.arraySize * sizeof(unsigned char*), 
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy2D(gpuWorkingBuffer.data, gpuWorkingBuffer.step,
                            buffer.deviceFrames.back().data, buffer.framePitch,
                            buffer.w * 4, buffer.h, cudaMemcpyDeviceToDevice));
    
    launch_filter(
        activeFilters, 
        filterCount, 
        gpuWorkingBuffer.data,
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

void checkDevices() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "OpenCV Cuda Support not found." << std::endl;
        std::cerr << "Reason: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "Check: Are NVIDIA drivers installed? Is the GPU seated?" << std::endl;
        exit(EXIT_FAILURE);
    } else {
        std::cout << "🚀 GPU Acceleration Active: " << device_count << " device(s) found." << std::endl;
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    }
}

void listGraphicsCards() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return;
    }
    
    std::cout << "Available Graphics Cards:" << std::endl;
    for (int i = 0; i < device_count; ++i) {
        cv::cuda::DeviceInfo device(i);
        std::cout << "  [" << i << "] " << device.name() 
                  << " - Total Memory: " << (device.totalMemory() / (1024*1024)) << " MB" << std::endl;
    }
}

std::string getCameraName(int device_index) {
    std::string sysfs_path = "/sys/class/video4linux/video" + std::to_string(device_index) + "/name";
    std::ifstream file(sysfs_path);
    if (file.is_open()) {
        std::string name;
        if (std::getline(file, name)) {
            name.erase(name.find_last_not_of(" \n\r\t") + 1);
            file.close();
            return name;
        }
        file.close();
    }
    return "Unknown Camera";
}

void listCameras() {
    std::cout << "Available Cameras:" << std::endl;
    int camera_count = 0;
    int stderr_backup = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDERR_FILENO);
    close(devnull);
    
    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
            dup2(stderr_backup, STDERR_FILENO);
            std::string camera_name = getCameraName(i);
            std::cout << "  [" << i << "] " << camera_name << std::endl;
            camera_count++;
            cap.release();
            devnull = open("/dev/null", O_WRONLY);
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
    }   
    dup2(stderr_backup, STDERR_FILENO);
    close(stderr_backup);
    if (camera_count == 0) {
        std::cout << "  No cameras found." << std::endl;
    }
}

class Interrupt {};

void signalHandler(int signum) {
    throw Interrupt();
}

int main(int argc, char** argv) {
    std::cout << "acidcam-gpu-cli " << CURRENT_VERSION << std::endl;
    std::cout << "https://lostsidedead.biz" << std::endl;
    std::cout.flush();
    checkDevices();
    auto cuda_device = cv::cuda::getDevice();
    cv::cuda::DeviceInfo device(cuda_device);
    std::ostringstream stream;
    stream << device.name() << " Total Memory: "<< device.totalMemory();
    std::string device_text = stream.str();
    Argz<std::string> argz(argc, argv);
    bool camera_mode = false;
    int camera_index = 0, filter_index = 0, dynamic_buffer = 10;
    unsigned char** d_ptrList = nullptr;
    ac_gpu::GPUFilter* d_filterList = nullptr;
    cv::Size vres(0,0), cres (1920, 1080);
    bool filtersChanged = true; 
    std::string inputArg, filtersArg, bufferArg, cameraArg, output_filename, tally;
    std::string output_crf= "23";
    double output_fps = 60.0;
    int tick_count = 1;
    size_t time_over = 0;
    bool show_hud = true;
    bool repeat = false;
    size_t start_pos = 0;
    size_t jump_pos = 0;
    bool expose = false;
    bool silent = false;
    double fps = 0.0;
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
    .addOptionDoubleValue(294, "time", "How many seconds to record")
    .addOptionDoubleValue(301, "device", "Select Cuda Device")
    .addOptionDouble(304, "repeat", "repeat video")
    .addOptionDoubleValue(305, "skip", "Skip to frame in video")
    .addOptionDoubleValue(306, "jump", "Jump to second in video")
    .addOptionDouble(302, "list", "List all devices")
    .addOptionDouble(300, "hide", "hide HUD")
    .addOptionDouble(307, "exposure", "Disable Auto Exposurre")
    .addOptionDouble(308, "list-devices", "List graphics cards and cameras")
    .addOptionDouble(320, "list", "List all filters")
    .addOptionSingle('h', "help")
    .addOptionDouble(400, "silent", "Supress video shown, run in command line mode");
    try {
        Argument<std::string> a;
        int code = 0;
        while ((code = argz.proc(a)) != -1) {
            switch (code) {
                case 320:
                    for(size_t i = 0; i < ac_gpu::AC_FILTER_MAX; ++i) {
                        std::cout << ac_gpu::filters[i].name << ": " << ac_gpu::filters[i].index << std::endl;
                    }
                    exit(EXIT_SUCCESS);
                break;
                case 'h': argz.help(std::cout); return 0;
                case 'i': case 255: 
                inputArg = a.arg_value;
                if(!std::filesystem::exists(inputArg)) {
                    std::cerr << "ac: File: " << inputArg << " does not exist." << std::endl;
                    exit(EXIT_FAILURE);
                }
                if(std::filesystem::is_directory(inputArg)) {
                    std::cerr << "ac: Argument passed is directory." << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
                case 307: expose = true; break;
                case 308:
                    listGraphicsCards();
                    std::cout << std::endl;
                    listCameras();
                    exit(EXIT_SUCCESS);
                break;
                case 400:
                    silent = true;
                break;
                case 'c': case 258: cameraArg = a.arg_value; break;
                case 'f': case 256: filtersArg = a.arg_value; break;
                case 'b': case 257: bufferArg = a.arg_value; break;
                case 'r': case 260: vres = extractResolution(a.arg_value); break;
                case 289: cres = extractResolution(a.arg_value); break;
                case 290: output_filename = a.arg_value; break;
                case 291: output_crf = a.arg_value; break;
                case 292: output_fps = std::stod(a.arg_value); break;
                case 293:
                    if(!isNumeric(a.arg_value)) {
                        std::cerr << "ac: Error speed must be a numeric value.\n";
                        exit(EXIT_FAILURE);
                    }
                    tick_count = std::stoi(a.arg_value); 
                    if(tick_count <= 0) {
                        std::cout << "ac: Error speed must be greater than zero\n";
                        exit(EXIT_FAILURE);
                    }
                    break;
                case 294: 
                    if(!isNumeric(a.arg_value)) {
                        std::cerr << "ac: Time must be a numeric value.\n";
                        exit(EXIT_FAILURE);
                    }
                    time_over = std::stoi(a.arg_value); 
                    if(time_over <= 0) {
                        std::cerr << "ac: Time must be greater than zero\n";
                        exit(EXIT_FAILURE);
                    }
                    break;
                case 300: show_hud = false; break;
                case 301:
                    if(isNumeric(a.arg_value)) {
                        cv::cuda::setDevice(std::stoi(a.arg_value));
                    } else {
                        std::cout << "ac: Device number must be an integer." << std::endl;
                    }
                break;
                case 302:
                    exit(EXIT_SUCCESS);
                    break;
                case 304:
                    repeat = true;
                    break;
                case 305:
                    start_pos = std::stoi(a.arg_value);
                break;
                case 306:
                    jump_pos = std::stoi(a.arg_value);
                break;
            }
        }
    }  
    catch(ArgException<std::string> &e) {
        std::cerr << e.text() << "\n";
        return -1;
    }
    
    if (!cameraArg.empty()) { 
        camera_mode = true; 
        if(isNumeric(cameraArg))
            camera_index = std::stoi(cameraArg); 
        else {
            std::cerr << "ac: Camera argumnet must be a valid integer." << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    if(filtersArg.empty()){
        std::cerr << "ac: Error requires filters argument." << std::endl;
        exit(EXIT_FAILURE);
    }
    if(!isNumeric(bufferArg)) {
        std::cerr << "ac: Error requires integer buffer argument." << std::endl;
        exit(EXIT_FAILURE);
    }
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
    if(dynamic_buffer < 2 || dynamic_buffer > 32) {
        std::cerr << "ac: Buffer must be valid from rang 3-32" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::VideoCapture cap;
    Writer writer;
    int frameCount = 0, screenshot_count = 1;
    double currentFPS = 0.0;
    auto lastFPSUpdate = std::chrono::steady_clock::now();

    if(camera_mode && silent == true) {
        std::cerr << "ac: Error silent mode requires video input." << std::endl;
        exit(EXIT_FAILURE);
    }

    if(silent == true && output_filename.empty()) {
        std::cerr << "ac: Error silent mode requires video to output." << std::endl;
        exit(EXIT_FAILURE);
    }

    try {
        if(silent == true)
            signal(SIGINT, signalHandler);
        if(camera_mode) {
#ifdef __linux__
            cap.open(camera_index, cv::CAP_V4L2);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, cres.width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, cres.height);
            cap.set(cv::CAP_PROP_FPS, output_fps);
            if(expose) 
                cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1); 
#else
            cap.open(camera_index);
#endif
        }
        else {
            cap.open(inputArg);
            if(cap.isOpened()) {
                cap.set(cv::CAP_PROP_POS_FRAMES, start_pos);
            }
            std::cout << "ac: Opened: "<<  inputArg << std::endl;
        }
        if (!cap.isOpened()) return -1;
        if(camera_mode == true) {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, cres.width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, cres.height);
            cap.set(cv::CAP_PROP_FPS, output_fps);
        }
        unsigned long total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        if(jump_pos != 0) {
            cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>((jump_pos * fps)));
        }
        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "ac: Resolution: " << width << "x" << height << " @ " << fps << " fps " << std::endl;
        ac_gpu::DynamicFrameBuffer buffer(dynamic_buffer);
        CHECK_CUDA(cudaMalloc(&d_ptrList, buffer.arraySize * sizeof(unsigned char*)));
        cv::cuda::GpuMat gpuWorkingBuffer(height, width, CV_8UC4);
        cv::cuda::GpuMat gpuDisplayBuffer(height, width, CV_8UC4);
        cv::Mat frame, rgba_out(height, width, CV_8UC4);
        if(silent == false) {
            cv::namedWindow("filter", cv::WINDOW_NORMAL);
            if(vres.width != 0 && vres.height != 0)
                cv::resizeWindow("filter", vres.width, vres.height);
            else
                cv::resizeWindow("filter", width, height);
            cv::moveWindow("filter", 0, 0);
        }
        if(!output_filename.empty()) {
            writer.open_ts(output_filename, width, height, fps, output_crf.c_str());
            std::cout << "ac: Opened: " << output_filename << " for writing " << width << "x" << height << " @ " << fps << " fps CRF: " << output_crf << std::endl;
        }
        auto frame_duration = std::chrono::milliseconds((int)(1000.0 / fps));
        auto app_start_time = std::chrono::steady_clock::now();
        while (1) {
            auto start_time = std::chrono::steady_clock::now();
            if (!cap.read(frame)) {
                if(repeat)  {
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    continue;
                } 
                break; 
            }
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
                unsigned long frame_count = writer.get_frame_count();
                double percentage = (static_cast<double>(frame_count) / total_frames) * 100.0;
                static double old_count = percentage;
                if(silent == true && static_cast<int>(old_count) < static_cast<int>(percentage)) {
                    std::cout << "acidcam-gpu writing [" << frame_count << "/" << total_frames << "] - " << std::fixed << std::setprecision(1) << percentage << "%\n";
                    old_count = percentage;
                }
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
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - app_start_time).count();
            int mins = (elapsed_ms / 60000);
            int secs = (elapsed_ms % 60000) / 1000; 

            static uint64_t frame_counter = 0;
            frame_counter++;

            if(!output_filename.empty()) {
                double time_elapsed = static_cast<double>(frame_counter) / fps;
                mins = static_cast<int>(time_elapsed / 60);
                secs = static_cast<int>(time_elapsed) % 60;
            }
           
            if (silent == false && (tick_count == 1) || (++tick % tick_count == 0)) {      
                if(show_hud) {  
                    std::string text = std::format("Acid Cam GPU - Time: {:02}:{:02} | FPS: {}", mins, secs, (int)currentFPS);
                    cv::putText(frame, text, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                    cv::putText(frame, device_text, cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(55,255,255), 2);
                }
                cv::imshow("filter",frame);
            }

            if(silent == false) {
                int key = cv::waitKey(1);
                if (key == 27) break; 
                else if (key == 'z'|| key == 'Z') {
                    auto t = std::time(nullptr);
                    auto tm = *std::localtime(&t);
                    std::ostringstream oss;
                    oss << "acidcam-gpu_" << width << "x" << height << " - " << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".png";
                    std::string outstr = oss.str();
                    std::string out = "acidcam-gpu_" + oss.str() + "-" + std::to_string(screenshot_count++) + ".png";
                    cv::imwrite(out, frame);
                    std::cout << "Saved: " << out << std::endl;
                }
                else if (key == 't') {
                    tally += std::to_string(filter_index) + ", ";
                }
                else if (key == 'w' || key == 'W') { 
                    if (filter_index > 0) {
                        filter_index--;
                        vlist.clear();
                        vlist.emplace_back(ac_gpu::Filter{filter_index, ac_gpu::filters[filter_index].name});
                        filtersChanged = true;
                        std::cout << "Filter: " << ac_gpu::filters[filter_index].name << " (" << filter_index << ")" << std::endl;
                    }
                }
                else if (key == 's' || key == 'S') { 
                    if (filter_index < ac_gpu::AC_FILTER_MAX - 1) {
                        filter_index++;
                        vlist.clear();
                        vlist.emplace_back(ac_gpu::Filter{filter_index, ac_gpu::filters[filter_index].name});
                        filtersChanged = true;
                        std::cout << "Filter: " << ac_gpu::filters[filter_index].name << " (" << filter_index << ")" << std::endl;
                    }
                } 
            }
            if(output_filename.empty() && time_over != 0 && (elapsed_ms / 1000) >= time_over) {
                std::cout << "ac: Time duration reached. Exiting..." << std::endl;
                break;
            } 
            if (time_over != 0 && !output_filename.empty() && writer.is_open()) {
                auto time_elapsed = static_cast<double>(writer.get_frame_count()) / fps;
                if (time_elapsed >= time_over) {
                    std::cout << "ac: Time duration reached exiting..." << std::endl;
                    break;
                }
            }



            if(camera_mode ==false && tick_count == 1) {
                auto end_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                if (elapsed.count() < frame_duration.count()) std::this_thread::sleep_for(std::chrono::milliseconds(frame_duration.count() - elapsed.count()));
            }
        } 
       
        if(!tally.empty()) 
            std::cout << "Total tally: " << tally << "\n";
    }
    catch (const Interrupt &)  {
        std::cout << "Interrupt called. Stopping..." << std::endl;
    }
    catch(std::exception &e) { 
        std::cerr << e.what() << std::endl; 
    }

    if(!output_filename.empty()) {
        double total_secs = static_cast<double>(writer.get_frame_count()) / fps;
        int final_mins = static_cast<int>(total_secs / 60);
        int final_secs = static_cast<int>(total_secs) % 60;
        std::cout << "Wrote: " << output_filename << " " << writer.get_frame_count() << " frames " << std::format("{:02}:{:02}", final_mins, final_secs) << std::endl;
    }

    if (d_filterList)
            CHECK_CUDA(cudaFree(d_filterList));
    CHECK_CUDA(cudaFree(d_ptrList));    
    if(!output_filename.empty()) writer.close();
    cap.release(); 
    return 0;
}